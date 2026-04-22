"""
demo.py — Hybrid DQN + Pressure-Based Traffic Controller
=========================================================

Display layout:  [ CAM 2: Intersection Overview ] [ Dashboard ]

Control architecture:
  - Pressure per arm = queue_length + 2 * avg_wait_time
  - Rule-based controller selects arm by max pressure + anti-starvation
  - DQN assists timing: action 0=keep, 1=allow-switch
  - Emergency: strict PRE_CLEAR → EMERGENCY → RECOVERY state machine
  - Fallback: FIXED_TIME arm rotation when sensor confidence is low

Usage:
    python demo.py
"""

import carla
import random
import math
import itertools
import numpy as np
import cv2
import threading
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

from signal_manager    import SignalManager
from controller        import PressureController, YELLOW_TICKS
from emergency_handler import EmergencyHandler, EmergencyState
from waiting_time      import IntersectionWaitingTimeTracker
from fallback          import FallbackController, ControlMode
from ground_sensors    import IntersectionGroundSensors

import time

# ── CARLA connection ──────────────────────────────────────────────────────────
TARGET_MAP = 'Town03'

print("Connecting to CARLA on localhost:2000 ...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    client.get_server_version()
except Exception as e:
    print(f"\n[ERROR] Cannot reach CARLA server: {e}")
    raise SystemExit(1)

print("Connected. Waiting for CARLA to be ready...")
time.sleep(5)

def get_world_with_retry(max_attempts=8, sleep_s=4):
    for attempt in range(1, max_attempts + 1):
        try:
            return client.get_world()
        except RuntimeError as e:
            print(f"[WARN] get_world timeout ({attempt}/{max_attempts}): {e}")
            time.sleep(sleep_s)
    print("[ERROR] CARLA world is not ready after multiple retries.")
    raise SystemExit(1)

world = get_world_with_retry()
current_map = world.get_map().name
print("Active map:", current_map)

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode   = False
world.apply_settings(settings)

def safe_world_tick(max_retries=3, label="tick", sleep_s=0.5):
    for attempt in range(1, max_retries + 1):
        try:
            world.tick()
            return True
        except RuntimeError as e:
            print(f"[WARN] {label} failed ({attempt}/{max_retries}): {e}")
            time.sleep(sleep_s)
    return False

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
print("Connected to CARLA.")

EMERGENCY_KW = ['ambulance', 'firetruck', 'police']

NUM_INT = 1

# ── Intersection detection ────────────────────────────────────────────────────
def group_lights(lights, threshold=45):
    groups = []
    for light in lights:
        loc   = light.get_location()
        added = False
        for group in groups:
            if loc.distance(group[0].get_location()) < threshold:
                group.append(light); added = True; break
        if not added:
            groups.append([light])
    return groups

traffic_lights = world.get_actors().filter("traffic.traffic_light")
groups         = group_lights(traffic_lights, threshold=45)

quad_groups  = [g for g in groups if len(g) >= 4]
valid_groups = quad_groups if quad_groups else [g for g in groups if len(g) >= 3]

PREFERRED_CENTRE = carla.Location(x=0, y=0, z=0)
valid_groups.sort(key=lambda g: sum(l.get_location().distance(PREFERRED_CENTRE)
                                    for l in g) / len(g))
intersections = valid_groups[:NUM_INT]

intersection_centers = []
for group in intersections:
    x = sum(l.get_location().x for l in group) / len(group)
    y = sum(l.get_location().y for l in group) / len(group)
    z = sum(l.get_location().z for l in group) / len(group)
    intersection_centers.append(carla.Location(x=x, y=y, z=z))

center = intersection_centers[0]

ROI_RADIUS          = 35
ROAD_DETECTION_DIST = 75

wait_trackers = [IntersectionWaitingTimeTracker(i + 1, ROI_RADIUS, intersection_centers[i])
                 for i in range(NUM_INT)]

# ground_sensor is initialised below, after arm_directions are derived from light positions

# ── Fallback controller (YOLO confidence → FIXED_TIME mode) ──────────────────
fallback_ctrl        = FallbackController(intersection_id=1)
yolo_conf            = 0.0
control_conf         = 0.95
control_count        = 0
fallback_event_count = 0
emergency_event_count = 0

SIM_LC_INTERVAL = 3600
SIM_LC_DURATION = 160
_sim_lc_timer   = 0
_sim_lc_active  = False
_sim_lc_dur_ctr = 0

# ── Vehicle spawning ──────────────────────────────────────────────────────────
blueprints      = world.get_blueprint_library().filter("vehicle.*")
EMERGENCY_BP_KW = ['ambulance', 'firetruck', 'police']
car_bps         = [bp for bp in blueprints
                   if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                   and not any(kw in bp.id.lower() for kw in EMERGENCY_BP_KW)]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

GLOBAL_SPAWN_LIMIT  = 300
NEARBY_SPAWN_LIMIT  = 180
REPLENISH_TARGET    = 120
REPLENISH_RADIUS_M  = 180

if 'Town03' in current_map:
    # Town03HD_Opt is heavier than Town03; keep actor count modest to avoid
    # simulator stalls/timeouts on lower-end GPUs.
    GLOBAL_SPAWN_LIMIT = 50
    NEARBY_SPAWN_LIMIT = 20
    REPLENISH_TARGET   = 35
    REPLENISH_RADIUS_M = 140
    print("[PERF] Town03 detected: using reduced traffic load profile.")

traffic_manager.set_global_distance_to_leading_vehicle(3.0)
traffic_manager.global_percentage_speed_difference(-15)

for sp in spawn_points[:GLOBAL_SPAWN_LIMIT]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True); vehicles.append(v)

nearby_sps = [sp for sp in spawn_points if sp.location.distance(center) < 80]
n = 0
for sp in nearby_sps:
    if n >= NEARBY_SPAWN_LIMIT: break
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True); vehicles.append(v); n += 1

print(f"Spawned {len(vehicles)} vehicles.")
for i in range(6):
    if not safe_world_tick(max_retries=2, label=f"startup tick {i+1}", sleep_s=0.25):
        print("[ERROR] Simulator not responding during startup ticks. "
              "Please restart CarlaUE4 and run demo again.")
        raise SystemExit(1)

# ── Camera (CAM 2: overhead intersection overview only) ───────────────────────
OVERVIEW_CAM_IDX = 0

overview_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
overview_cam_bp.set_attribute('image_size_x', '640')
overview_cam_bp.set_attribute('image_size_y', '640')
overview_cam_bp.set_attribute('fov', '110')
overview_cam_bp.set_attribute('sensor_tick', '0.05')

carla_map = world.get_map()

# Emergency spawn candidates: 30–52 m (inside detection range of 55 m)
_approach_spawn_candidates = []
for _sp in world.get_map().get_spawn_points():
    if 30 < _sp.location.distance(center) < 52:
        _approach_spawn_candidates.append(_sp)
_approach_spawn_candidates.sort(key=lambda sp: sp.location.distance(center), reverse=True)
print(f"Emergency spawn candidates: {len(_approach_spawn_candidates)}")

overview_cam_transform = carla.Transform(
    carla.Location(x=center.x, y=center.y, z=55),
    carla.Rotation(pitch=-90, yaw=0, roll=0)
)

camera_transforms = [overview_cam_transform]

latest_frames = {OVERVIEW_CAM_IDX: None}
frame_locks   = {OVERVIEW_CAM_IDX: threading.Lock()}
frame_counts  = {OVERVIEW_CAM_IDX: 0}

# ── Assign lights uniquely to N/S/E/W arms ───────────────────────────────────
def _light_angle_from_center(light):
    loc = light.get_location()
    dx  = loc.x - center.x
    dy  = loc.y - center.y
    return math.degrees(math.atan2(dy, dx))

def _assign_lights_to_arms(lights):
    if not lights:
        return {}
    # Light angles at corners: N is Top-Left, S is Bottom-Right, E is Top-Right, W is Bottom-Left
    # User Mapping: 1->C(E), 2->E(S), 3->G(W), 4->A(N)
    # Light 1=-45, 2=45, 3=135, 4=-135
    targets = {'N': -135.0, 'S': 45.0, 'E': -45.0, 'W': 135.0}
    arms    = ['N', 'S', 'E', 'W']
    candidates = sorted(lights, key=lambda l: l.get_location().distance(center))[:4]
    if len(candidates) < 4:
        candidates = list(lights)
    if len(candidates) >= 4:
        best_cost, best_map = float('inf'), None
        for perm in itertools.permutations(candidates[:4], 4):
            cost = sum(abs((_light_angle_from_center(l) - targets[a] + 180) % 360 - 180)
                       for a, l in zip(arms, perm))
            if cost < best_cost:
                best_cost = cost
                best_map  = {a: l for a, l in zip(arms, perm)}
        if best_map is not None:
            return best_map
    # greedy fallback
    remaining, assigned = list(candidates), {}
    for arm in arms:
        if not remaining: break
        best = min(remaining,
                   key=lambda l: abs((_light_angle_from_center(l) - targets[arm] + 180) % 360 - 180))
        assigned[arm] = best
        remaining.remove(best)
    return assigned

main_intersection_lights = list(intersections[0])
lane_lights = _assign_lights_to_arms(main_intersection_lights)
print("Lane-light mapping:", {arm: lane_lights[arm].id for arm in lane_lights})

# ── Assign ALL intersection lights to arms (not just 1 per arm) ───────────────
# Pure angle-based assignment fails on irregular intersections (Town03 puts
# S arm's light at an angle closer to another arm → S ends up with 0 lights,
# vehicles ignore signals, emergency vehicles drive straight through on RED).
#
# Two-phase strategy:
#   1. Seed each arm with its permutation-optimal canonical light from lane_lights.
#      This is correct by construction — the permutation search guarantees it.
#   2. Assign any remaining lights to the nearest arm by angle (multi-lane support).
_arm_angle_targets = {'N': -135.0, 'S': 45.0, 'E': -45.0, 'W': 135.0}
arm_light_groups   = {'N': [], 'S': [], 'E': [], 'W': []}
_canonical_ids     = set()

for _arm, _clight in lane_lights.items():
    arm_light_groups[_arm].append(_clight)
    _canonical_ids.add(_clight.id)

for _light in main_intersection_lights:
    if _light.id in _canonical_ids:
        continue
    _angle    = _light_angle_from_center(_light)
    _best_arm = min(_arm_angle_targets,
                    key=lambda a: abs((_angle - _arm_angle_targets[a] + 180) % 360 - 180))
    arm_light_groups[_best_arm].append(_light)

for _arm in ['N', 'S', 'E', 'W']:
    if not arm_light_groups[_arm]:
        print(f"[WARN] arm_light_groups: {_arm} still has no lights after seeding — check lane_lights.")
print("Arm light groups:", {arm: len(arm_light_groups[arm]) for arm in arm_light_groups})

# Derive actual road directions from light positions for ground sensors
arm_directions = {}
for _arm, _lights in arm_light_groups.items():
    if _lights:
        _angles = [_light_angle_from_center(_l) for _l in _lights]
        arm_directions[_arm] = (sum(_angles) / len(_angles)) % 360
    else:
        arm_directions[_arm] = {'N': 0, 'S': 180, 'E': 90, 'W': 270}[_arm]

ground_sensor = IntersectionGroundSensors(
    intersection_id=1, intersection_center=center, arm_directions=arm_directions)

# ── Signal manager (enforces single-green invariant) ─────────────────────────
signal_manager = SignalManager(arm_light_groups)

# ── Pressure controller + emergency handler ───────────────────────────────────
pressure_ctrl    = PressureController()
emergency_handler = EmergencyHandler(center, roi_radius=ROI_RADIUS + 20)

# ── Fixed-time cycling state (used in FALLBACK mode) ─────────────────────────
FIXED_ARM_ORDER    = ['N', 'S', 'E', 'W']
FIXED_GREEN_TICKS  = 600    # 30 s per arm in FALLBACK
fallback_arm_ticks = 0

# ── Yellow-transition state (shared across all non-emergency modes) ───────────
yellow_active  = False
yellow_arm     = None      # arm currently showing yellow
yellow_counter = 0
pending_arm    = None      # arm to go GREEN after yellow completes

# ── Camera spawn + warm-up ───────────────────────────────────────────────────
overview_cam = world.spawn_actor(overview_cam_bp, overview_cam_transform)
cameras      = [overview_cam]

def on_image(image, idx):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb   = array[:, :, :3][:, :, ::-1].copy()
    with frame_locks[idx]:
        latest_frames[idx] = rgb
        frame_counts[idx] += 1

overview_cam.listen(lambda img: on_image(img, OVERVIEW_CAM_IDX))

for _light in intersections[0]:
    _light.freeze(True)
    _light.set_state(carla.TrafficLightState.Red)  # all RED before we assign green
print("Traffic lights frozen — manual control active.")

print("Cameras ready. Warming up...")
for i in range(8):
    if not safe_world_tick(max_retries=4, label=f"camera warm-up tick {i+1}", sleep_s=1.0):
        print("[WARN] Simulator slow during camera warm-up — continuing anyway.")
        break

frame_ready = False
for i in range(20):
    if frame_counts[OVERVIEW_CAM_IDX] > 0:
        frame_ready = True
        break
    if not safe_world_tick(max_retries=4, label=f"first-frame wait {i+1}", sleep_s=0.5):
        break

if not frame_ready:
    print("[WARN] Camera frame not received in time; using placeholder until stream recovers.")
    with frame_locks[OVERVIEW_CAM_IDX]:
        latest_frames[OVERVIEW_CAM_IDX] = np.zeros((640, 640, 3), dtype=np.uint8)

print("Camera warm-up complete.")

# Startup: pick the most-demanded arm as initial green
_startup_actors = list(world.get_actors().filter("vehicle.*"))
wait_trackers[0].update(_startup_actors, center, 0)
_startup_sensor = ground_sensor.update(_startup_actors, [])
_startup_counts = _startup_sensor['arm_counts']
initial_arm = max(['N', 'S', 'E', 'W'], key=lambda a: _startup_counts.get(a, 0))
pressure_ctrl.current_arm = initial_arm
signal_manager.set_arm_green(initial_arm)
print(f"Initial green arm: {initial_arm}")

# ── Camera projection helpers ─────────────────────────────────────────────────
_CAM_W, _CAM_H = 640, 640

_OV_FOV   = 110
_ov_focal = _CAM_W / (2.0 * math.tan(math.radians(_OV_FOV / 2.0)))
OV_K = np.array([[_ov_focal, 0, _CAM_W / 2.0],
                 [0, _ov_focal, _CAM_H / 2.0],
                 [0, 0,         1.0          ]], dtype=np.float64)

def _world_to_cam(transform):
    cy = math.cos(math.radians(transform.rotation.yaw))
    sy = math.sin(math.radians(transform.rotation.yaw))
    cr = math.cos(math.radians(transform.rotation.roll))
    sr = math.sin(math.radians(transform.rotation.roll))
    cp = math.cos(math.radians(transform.rotation.pitch))
    sp = math.sin(math.radians(transform.rotation.pitch))
    tx, ty, tz = transform.location.x, transform.location.y, transform.location.z
    c2w = np.array([
        [cp*cy,  cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr, tx],
        [cp*sy,  sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr, ty],
        [sp,    -cp*sr,             cp*cr,             tz],
        [0,      0,                 0,                  1]
    ], dtype=np.float64)
    return np.linalg.inv(c2w)

def project_actor(actor, cam_idx):
    loc = actor.get_location()
    w2c = _world_to_cam(camera_transforms[cam_idx])
    p   = w2c @ np.array([loc.x, loc.y, loc.z, 1.0])
    if p[0] <= 0:
        return None
    u = int(OV_K[0, 0] * p[1] / p[0] + OV_K[0, 2])
    v = int(OV_K[1, 1] * (-p[2]) / p[0] + OV_K[1, 2])
    if not (0 <= u < _CAM_W and 0 <= v < _CAM_H):
        return None
    return u, v, p[0]

# ── Siren ─────────────────────────────────────────────────────────────────────
_siren_on = False

def _siren_loop():
    while _siren_on:
        if _HAS_WINSOUND:
            winsound.Beep(1400, 180)
            winsound.Beep(900,  180)
        else:
            time.sleep(0.36)

def start_siren():
    global _siren_on
    if not _siren_on:
        _siren_on = True
        threading.Thread(target=_siren_loop, daemon=True).start()

def stop_siren():
    global _siren_on
    _siren_on = False

# ── Shared display constants ──────────────────────────────────────────────────
PHASE_COLORS = {0: (0, 255, 0), 1: (0, 215, 255), 2: (0, 0, 255)}
PHASE_NAMES  = {0: 'GREEN',     1: 'YELLOW',       2: 'RED'}

ARM_COLORS = {'N': (0, 255, 100), 'S': (0, 200, 255),
              'E': (255, 180, 0), 'W': (200, 100, 255)}

# ── CAM 2: Intersection overview ──────────────────────────────────────────────
def draw_overview_annotated(display_phase, system_mode, vehicles_in_roi,
                            current_arm, pressures,
                            emg_vehicle=None):
    """
    Grab overhead intersection frame and annotate with:
      - Signal-light dots (coloured by actual state)
      - Emergency vehicle box
      - Per-arm pressure bars
      - Current green arm + mode overlay
    Returns annotated_bgr_760x760 or None.
    """
    with frame_locks[OVERVIEW_CAM_IDX]:
        frame = latest_frames[OVERVIEW_CAM_IDX]
    if frame is None:
        return None

    annotated = frame[:, :, ::-1].copy()

    # HUD bar
    ph_color = PHASE_COLORS[display_phase]
    ph_name  = PHASE_NAMES[display_phase]
    overlay  = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (640, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    cv2.putText(annotated, "CAM 2  |  INTERSECTION OVERVIEW  (SYSTEM RESPONSE)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 100), 2)
    cv2.putText(annotated, f"Vehicles in zone: {vehicles_in_roi}   |   Signal: {ph_name}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
    cv2.circle(annotated, (580, 28), 17, ph_color, -1)
    cv2.circle(annotated, (580, 28), 17, (255, 255, 255), 1)
    cv2.putText(annotated, ph_name, (492, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ph_color, 2)

    # Traffic-light dots at actual CARLA positions
    for light in intersections[0]:
        pt = project_actor(light, OVERVIEW_CAM_IDX)
        if pt is None: continue
        u, v_px, _ = pt
        state = light.get_state()
        dot_col = ((0, 230, 0) if state == carla.TrafficLightState.Green else
                   (0, 230, 230) if state == carla.TrafficLightState.Yellow else
                   (0, 0, 230))
        cv2.circle(annotated, (u, v_px), 10, dot_col, -1)
        cv2.circle(annotated, (u, v_px), 10, (255, 255, 255), 2)

    # Emergency vehicle box
    if emg_vehicle and emg_vehicle.is_alive:
        name = emg_vehicle.type_id.split('.')[-1].upper()
        pt   = project_actor(emg_vehicle, OVERVIEW_CAM_IDX)
        if pt is not None:
            u, v_px, _ = pt
            hw, hh = 30, 20
            x1 = max(0, u - hw);       y1 = max(0, v_px - hh)
            x2 = min(_CAM_W-1, u + hw); y2 = min(_CAM_H-1, v_px + hh)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(annotated, (x1, y1), (x2, y1 + 20), (0, 0, 200), -1)
            cv2.putText(annotated, f"EMG: {name}", (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Per-arm pressure bars (bottom-left block)
    if pressures:
        max_p = max(pressures.values(), default=1.0) or 1.0
        bx, by = 8, 520
        for arm in ['N', 'S', 'E', 'W']:
            p   = pressures.get(arm, 0.0)
            col = ARM_COLORS.get(arm, (180, 180, 180))
            if arm == current_arm:
                cv2.rectangle(annotated, (bx - 2, by - 2), (bx + 112, by + 16), col, 1)
            bw = int(108 * p / max_p)
            cv2.rectangle(annotated, (bx, by), (bx + 108, by + 12), (30, 30, 30), -1)
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + 12), col, -1)
            cv2.putText(annotated, f"{arm}:{p:.1f}", (bx + 2, by + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 220, 220), 1)
            by += 18

    # Current arm label
    arm_col = ARM_COLORS.get(current_arm, (255, 255, 255))
    cv2.putText(annotated, f"GREEN: {current_arm}",
                (8, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.55, arm_col, 2)

    # Mode banner
    _MODE_BANNERS = {
        EmergencyState.PRE_CLEAR: ("  PRE-CLEARANCE — all RED", (100, 180, 255), (60, 40, 0)),
        EmergencyState.EMERGENCY: ("  EMERGENCY OVERRIDE",       (80, 80, 255),  (0, 0, 160)),
        EmergencyState.RECOVERY:  ("  RECOVERY — draining backlog", (0, 200, 230), (0, 45, 70)),
        'FALLBACK':               ("  FIXED-TIME FALLBACK",      (0, 180, 255),  (0, 40, 80)),
    }
    txt, col, bg = _MODE_BANNERS.get(
        system_mode, ("  PRESSURE CONTROL", (70, 200, 70), (0, 40, 0))
    )
    cv2.rectangle(annotated, (0, 596), (640, 640), bg, -1)
    cv2.putText(annotated, txt, (8, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)

    annotated = cv2.resize(annotated, (760, 760), interpolation=cv2.INTER_LINEAR)
    return annotated


# ── Helper functions ──────────────────────────────────────────────────────────
def compute_gt(c):
    actors = world.get_actors().filter("vehicle.*")
    cnt, ss = 0, 0.0
    for v in actors:
        if v.get_location().distance(c) < ROI_RADIUS:
            cnt += 1
            vel  = v.get_velocity()
            ss  += math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return cnt, (ss / cnt if cnt > 0 else 0.0)

def get_arm_counts(actors, c):
    n = s = e = w = 0
    for v in actors:
        loc = v.get_location()
        if loc.distance(c) < ROI_RADIUS:
            dx, dy = loc.x - c.x, loc.y - c.y
            if abs(dx) >= abs(dy):
                if dx >= 0: e += 1
                else:       w += 1
            else:
                if dy >= 0: s += 1
                else:       n += 1
    return n, s, e, w

def is_emergency_vehicle(v):
    role = str(v.attributes.get('role_name', '')).lower()
    return role == 'emergency' or any(kw in v.type_id.lower() for kw in EMERGENCY_KW)

def _is_visible_in_overview(v):
    return project_actor(v, OVERVIEW_CAM_IDX) is not None

def _spawn_point_arm(sp_loc):
    """Return which arm (N/S/E/W) a spawn-point location belongs to."""
    dx = sp_loc.x - center.x
    dy = sp_loc.y - center.y
    if abs(dx) >= abs(dy):
        return 'E' if dx >= 0 else 'W'
    return 'S' if dy >= 0 else 'N'


# ── Emergency spawner ──────────────────────────────────────────────────────────
emergency_vehicle = None

def _safe_destroy_actor(actor, tracked_list=None):
    if actor is None: return False
    try:
        if not actor.is_alive: return False
    except RuntimeError:
        return False
    try:
        if world.get_actor(actor.id) is None: return False
    except RuntimeError:
        return False
    try:
        actor.destroy()
        if tracked_list is not None:
            try: tracked_list.remove(actor)
            except ValueError: pass
        return True
    except RuntimeError:
        return False

def _clear_spawn_point(sp, radius=5.0):
    for other in world.get_actors().filter("vehicle.*"):
        if not other.is_alive: continue
        if is_emergency_vehicle(other): continue   # never destroy the emergency actor
        if other.get_location().distance(sp.location) < radius:
            _safe_destroy_actor(other, tracked_list=vehicles)

def _find_emergency_spawn_transform():
    """
    Find a road waypoint 30-50 m from the intersection center on any arm.
    Uses carla_map.get_waypoint() so the vehicle always lands on a valid
    drivable road — fixes the 'snaps to 134 m' bug caused by off-road
    spawn-point coordinates.
    Returns a carla.Transform or None.
    """
    _green_arm = signal_manager.current_green_arm
    # Try non-green arms first so vehicle stops at RED before EMERGENCY_ACTIVE
    arms = sorted(arm_directions.keys(),
                  key=lambda a: 0 if a != _green_arm else 1)

    for arm in arms:
        angle_rad = math.radians(arm_directions[arm])
        for dist in [40, 35, 45, 30, 48]:
            tx = center.x + math.cos(angle_rad) * dist
            ty = center.y + math.sin(angle_rad) * dist
            loc = carla.Location(x=tx, y=ty, z=0.0)
            wp = carla_map.get_waypoint(loc, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            actual = wp.transform.location.distance(center)
            if 15.0 < actual < 55.0:
                tf = wp.transform
                tf.location.z += 0.3   # slight lift to avoid road-surface clipping
                print(f"[EMERGENCY] Waypoint found: arm={arm} dist={actual:.0f}m")
                return tf
    return None


def spawn_emergency():
    lib          = world.get_blueprint_library()
    ambulance_bps = list(lib.filter("vehicle.*ambulance*"))
    fallback_bps  = (list(lib.filter("vehicle.*firetruck*")) +
                     list(lib.filter("vehicle.*police*")))
    generic_bps   = [bp for bp in lib.filter("vehicle.*")
                     if int(bp.get_attribute('number_of_wheels').as_int()) == 4]
    bps = ambulance_bps if ambulance_bps else fallback_bps
    if not bps: bps = generic_bps
    if not bps:
        print("[EMERGENCY] No blueprints — cannot spawn.")
        return None

    bp_choice = random.choice(bps)
    if bp_choice.has_attribute('role_name'):
        bp_choice.set_attribute('role_name', 'emergency')

    # ── Waypoint-based spawn (reliable road position at correct distance) ──────
    spawn_tf = _find_emergency_spawn_transform()
    if spawn_tf is None:
        print("[EMERGENCY] No valid road waypoint found near intersection.")
        return None

    _clear_spawn_point(spawn_tf)   # clear blocking vehicles at waypoint
    v = world.try_spawn_actor(bp_choice, spawn_tf)
    if not v:
        print("[EMERGENCY] Spawn actor failed at waypoint.")
        return None

    loc         = v.get_location()
    actual_dist = loc.distance(center)
    if actual_dist > 55:
        print(f"[EMERGENCY] Spawn rejected: {v.type_id} landed at "
              f"({loc.x:.1f},{loc.y:.1f}) dist={actual_dist:.0f}m — outside detection range.")
        v.destroy()
        return None
    print(f"[EMERGENCY] Spawned {v.type_id} at ({loc.x:.1f},{loc.y:.1f}) "
          f"dist={actual_dist:.0f}m")

    safe_world_tick(max_retries=2, label="emergency spawn sync", sleep_s=0.2)

    v.set_autopilot(True)
    # Respect traffic lights — vehicle stops at RED during PRE_CLEAR and proceeds
    # when EMERGENCY_ACTIVE gives its arm GREEN.
    traffic_manager.ignore_lights_percentage(v, 0)
    traffic_manager.ignore_signs_percentage(v, 100)
    traffic_manager.ignore_vehicles_percentage(v, 0)
    traffic_manager.force_lane_change(v, False)
    traffic_manager.auto_lane_change(v, False)
    traffic_manager.vehicle_percentage_speed_difference(v, -60)
    traffic_manager.distance_to_leading_vehicle(v, 1.5)

    # Clear only the immediate spawn area (≤8 m) — not the whole approach road.
    # The old 70 m forward-sweep was destroying 10+ regular vehicles unnecessarily.
    emg_loc = v.get_location()
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == v.id or not other.is_alive: continue
        if emg_loc.distance(other.get_location()) <= 8.0:
            _safe_destroy_actor(other, tracked_list=vehicles)

    return v

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASH_W   = 420
HIST_LEN = 120

def draw_dashboard(display_phase, yolo_count, stats, system_mode,
                   wait_history, tick_count, current_arm, pressures,
                   yolo_conf=0.0, emg_count=0, fb_count=0):
    H, W  = 760, DASH_W
    panel = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)

    PH_TXT = ['GREEN', 'YELLOW', 'RED']
    PH_COL = [(0, 210, 0), (0, 210, 230), (0, 0, 230)]

    # Title
    cv2.rectangle(panel, (0, 0), (W, 54), (28, 28, 50), -1)
    cv2.putText(panel, "TRAFFIC INTELLIGENCE SYSTEM",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 1)
    cv2.putText(panel, f"Tick {tick_count:>6d}   |   Pressure-based control",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 160), 1)
    cv2.line(panel, (0, 54), (W, 54), (60, 60, 100), 1)

    sy = 58

    # Mode badge
    _MODE_BADGE = {
        EmergencyState.PRE_CLEAR: ("PRE-CLEARANCE",          (255, 120, 0),   (100, 50, 0)),
        EmergencyState.EMERGENCY: ("EMERGENCY OVERRIDE",      (120, 120, 255), (50,  0, 100)),
        EmergencyState.RECOVERY:  ("POST-EMERGENCY RECOVERY", (0, 200, 230),   (0, 45, 70)),
        'FALLBACK':               ("FIXED-TIME FALLBACK",     (0, 180, 255),   (0, 40, 80)),
    }
    mtxt, mcol, mbg = _MODE_BADGE.get(
        system_mode, ("PRESSURE CONTROL", (0, 230, 110), (0, 50, 20))
    )
    cv2.rectangle(panel, (4, sy), (W-4, sy + 24), mbg, -1)
    cv2.putText(panel, f"MODE: {mtxt}", (8, sy + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mcol, 1)
    sy += 28

    # Signal circles
    for pi in range(3):
        cx  = 28 + pi * 44
        col = PH_COL[pi] if pi == display_phase else (28, 28, 28)
        cv2.circle(panel, (cx, sy + 24), 18, col, -1)
        if pi == display_phase:
            cv2.circle(panel, (cx, sy + 24), 18, (255, 255, 255), 1)
    cv2.putText(panel, PH_TXT[display_phase],
                (168, sy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PH_COL[display_phase], 2)

    # Current arm highlight
    arm_col = ARM_COLORS.get(current_arm, (200, 200, 200))
    cv2.putText(panel, f"GREEN ARM: {current_arm}",
                (8, sy + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, arm_col, 2)
    sy += 70

    # Per-arm pressure bars
    cv2.putText(panel, "ARM PRESSURES  (q + w/10)",
                (8, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 180), 1)
    sy += 8
    if pressures:
        max_p = max(pressures.values(), default=1.0) or 1.0
        for arm in ['N', 'S', 'E', 'W']:
            p   = pressures.get(arm, 0.0)
            col = ARM_COLORS.get(arm, (160, 160, 160))
            bw  = int((W - 80) * p / max_p)
            is_green = (arm == current_arm)
            lbl = f"{arm}{'*' if is_green else ' '}: {p:5.1f}"
            cv2.putText(panel, lbl, (6, sy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            cv2.rectangle(panel, (60, sy + 2), (W - 10, sy + 14), (35, 35, 50), -1)
            if bw > 0:
                cv2.rectangle(panel, (60, sy + 2), (60 + bw, sy + 14), col, -1)
            if is_green:
                cv2.rectangle(panel, (60, sy + 2), (W - 10, sy + 14), col, 1)
            sy += 18
    sy += 4

    # Stats rows
    q   = stats.get('queue_length', 0)
    aw  = stats.get('avg_waiting_time', 0.0)
    tp  = stats.get('throughput_vpm', 0.0)
    aw_col = (0, 70, 220) if aw > 15 else (0, 200, 100)
    cv2.line(panel, (0, sy), (W, sy), (50, 50, 80), 1)
    sy += 6
    for lbl, val, col in [
        ("Vehicles (sensors)", f"{yolo_count}",    (180, 180, 180)),
        ("Queue Length",       f"{q}",             (180, 180, 180)),
        ("Avg Wait",           f"{aw:.1f} s",      aw_col),
        ("Throughput",         f"{tp:.1f} vpm",    (150, 220, 150)),
    ]:
        cv2.putText(panel, lbl, (10, sy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 120, 150), 1)
        cv2.putText(panel, val, (W - 90, sy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)
        sy += 22

    # Queue bar
    cv2.line(panel, (0, sy), (W, sy), (50, 50, 80), 1)
    sy += 6
    fill = int((W - 20) * min(q, 20) / 20)
    cv2.rectangle(panel, (10, sy), (W - 10, sy + 12), (35, 35, 50), -1)
    if fill > 0:
        cv2.rectangle(panel, (10, sy), (10 + fill, sy + 12),
                      (0, 60, 210) if q > 10 else (0, 185, 80), -1)
    cv2.putText(panel, "QUEUE", (W - 55, sy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 100), 1)
    sy += 18

    # Sensor confidence
    conf_col = (0, 200, 80) if yolo_conf > 0.5 else (0, 180, 220) if yolo_conf > 0.35 else (0, 60, 220)
    cv2.putText(panel, f"Sensor Conf: {yolo_conf:.2f}",
                (10, sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.41, conf_col, 1)
    bw = int((W - 28) * max(0.0, min(1.0, yolo_conf)))
    cv2.rectangle(panel, (10, sy + 18), (W - 18, sy + 27), (40, 40, 55), -1)
    cv2.rectangle(panel, (10, sy + 18), (10 + bw, sy + 27), conf_col, -1)
    thresh_x = 10 + int((W - 28) * 0.35)
    cv2.line(panel, (thresh_x, sy + 16), (thresh_x, sy + 29), (0, 100, 200), 1)
    sy += 34

    # Event counters
    cv2.line(panel, (0, sy), (W, sy), (50, 50, 80), 1)
    sy += 4
    cv2.putText(panel, f"Emergency events: {emg_count}",
                (10, sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (150, 100, 255), 1)
    cv2.putText(panel, f"Fallback events:  {fb_count}",
                (10, sy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 160, 255), 1)
    sy += 36

    # Wait-time graph
    cv2.line(panel, (0, sy), (W, sy), (50, 50, 80), 1)
    sy += 4
    GT = sy + 14
    cv2.putText(panel, "AVG WAIT TIME (last 120 ticks)",
                (8, GT - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 130, 200), 1)
    GX0, GY0 = 8,   GT
    GX1, GY1 = W-8, min(H - 32, GT + 180)
    GH, GW   = GY1 - GY0, GX1 - GX0

    cv2.rectangle(panel, (GX0, GY0), (GX1, GY1), (22, 22, 38), -1)
    cv2.rectangle(panel, (GX0, GY0), (GX1, GY1), (55, 55, 85),  1)
    for gi in range(1, 4):
        gy = GY0 + int(GH * gi / 4)
        cv2.line(panel, (GX0, gy), (GX1, gy), (38, 38, 60), 1)

    hist    = wait_history[0]
    max_val = max(max(hist) if hist else 30, 30)
    if len(hist) >= 2:
        n   = len(hist)
        pts = [(GX0 + int(GW * i / (n - 1)),
                max(GY0, min(GY1, GY1 - int(GH * min(v, max_val) / max_val))))
               for i, v in enumerate(hist)]
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i-1], pts[i], (0, 220, 110), 2)

    cv2.putText(panel, f"{max_val:.0f}s",
                (GX0, GY0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 120), 1)
    cv2.putText(panel, "0s",
                (GX0, GY1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 120), 1)

    # Status bar
    is_emg = system_mode in (EmergencyState.PRE_CLEAR, EmergencyState.EMERGENCY)
    is_fb  = system_mode == 'FALLBACK'
    if is_emg:
        sb_bg, sb_col = (50, 0, 0),   (120, 80, 255)
        sb_txt = "EMERGENCY PRIORITY ACTIVE"
    elif is_fb:
        sb_bg, sb_col = (0, 35, 65),  (0, 160, 255)
        sb_txt = "FALLBACK: FIXED-TIME SAFE MODE"
    else:
        sb_bg, sb_col = (0, 28, 0),   (70, 200, 70)
        sb_txt = f"PRESSURE CONTROL  |  GREEN: {current_arm}"
    cv2.rectangle(panel, (0, 729), (W, 760), sb_bg, -1)
    cv2.putText(panel, sb_txt, (6, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.39, sb_col, 1)

    return panel

# ── Display window ─────────────────────────────────────────────────────────────
WIN_NAME = "Traffic Intelligence System — Pressure-Based Control"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 760 + DASH_W, 760)  # Overview | Dashboard

# ── Main-loop state ───────────────────────────────────────────────────────────
YOLO_INTERVAL      = 3
EMERGENCY_INTERVAL = 180
EMERGENCY_LIFETIME = 420
EMERGENCY_POST_GAP = 100

tick_count      = 0
emg_counter     = EMERGENCY_INTERVAL
emg_age         = 0
emg_post_gap    = 0
emg_stuck_ticks = 0

system_mode   = 'PRESSURE' # display string: 'PRESSURE'|'FALLBACK'|PRE_CLEAR|EMERGENCY|RECOVERY
display_phase = 0         # 0=green, 1=yellow, 2=red — for dashboard circles

overview_frame = None
wait_history   = [[] for _ in range(NUM_INT)]
pressures      = {arm: 0.0 for arm in ['N', 'S', 'E', 'W']}
latest_sensor_counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}

print(f"\nDemo running — Pressure-Based Traffic Controller.")
print(f"  CAM 2: Intersection overview  |  Dashboard: signal metrics")
print(f"  Press Q to quit.")
print(f"  Press E to manually trigger an emergency vehicle now.\n")

try:
    while True:
        if not safe_world_tick(max_retries=2,
                               label=f"main-loop tick {tick_count + 1}",
                               sleep_s=0.25):
            print("[ERROR] Simulator stopped responding in main loop.")
            break
        tick_count += 1

        # ── Emergency vehicle lifecycle ────────────────────────────────────────
        if emergency_vehicle and emergency_vehicle.is_alive:
            ev_vel   = emergency_vehicle.get_velocity()
            ev_speed = math.sqrt(ev_vel.x**2 + ev_vel.y**2 + ev_vel.z**2)
            if ev_speed < 0.5:
                emg_stuck_ticks += 1
                # Don't teleport during PRE_CLEAR — vehicle is correctly stopped
                # at RED. Firing at 20 ticks would move it into the intersection
                # zone (<8 m) and break detection. Only teleport after 80 ticks
                # (4 s) and only when not in PRE_CLEAR.
                _in_pre_clear = (emergency_handler.state == EmergencyState.PRE_CLEAR)
                if emg_stuck_ticks >= 80 and not _in_pre_clear:
                    wp = carla_map.get_waypoint(emergency_vehicle.get_location(),
                                                project_to_road=True,
                                                lane_type=carla.LaneType.Driving)
                    if wp:
                        nexts = wp.next(8.0)
                        if nexts:
                            tf = nexts[0].transform
                            emergency_vehicle.set_transform(tf)
                            safe_world_tick(max_retries=2,
                                            label="emergency unstuck sync",
                                            sleep_s=0.2)
                    emg_stuck_ticks = 0
            else:
                emg_stuck_ticks = 0
            emg_age += 1
            if emg_age >= EMERGENCY_LIFETIME:
                _safe_destroy_actor(emergency_vehicle, tracked_list=vehicles)
                emergency_vehicle = None
                emg_age = 0; emg_counter = 0; emg_post_gap = EMERGENCY_POST_GAP
        else:
            if emg_post_gap > 0:
                emg_post_gap -= 1
            else:
                emg_counter += 1
                if emg_counter >= EMERGENCY_INTERVAL:
                    emergency_vehicle = spawn_emergency()
                    emg_age = 0
                    emg_counter = 0 if emergency_vehicle else 0

        timestamp  = world.get_snapshot().timestamp.elapsed_seconds
        all_actors = list(world.get_actors().filter("vehicle.*"))

        # ── Sensor update ──────────────────────────────────────────────────────
        emergency_inputs = ([emergency_vehicle]
                            if emergency_vehicle is not None and emergency_vehicle.is_alive
                            else [])
        sensor_result = ground_sensor.update(all_actors, emergency_inputs)
        raw_sensor_counts = dict(sensor_result['arm_counts'])
        control_conf = 0.95

        wait_trackers[0].update(all_actors, center, tick_count)
        gt_count, avg_speed = compute_gt(center)
        _wt_stats = wait_trackers[0].get_stats()

        # Pressure input: use ONLY stopped-vehicle queues from the wait tracker.
        # The wait tracker counts vehicles with speed < 0.5 m/s per arm — this
        # correctly reflects actual queued demand.  Geometry (any vehicle in ROI)
        # and raw ground-sensor counts both over-count moving vehicles that are
        # just passing through, producing false pressure on empty arms.
        for arm in ('N', 'S', 'E', 'W'):
            latest_sensor_counts[arm] = _wt_stats['arm_queues'].get(arm, 0)
        control_count = sum(latest_sensor_counts.values())

        # Per-arm queues and waits fed to pressure controller.
        # Normalize waits relative to the current green arm so that saturated
        # intersections (where all arms have high absolute wait) don't deadlock
        # by symmetry — only the excess wait over half the serving arm's wait counts.
        arm_queues = dict(latest_sensor_counts)
        raw_waits  = dict(_wt_stats['arm_avg_waits'])
        current_wait = raw_waits.get(pressure_ctrl.current_arm, 0.0)
        arm_avg_waits = {
            arm: max(0.0, w - current_wait * 0.5)
            for arm, w in raw_waits.items()
        }

        # ── Vehicle replenishment ──────────────────────────────────────────────
        if tick_count % 200 == 0:
            nearby = sum(1 for v in all_actors
                         if v.is_alive
                         and v.get_location().distance(center) < REPLENISH_RADIUS_M
                         and not is_emergency_vehicle(v))
            if nearby < REPLENISH_TARGET:
                need    = REPLENISH_TARGET - nearby
                all_sps = world.get_map().get_spawn_points()
                random.shuffle(all_sps)
                added   = 0
                for sp in all_sps:
                    if added >= need: break
                    if 30 < sp.location.distance(center) < REPLENISH_RADIUS_M:
                        bp = random.choice(car_bps)
                        nv = world.try_spawn_actor(bp, sp)
                        if nv:
                            nv.set_autopilot(True)
                            vehicles.append(nv)
                            added += 1

        # ── Simulated sensor degradation (FALLBACK demo) ───────────────────────
        if system_mode == 'DQN':
            _sim_lc_timer += 1
        if not _sim_lc_active and _sim_lc_timer >= SIM_LC_INTERVAL and system_mode == 'DQN':
            _sim_lc_active = True; _sim_lc_dur_ctr = 0; _sim_lc_timer = 0
            print(f"[DEMO] tick {tick_count}: Simulating sensor degradation "
                  f"({SIM_LC_DURATION} ticks).")
        if _sim_lc_active:
            _sim_lc_dur_ctr += 1
            if _sim_lc_dur_ctr >= SIM_LC_DURATION:
                _sim_lc_active = False; _sim_lc_dur_ctr = 0
        effective_control_conf = 0.10 if _sim_lc_active else control_conf

        # Update fallback controller (emergency handled separately).
        # Pass gt_count (all vehicles in ROI) not control_count (stopped only);
        # an empty queue at startup does not mean the sensor is blind.
        _emg_flag     = 1 if emergency_handler.is_active() else 0
        fallback_mode = fallback_ctrl.update(effective_control_conf, gt_count, _emg_flag)

        # ── MAIN CONTROL LOOP (EVERY SECOND) ──────────────────────────────────
        # The user requested specific logic to run "every second".
        # 20 ticks @ 0.05s = 1 second.
        if tick_count % 20 == 0:
            # 1. Vehicles are driving around (handled by CARLA tick)

            # 2. Count stopped vehicles and track wait time
            wait_trackers[0].update(all_actors, center, tick_count)
            _wt_stats = wait_trackers[0].get_stats()
            
            # 3. Pick a score for each road
            arm_queues = dict(_wt_stats['arm_queues'])
            arm_avg_waits = dict(_wt_stats['arm_avg_waits'])
            pressures = pressure_ctrl.compute_pressures(arm_queues, arm_avg_waits)

            # ── Special situations: Ambulance ─────────────────────────────────
            _was_active = emergency_handler.is_active()
            _known_emg = [emergency_vehicle] if (emergency_vehicle and emergency_vehicle.is_alive) else []
            em_state = emergency_handler.update_state_machine(all_actors, known=_known_emg)
            
            if emergency_handler.is_active():
                system_mode = em_state
                emergency_handler.apply_emergency_control(signal_manager)
            
            # ── Special situations: Sensor Blind (Fallback) ───────────────────
            elif effective_control_conf < 0.35:
                system_mode = 'FALLBACK'
                # Rotate N -> S -> E -> W every 30 seconds (600 ticks)
                fallback_arm_ticks += 20  # we are in the 1s block
                if fallback_arm_ticks >= 600:
                    cur_idx = FIXED_ARM_ORDER.index(pressure_ctrl.current_arm) if pressure_ctrl.current_arm in FIXED_ARM_ORDER else 0
                    pending_arm = FIXED_ARM_ORDER[(cur_idx + 1) % 4]
                    yellow_arm = pressure_ctrl.current_arm
                    yellow_active = True
                    yellow_counter = 0
                    fallback_arm_ticks = 0
                    signal_manager.set_arm_yellow(yellow_arm)
                else:
                    signal_manager.set_arm_green(pressure_ctrl.current_arm)

            # ── Normal Pressure-based control ─────────────────────────────────
            else:
                system_mode = 'PRESSURE'
                if not yellow_active:
                    # 4. Switch decision
                    do_switch, best_arm = pressure_ctrl.should_switch(pressures)
                    
                    # 5. If yes — yellow for 3s, then green
                    if do_switch:
                        pending_arm = best_arm
                        yellow_arm = pressure_ctrl.current_arm
                        yellow_active = True
                        yellow_counter = 0
                        signal_manager.set_arm_yellow(yellow_arm)
                        print(f"[SWITCH] {yellow_arm} -> {best_arm} (Score: {pressures[yellow_arm]:.1f} -> {pressures[best_arm]:.1f})")
                    # 6. If no — stay green
                    else:
                        signal_manager.set_arm_green(pressure_ctrl.current_arm)

        # 7. The green light is pushed to CARLA every single tick
        if yellow_active:
            yellow_counter += 1
            signal_manager.set_arm_yellow(yellow_arm)
            if yellow_counter >= YELLOW_TICKS:
                yellow_active = False
                pressure_ctrl.commit_switch(pending_arm)
                signal_manager.set_arm_green(pending_arm)
        elif not emergency_handler.is_active():
            signal_manager.set_arm_green(pressure_ctrl.current_arm)
        else:
            # Emergency handler handles its own signal pushing in apply_emergency_control
            emergency_handler.apply_emergency_control(signal_manager)

        # ── Dashboard logic ──────────────────────────────────────────────────
        if yellow_active:
            display_phase = 1
        elif emergency_handler.is_active():
            if emergency_handler.state == EmergencyState.PRE_CLEAR:
                display_phase = 2
            elif emergency_handler.in_yellow:
                display_phase = 1
            else:
                display_phase = 0
        else:
            display_phase = 0

        # ── Screen update ─────────────────────────────────────────────────────
        if tick_count % YOLO_INTERVAL == 0:
            yolo_conf = effective_control_conf

            # 8. Screen updates - dashboard shows pressures, wait time, green arm
            ov = draw_overview_annotated(
                display_phase, system_mode, gt_count,
                pressure_ctrl.current_arm, pressures,
                emg_vehicle=emergency_vehicle,
            )
            if ov is not None:
                overview_frame = ov

            s = wait_trackers[0].get_stats()
            wait_history[0].append(s['avg_waiting_time'])
            if len(wait_history[0]) > HIST_LEN:
                wait_history[0].pop(0)

            dash = draw_dashboard(
                display_phase, gt_count, s, system_mode,
                wait_history, tick_count, pressure_ctrl.current_arm, pressures,
                yolo_conf=yolo_conf,
                emg_count=emergency_event_count,
                fb_count=fallback_event_count
            )
            if overview_frame is not None:
                combined = np.hstack((overview_frame, dash))
                cv2.imshow("CARLA Traffic Intelligence Demo", combined)

        # ── Input handling ────────────────────────────────────────────────────
        _key = cv2.waitKey(1) & 0xFF
        if _key == ord('q'):
            break
        elif _key == ord('e'):
            if emergency_vehicle is None or not emergency_vehicle.is_alive:
                emergency_vehicle = spawn_emergency()
                if emergency_vehicle:
                    print(f"[MANUAL] tick {tick_count}: Emergency vehicle triggered.")

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    stop_siren()
    for cam in cameras:
        try:
            cam.stop(); cam.destroy()
        except RuntimeError:
            pass
    for v in vehicles:
        _safe_destroy_actor(v)
    _safe_destroy_actor(emergency_vehicle)
    settings.synchronous_mode = False
    try:
        world.apply_settings(settings)
    except RuntimeError as e:
        print(f"[WARN] Could not apply async settings on shutdown: {e}")
    print("Demo ended.")
