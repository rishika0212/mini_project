"""
demo.py — Live visual demo of trained DQN agent

Two cameras with distinct roles:
  Camera 0  (Road View)    — positioned ~50 m back on the approach road, looking
                             toward the intersection.  YOLO detects all vehicles
                             and flags an emergency if one enters within
                             ROAD_DETECTION_DIST metres.  Triggering turns that
                             road's signal GREEN immediately.

  Camera 1  (Overview)     — high overhead, almost straight-down bird's-eye view
                             of the full intersection.  Shows every approaching
                             road, live signal-light dots, and an emergency
                             bounding-box if one is present.

Display layout:  [ Road Cam ] [ Intersection Overview ] [ Dashboard ]

Usage:
    python demo.py
"""

import carla
import random
import math
import numpy as np
import cv2
from ultralytics import YOLO
import threading
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

from dqn_agent    import DQNAgent, build_state
from waiting_time import IntersectionWaitingTimeTracker
from fallback     import FallbackController, ControlMode, get_yolo_confidence
from ground_sensors import IntersectionGroundSensors

# ── Connect & load map ────────────────────────────────────────────────────
# Town03 has straight 4-way intersections and a clean road grid — ideal for
# the surveillance demo.  Town10 has curved/complex roads that make the
# approach camera look like a side-street view.
TARGET_MAP = 'Town03'

import time

print("Connecting to CARLA on localhost:2000 ...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    # Test the connection before doing anything else
    client.get_server_version()
except Exception as e:
    print(f"\n[ERROR] Cannot reach CARLA server: {e}")
    print("  Make sure CARLA is fully loaded before running this script.")
    raise SystemExit(1)

print("Connected. Checking map...")
try:
    current_map = client.get_world().get_map().name
except Exception as e:
    print(f"\n[ERROR] Could not get world/map: {e}")
    raise SystemExit(1)

if TARGET_MAP not in current_map:
    print(f"WARNING: Currently on '{current_map}', not '{TARGET_MAP}'.")
    print(f"  Recommended: launch CARLA with Town03 pre-loaded:")
    print(f"    CarlaUE4.exe /Game/Carla/Maps/Town03")
    print(f"  Attempting map switch (may crash on some systems due to shader recompilation)...")
    try:
        world = client.load_world(TARGET_MAP)
        time.sleep(15)   # UE4 needs time to reload
    except Exception as e:
        print(f"\n[ERROR] Map switch failed: {e}")
        print(f"  Falling back to current map: {current_map}")
        # Re-create the client — the failed load_world() call leaves the
        # existing client in a broken/timed-out state so any subsequent RPC
        # (including get_world()) would also time out.
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(30.0)
            world = client.get_world()
            print(f"  Continuing on {world.get_map().name} — intersection detection will still work.")
        except Exception as e2:
            print(f"\n[ERROR] Could not reconnect after failed map switch: {e2}")
            raise SystemExit(1)
else:
    world = client.get_world()
    print(f"Already on {TARGET_MAP}.")

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode   = False
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

print("Connected to CARLA.")

# ── YOLOv8 ─────────────────────────────────────────────────────────────────
model              = YOLO('yolov8s.pt')  # Small model (better accuracy than nano)
VEHICLE_CLASSES    = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
PEDESTRIAN_CLASSES = {0: 'person', 1: 'bicycle'}   # detected separately, not counted as vehicles
EMERGENCY_KW       = ['ambulance', 'firetruck', 'police']

# ── DQN agent (single intersection) ───────────────────────────────────────
NUM_INT = 1
agents  = []
for i in range(NUM_INT):
    agent         = DQNAgent()
    agent.epsilon = 0.0
    agent.load(f"data/dqn_weights_int{i+1}.json")
    agent.epsilon = 0.0
    agents.append(agent)

print("DQN agent loaded (greedy mode).")

# ── Intersection ──────────────────────────────────────────────────────────
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

traffic_lights  = world.get_actors().filter("traffic.traffic_light")
groups          = group_lights(traffic_lights, threshold=45)

# Prefer groups with 4 lights (true 4-way junction) for a clean "+" view.
# Fall back to 3-light groups if none found.
quad_groups  = [g for g in groups if len(g) >= 4]
valid_groups = quad_groups if quad_groups else [g for g in groups if len(g) >= 3]

# Among candidates, pick the one closest to Town03's main grid centre
# (roughly world origin) — that junction has perfectly straight arms.
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

center = intersection_centers[0]          # single intersection centre

# ROAD_DETECTION_DIST: fixed range on the approach road that triggers the
# emergency-override when an emergency vehicle enters it.
ROI_RADIUS          = 60    # metres — used for queue/wait-time tracking
ROAD_DETECTION_DIST = 75    # keep trigger inside CAM 2 footprint for visible demos

wait_trackers = [IntersectionWaitingTimeTracker(i + 1, ROI_RADIUS, intersection_centers[i])
                 for i in range(NUM_INT)]

# Ground-sensor backend for control decisions (realistic per-arm inputs).
arm_directions = {'N': 0, 'S': 180, 'E': 90, 'W': 270}
ground_sensor = IntersectionGroundSensors(
    intersection_id=1,
    intersection_center=center,
    arm_directions=arm_directions,
)

# ── Fallback controller (Scenario 4: YOLO failure → FIXED_TIME) ──────────
# Always pass emergency_flag=0 so it only handles DQN ↔ FIXED_TIME switching;
# emergency is handled separately by the demo state machine.
fallback_ctrl        = FallbackController(intersection_id=1)
yolo_conf            = 0.0   # visualization confidence (from YOLO camera)
control_conf         = 0.95  # control confidence (from ground sensors)
control_count        = 0     # control count (from ground sensors)
fallback_event_count = 0
emergency_event_count = 0

# Periodic simulated low-confidence window so Scenario 4 plays out in demo
# without requiring actual YOLO failure.
SIM_LC_INTERVAL = 3600   # ~180 s between fallback demo events (3 minutes)
SIM_LC_DURATION = 160    # ~8 s of forced low conf (> LOW_CONF_WINDOW=5 ticks)
_sim_lc_timer   = 0
_sim_lc_active  = False
_sim_lc_dur_ctr = 0

# ── Vehicles ───────────────────────────────────────────────────────────────
blueprints      = world.get_blueprint_library().filter("vehicle.*")
EMERGENCY_BP_KW = ['ambulance', 'firetruck', 'police']
car_bps         = [bp for bp in blueprints
                   if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                   and not any(kw in bp.id.lower() for kw in EMERGENCY_BP_KW)]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

# Demo density knobs (higher values make DQN behavior more visible on screen).
GLOBAL_SPAWN_LIMIT   = 240
NEARBY_SPAWN_LIMIT   = 140
REPLENISH_TARGET     = 90
REPLENISH_RADIUS_M   = 180

# Global TM safety margins — set BEFORE spawning so every vehicle inherits them.
traffic_manager.set_global_distance_to_leading_vehicle(3.0)   # 3 m gap
traffic_manager.global_percentage_speed_difference(10)         # 10 % below limit

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
for _ in range(20): world.tick()

# ── Cameras ────────────────────────────────────────────────────────────────
#
#  ROAD_CAM  (index 0)  — CAM 1: Perception / Detection
#    Placed 50 m back along the ACTUAL approach lane (via waypoint.previous).
#    Faces oncoming traffic (yaw = road direction + 180°).
#    Height 7 m, pitch -15°.  Gives the "CCTV watching traffic come in" view.
#    Used for: YOLO detection, emergency detection.
#
#  OVERVIEW_CAM  (index 1)  — CAM 2: System Response / Intersection View
#    Directly above the intersection centre at 20 m, pitch -80°.
#    Yaw aligned to the approach road so the 4-way cross is axis-aligned.
#    Used for: live flow, signal-light dots, emergency box.
#
ROAD_CAM_IDX     = 0
OVERVIEW_CAM_IDX = 1

road_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
road_cam_bp.set_attribute('image_size_x', '640')
road_cam_bp.set_attribute('image_size_y', '640')
road_cam_bp.set_attribute('fov', '75')
road_cam_bp.set_attribute('sensor_tick', '0.05')

overview_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
overview_cam_bp.set_attribute('image_size_x', '640')
overview_cam_bp.set_attribute('image_size_y', '640')
overview_cam_bp.set_attribute('fov', '110')   # wide — captures all four approach roads
overview_cam_bp.set_attribute('sensor_tick', '0.05')

# ── Waypoint chain: shared reference for CAM 1 and emergency spawner ─────
#
# approach_wp   — 60 m back from the intersection.  Used by spawn_emergency
#                 as the reference point to walk further back from.
# cam1_wp       — 20 m back from the intersection.  CAM 1 sits here so it
#                 shows a close-up of vehicles queuing / passing through
#                 the intersection arm, not a distant street view.
#
carla_map   = world.get_map()
approach_wp = carla_map.get_waypoint(center, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
for _ in range(6):             # walk back 6×10 m = 60 m (for emergency spawner)
    prev = approach_wp.previous(10)
    if prev:
        approach_wp = prev[0]

# Derive cam1_wp 40 m back — far enough that a vehicle spawned at 30–65 m
# is in front of the camera, not behind it.
cam1_wp = carla_map.get_waypoint(center, project_to_road=True,
                                 lane_type=carla.LaneType.Driving)
for _ in range(4):             # 4×10 m = 40 m back (was 2×10 = 20 m)
    prev = cam1_wp.previous(10)
    if prev:
        cam1_wp = prev[0]

road_loc  = cam1_wp.transform.location
road_yaw  = cam1_wp.transform.rotation.yaw   # direction traffic flows → intersection
cam1_yaw  = road_yaw + 180.0                 # flip: face oncoming vehicles

# height 4.5 m, pitch -15° — steeper angle to better frame vehicles in approach zone
road_cam_transform = carla.Transform(
    carla.Location(x=road_loc.x, y=road_loc.y, z=road_loc.z + 4.5),
    carla.Rotation(pitch=-15, yaw=cam1_yaw)
)
print(f"Road cam (intersection arm): ({road_loc.x:.1f}, {road_loc.y:.1f})  "
      f"road_yaw={road_yaw:.1f}°  cam_yaw={cam1_yaw:.1f}°  "
      f"dist_to_center={road_loc.distance(center):.1f} m")

# ── Pre-cache spawn points visible in both cameras (used by spawn_emergency)
# Spawn range 30–65 m: cam2 covers ~78 m radius so this always lands inside it.
# Candidates sorted farthest-first so the vehicle spawns at ~60–65 m, ensuring
# it starts well in front of cam1 (which is at 40 m) and stays visible longest.
_approach_spawn_candidates = []
for _sp in world.get_map().get_spawn_points():
    _d = _sp.location.distance(center)
    if _d < 30 or _d > 65:
        continue
    _dx = _sp.location.x - center.x
    _dy = _sp.location.y - center.y
    _sp_angle     = math.degrees(math.atan2(_dy, _dx))
    _approach_dir = (road_yaw + 180) % 360   # direction AWAY from center along approach arm
    _diff = abs((_sp_angle - _approach_dir + 180) % 360 - 180)
    if _diff < 60:
        _approach_spawn_candidates.append(_sp)
# Farthest from intersection first — vehicle spawns at 60-65 m, well in front
# of cam1 (40 m), giving maximum dwell time in both camera views.
_approach_spawn_candidates.sort(key=lambda sp: sp.location.distance(center),
                                reverse=True)
print(f"Approach arm spawn candidates (30-65 m, farthest first): {len(_approach_spawn_candidates)}")

# ── Split selected intersection lights into two movement groups ────────────
# Only use the chosen intersection actor group to avoid mixing nearby junctions.
def _angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)

def split_intersection_arms(group):
    if not group:
        return [], []
    ref_yaw = group[0].get_transform().rotation.yaw
    arm_a, arm_b = [], []
    for light in group:
        lyaw = light.get_transform().rotation.yaw
        diff = _angle_diff(lyaw, ref_yaw)
        if diff < 45 or diff > 135:
            arm_a.append(light)
        else:
            arm_b.append(light)
    if not arm_a or not arm_b:
        mid = max(1, len(group) // 2)
        arm_a = list(group[:mid])
        arm_b = list(group[mid:])
    return arm_a, arm_b

main_intersection_lights = list(intersections[0])
group_A, group_B = split_intersection_arms(main_intersection_lights)

def _arm_from_location_for_labels(loc):
    dx = loc.x - center.x
    dy = loc.y - center.y
    if abs(dx) >= abs(dy):
        return 'E' if dx >= 0 else 'W'
    return 'S' if dy >= 0 else 'N'

group_A_arms = sorted({_arm_from_location_for_labels(l.get_location()) for l in group_A})
group_B_arms = sorted({_arm_from_location_for_labels(l.get_location()) for l in group_B})

print(f"Selected intersection lights: {len(main_intersection_lights)}")
print(f"Signal containers — A: {len(group_A)} lights (arms={group_A_arms}), "
      f"B: {len(group_B)} lights (arms={group_B_arms})")
print("[INFO] A/B are CARLA light-head containers only; control pressure uses N/S/E/W separately.")

# ── CAM 2: True top-down intersection view ────────────────────────────────
#
# pitch = -90  →  dead straight down, zero perspective lean
# yaw   =   0  →  world X/Y axis aligned → 4-way arms form a clean "+"
# roll  =   0  →  no image-plane rotation
# z     =  30  →  30 m gives enough ground footprint at 110° FOV to show
#                 all four arms plus stopped vehicles on each approach
#
# NEVER set yaw = road_yaw here — that rotates the "+" diagonally.
#
overview_cam_transform = carla.Transform(
    carla.Location(x=center.x, y=center.y, z=55),   # raised: 55 m → ~78 m ground radius at 110° FOV
    carla.Rotation(pitch=-90, yaw=0, roll=0)
)

camera_transforms = [road_cam_transform, overview_cam_transform]

latest_frames = {0: None, 1: None}
frame_locks   = {0: threading.Lock(), 1: threading.Lock()}
frame_counts  = {0: 0, 1: 0}

road_cam     = world.spawn_actor(road_cam_bp,     road_cam_transform)
overview_cam = world.spawn_actor(overview_cam_bp, overview_cam_transform)
cameras      = [road_cam, overview_cam]

def on_image(image, idx):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb   = array[:, :, :3][:, :, ::-1].copy()
    with frame_locks[idx]:
        latest_frames[idx] = rgb
        frame_counts[idx] += 1

road_cam.listen(lambda img: on_image(img, ROAD_CAM_IDX))
overview_cam.listen(lambda img: on_image(img, OVERVIEW_CAM_IDX))

# Freeze BEFORE warm-up so CARLA's internal timer cannot put lights
# into an unknown state during the 30-tick warm-up period.
for _light in intersections[0]:
    _light.freeze(True)
print("Traffic lights frozen — manual control active.")

print("Cameras ready. Warming up...")
for _ in range(30): world.tick()

# Initialize to a demand-aligned green state to avoid long startup all-red.
_startup_actors = list(world.get_actors().filter("vehicle.*"))
wait_trackers[0].update(_startup_actors, center, 0)
_startup_waits = wait_trackers[0].get_stats()
_startup_sensor = ground_sensor.update(_startup_actors, [])
_a_count = _startup_sensor['arm_counts']['N'] + _startup_sensor['arm_counts']['S']
_b_count = _startup_sensor['arm_counts']['E'] + _startup_sensor['arm_counts']['W']
initial_green_phase = 0 if _a_count >= _b_count else 2

# ── Camera projection helpers ──────────────────────────────────────────────
_CAM_W, _CAM_H = 640, 640

# Road camera intrinsics  (FOV 75°)
_ROAD_FOV   = 75
_road_focal = _CAM_W / (2.0 * math.tan(math.radians(_ROAD_FOV / 2.0)))
ROAD_K = np.array([[_road_focal, 0,           _CAM_W / 2.0],
                   [0,           _road_focal,  _CAM_H / 2.0],
                   [0,           0,            1.0          ]], dtype=np.float64)

# Overview camera intrinsics  (FOV 110°)
_OV_FOV   = 110
_ov_focal = _CAM_W / (2.0 * math.tan(math.radians(_OV_FOV / 2.0)))
OV_K = np.array([[_ov_focal, 0,          _CAM_W / 2.0],
                 [0,          _ov_focal,  _CAM_H / 2.0],
                 [0,          0,          1.0          ]], dtype=np.float64)

def _world_to_cam(transform):
    """
    Return 4×4 world-to-camera matrix for a carla.Transform.

    Builds the camera-to-world rotation matrix from CARLA's yaw/pitch/roll
    convention, then inverts it.  For the top-down overview camera (pitch=-90)
    this correctly places p[0] = tz - z_actor (positive depth) so traffic-light
    dots and emergency-vehicle boxes are projected into the frame.
    """
    cy = math.cos(math.radians(transform.rotation.yaw))
    sy = math.sin(math.radians(transform.rotation.yaw))
    cr = math.cos(math.radians(transform.rotation.roll))
    sr = math.sin(math.radians(transform.rotation.roll))
    cp = math.cos(math.radians(transform.rotation.pitch))
    sp = math.sin(math.radians(transform.rotation.pitch))
    tx = transform.location.x
    ty = transform.location.y
    tz = transform.location.z
    c2w = np.array([
        [cp*cy,  cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr, tx],
        [cp*sy,  sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr, ty],
        [sp,    -cp*sr,             cp*cr,             tz],
        [0,      0,                 0,                 1 ]
    ], dtype=np.float64)
    return np.linalg.inv(c2w)

def project_actor(actor, cam_idx):
    """
    Project actor centre to pixel (u, v, depth) for the given camera.
    Returns None if behind the camera or outside the frame.
    """
    K   = OV_K if cam_idx == OVERVIEW_CAM_IDX else ROAD_K
    loc = actor.get_location()
    w2c = _world_to_cam(camera_transforms[cam_idx])
    p   = w2c @ np.array([loc.x, loc.y, loc.z, 1.0])
    if p[0] <= 0:
        return None
    u = int(K[0, 0] * p[1] / p[0] + K[0, 2])
    v = int(K[1, 1] * (-p[2]) / p[0] + K[1, 2])
    if not (0 <= u < _CAM_W and 0 <= v < _CAM_H):
        return None
    return u, v, p[0]

# ── Siren ──────────────────────────────────────────────────────────────────
_siren_on = False

def _siren_loop():
    while _siren_on:
        if _HAS_WINSOUND:
            winsound.Beep(1400, 180)
            winsound.Beep(900,  180)
        else:
            time.sleep(0.36)   # keep thread alive on non-Windows

def start_siren():
    global _siren_on
    if not _siren_on:
        _siren_on = True
        threading.Thread(target=_siren_loop, daemon=True).start()

def stop_siren():
    global _siren_on
    _siren_on = False

# ── Shared phase colours / names ───────────────────────────────────────────
PHASE_COLORS = {0: (0, 255, 0), 1: (0, 215, 255), 2: (0, 0, 255)}
PHASE_NAMES  = {0: 'GREEN',     1: 'YELLOW',       2: 'RED'}

# ── Road camera — YOLO annotated frame ────────────────────────────────────
def run_yolo_annotated(phase, yolo_count, is_emergency=False, emg_vehicle=None,
                       sim_low_conf=False):
    """
    Grab the road-camera frame, run YOLO, draw bounding boxes and HUD.
    Returns (annotated_bgr_760x760, vehicle_count, avg_confidence).
    sim_low_conf: when True, override displayed confidence to 0.10 to show
                  the FALLBACK scenario visually.
    """
    with frame_locks[ROAD_CAM_IDX]:
        frame = latest_frames[ROAD_CAM_IDX]
    if frame is None:
        return None, 0, 0.0

    results   = model(frame, verbose=False, conf=0.25)
    avg_conf, _ = get_yolo_confidence(results)
    if sim_low_conf:
        avg_conf = 0.10   # simulate degraded sensor for Scenario 4 display
    annotated = frame[:, :, ::-1].copy()   # RGB → BGR
    count     = 0
    ped_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            elif cls_id in PEDESTRIAN_CLASSES:
                ped_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{PEDESTRIAN_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 165, 0), 1)

    # ── HUD bar ────────────────────────────────────────────────────────────
    ph_color = PHASE_COLORS[phase]
    ph_name  = PHASE_NAMES[phase]

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (640, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    cv2.putText(annotated, "CAM 1  —  APPROACH / DETECTION  (PERCEPTION)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    # Confidence bar and vehicle count
    conf_color = (0, 200, 80) if avg_conf > 0.5 else (0, 180, 220) if avg_conf > 0.35 else (0, 60, 220)
    cv2.putText(annotated,
                f"YOLO: {count} veh  {ped_count} ped  |  Conf: {avg_conf:.2f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, conf_color, 1)
    # Small inline confidence bar
    bar_w = int(180 * max(0.0, min(1.0, avg_conf)))
    cv2.rectangle(annotated, (310, 38), (490, 46), (50, 50, 50), -1)
    cv2.rectangle(annotated, (310, 38), (310 + bar_w, 46), conf_color, -1)

    # Signal indicator (top-right circle)
    cv2.circle(annotated, (580, 28), 17, ph_color, -1)
    cv2.circle(annotated, (580, 28), 17, (255, 255, 255), 1)
    cv2.putText(annotated, ph_name,
                (492, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ph_color, 2)

    # ── Project emergency vehicle as red box (uses tracked object directly) ──
    if emg_vehicle and emg_vehicle.is_alive:
        pt = project_actor(emg_vehicle, ROAD_CAM_IDX)
        if pt is not None:
            u, v_px, depth = pt
            ext = emg_vehicle.bounding_box.extent
            hw  = max(22, int(ROAD_K[0, 0] * ext.y / depth))
            hh  = max(22, int(ROAD_K[1, 1] * ext.z / depth))
            x1 = max(0,         u - hw)
            y1 = max(0,         v_px - hh)
            x2 = min(_CAM_W-1,  u + hw)
            y2 = min(_CAM_H-1,  v_px + hh)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
            name = emg_vehicle.type_id.split('.')[-1].upper()
            cv2.putText(annotated, f"EMERGENCY: {name}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # ── Mode banner below HUD ─────────────────────────────────────────────
    if is_emergency:
        cv2.rectangle(annotated, (0, 64), (640, 104), (0, 0, 160), -1)
        cv2.putText(annotated, "  EMERGENCY OVERRIDE ACTIVE",
                    (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 255), 2)
        cv2.putText(annotated, "  Approach signal: GREEN  |  All others: RED",
                    (8, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)
    elif sim_low_conf:
        cv2.rectangle(annotated, (0, 64), (640, 104), (0, 40, 80), -1)
        cv2.putText(annotated, "  SENSOR DEGRADED — YOLO low confidence",
                    (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
        cv2.putText(annotated, "  Switching to FIXED-TIME fallback ...",
                    (8, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
    else:
        cv2.rectangle(annotated, (0, 64), (640, 86), (0, 50, 0), -1)
        cv2.putText(annotated, "  RL MODE  —  DQN agent controlling signals",
                    (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 220, 70), 1)

    annotated = cv2.resize(annotated, (760, 760), interpolation=cv2.INTER_LINEAR)
    return annotated, count, avg_conf


# ── Overview camera — intersection flow annotated frame ───────────────────
def draw_overview_annotated(phase, is_emergency, vehicles_in_roi,
                            emg_vehicle=None, system_mode='DQN'):
    """
    Grab the overhead intersection frame and annotate with:
      • Signal-light dots projected from CARLA world positions
      • High-visibility emergency vehicle box (passed directly)
      • HUD bar showing live signal state and vehicle count
      • Mode banner at bottom (Emergency / Fallback / RL)
    Returns annotated_bgr_760x760 or None.
    """
    with frame_locks[OVERVIEW_CAM_IDX]:
        frame = latest_frames[OVERVIEW_CAM_IDX]
    if frame is None:
        return None

    annotated = frame[:, :, ::-1].copy()   # RGB → BGR

    # ── HUD bar ────────────────────────────────────────────────────────────
    ph_color = PHASE_COLORS[phase]
    ph_name  = PHASE_NAMES[phase]

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (640, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    cv2.putText(annotated, "CAM 2  —  INTERSECTION OVERVIEW  (SYSTEM RESPONSE)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 100), 2)
    cv2.putText(annotated, f"Vehicles in zone: {vehicles_in_roi}   |   Signal: {ph_name}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    # Signal indicator (top-right)
    cv2.circle(annotated, (580, 28), 17, ph_color, -1)
    cv2.circle(annotated, (580, 28), 17, (255, 255, 255), 1)
    cv2.putText(annotated, ph_name,
                (492, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ph_color, 2)

    # ── Project each traffic light in the intersection as a coloured dot ───
    for light in intersections[0]:
        pt = project_actor(light, OVERVIEW_CAM_IDX)
        if pt is None:
            continue
        u, v_px, _ = pt
        state = light.get_state()
        if   state == carla.TrafficLightState.Green:  dot_col = (0, 230, 0)
        elif state == carla.TrafficLightState.Yellow: dot_col = (0, 230, 230)
        else:                                          dot_col = (0, 0, 230)
        cv2.circle(annotated, (u, v_px), 10, dot_col, -1)
        cv2.circle(annotated, (u, v_px), 10, (255, 255, 255), 2)

    # ── Emergency vehicle overlay ─────────────────────────────────────────
    # Uses the tracked vehicle object directly so fallback blueprints
    # (police cars, regular cars) are always shown regardless of type_id.
    if emg_vehicle and emg_vehicle.is_alive:
        name = emg_vehicle.type_id.split('.')[-1].upper()

        pt = project_actor(emg_vehicle, OVERVIEW_CAM_IDX)
        if pt is not None:
            u, v_px, _ = pt
            # Fixed 30×20 px — top-down at 55 m makes vehicles ~10 px wide
            # so we enforce a visible minimum size.
            hw, hh = 30, 20
            x1 = max(0,        u - hw)
            y1 = max(0,        v_px - hh)
            x2 = min(_CAM_W-1, u + hw)
            y2 = min(_CAM_H-1, v_px + hh)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(annotated, (x1, y1), (x2, y1 + 20), (0, 0, 200), -1)
            cv2.putText(annotated, f"EMERGENCY: {name}",
                        (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── Mode banner at bottom of overview ─────────────────────────────────
    if is_emergency:
        cv2.rectangle(annotated, (0, 596), (640, 640), (0, 0, 160), -1)
        cv2.putText(annotated, "  EMERGENCY OVERRIDE",
                    (8, 618), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 255), 2)
        cv2.putText(annotated, "  Approach road cleared — all other signals RED",
                    (8, 636), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)
    elif system_mode == 'RECOVERY':
        cv2.rectangle(annotated, (0, 596), (640, 640), (0, 45, 70), -1)
        cv2.putText(annotated, "  RECOVERY — fixed-time draining backlog",
                    (8, 618), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 230), 2)
        cv2.putText(annotated, "  RL resumes after backlog clears",
                    (8, 636), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 210, 240), 1)
    elif system_mode == 'FALLBACK':
        cv2.rectangle(annotated, (0, 596), (640, 640), (0, 40, 80), -1)
        cv2.putText(annotated, "  FALLBACK — FIXED-TIME (low YOLO confidence)",
                    (8, 618), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 180, 255), 2)
        cv2.putText(annotated, "  30s green → 30s red cycle — safe default",
                    (8, 636), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
    else:
        cv2.rectangle(annotated, (0, 618), (640, 640), (0, 40, 0), -1)
        cv2.putText(annotated, "  RL MODE  —  DQN controlling signal cycle",
                    (8, 634), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 200, 70), 1)

    annotated = cv2.resize(annotated, (760, 760), interpolation=cv2.INTER_LINEAR)
    return annotated


# ── Ground truth + arm counts + emergency ────────────────────────────────
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
    """Count vehicles per approach arm (N/S/E/W). +X=East, +Y=South in CARLA."""
    n = s = e = w = 0
    for v in actors:
        loc = v.get_location()
        if loc.distance(c) < ROI_RADIUS:
            dx = loc.x - c.x
            dy = loc.y - c.y
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

def get_emergency_vehicles(c):
    """Return list of all live emergency vehicles within ROAD_DETECTION_DIST."""
    result = []
    for v in world.get_actors().filter("vehicle.*"):
        if not v.is_alive: continue
        dist = v.get_location().distance(c)
        is_tracked = emergency_vehicle is not None and v.id == emergency_vehicle.id
        if dist < ROAD_DETECTION_DIST:
            # Only trigger emergency override when vehicle is visible in CAM 2.
            if (is_emergency_vehicle(v) or is_tracked) and _is_visible_in_overview(v):
                result.append(v)
    return result

# ── Signal & Emergency State ────────────────────────────────────────────────
phase_states       = [carla.TrafficLightState.Green,
                      carla.TrafficLightState.Yellow,
                      carla.TrafficLightState.Red]
phases          = [initial_green_phase] * NUM_INT
counters        = [0] * NUM_INT
phase_durations = [100] * NUM_INT
yolo_counts     = [0] * NUM_INT
pending_green_phase = [0] * NUM_INT
last_green_phase = [2] * NUM_INT
last_green_head  = [None] * NUM_INT

latest_arm_stats = {
    'arm_queues': dict(_startup_sensor['arm_counts']),
    'arm_avg_waits': dict(_startup_waits['arm_avg_waits']),
}

latest_sensor_counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}

DEMAND_EPS = 0.05

def _arm_from_location(loc):
    dx = loc.x - center.x
    dy = loc.y - center.y
    if abs(dx) >= abs(dy):
        return 'E' if dx >= 0 else 'W'
    return 'S' if dy >= 0 else 'N'

def _axis_demands(arm_stats):
    aq = arm_stats['arm_queues']
    aw = arm_stats['arm_avg_waits']

    # Queue-first demand: if no vehicles are present, ignore stale wait values.
    ns_q = aq['N'] + aq['S']
    ew_q = aq['E'] + aq['W']
    ns_w = (aw['N'] + aw['S']) if ns_q > 0 else 0.0
    ew_w = (aw['E'] + aw['W']) if ew_q > 0 else 0.0

    ns = ns_q + 0.12 * ns_w
    ew = ew_q + 0.12 * ew_w
    return ns, ew

def _phase_pressure(arm_stats, phase_id):
    # Phase 0 serves N/S demand, phase 2 serves E/W demand.
    ns, ew = _axis_demands(arm_stats)
    return ns if phase_id == 0 else ew

def _pick_best_green_phase(arm_stats, default_phase=0):
    a_pressure = _phase_pressure(arm_stats, 0)
    b_pressure = _phase_pressure(arm_stats, 2)
    if a_pressure <= DEMAND_EPS and b_pressure <= DEMAND_EPS:
        return default_phase
    return 0 if a_pressure >= b_pressure else 2

# ── State machine ──────────────────────────────────────────────────────────
# system_mode tracks which of the five control states we are in.
# Never add plain boolean flags here — all logic flows through system_mode.
system_mode        = 'DQN'   # 'DQN' | 'GRACE' | 'PRE_CLEAR' | 'EMERGENCY' | 'RECOVERY'
grace_counter      = 0        # counts down from GRACE_TICKS
preclear_counter   = 0        # counts down from PRECLEAR_TICKS
recovery_counter   = 0        # counts down from RECOVERY_TICKS
emergency_timeout  = 0        # extends GREEN time based on ambulance speed
emergency_crossed_center = False
ns_starve_ticks    = 0        # ticks N/S demand is waiting without service
ew_starve_ticks    = 0        # ticks E/W demand is waiting without service

GRACE_TICKS      = 40     # ~2 s   — detect emergency, prepare for clearance
PRECLEAR_TICKS   = 40     # ~2 s   — all RED to ensure intersection empty
RECOVERY_TICKS   = 200    # ~10 s  — rebalance backed-up traffic
MIN_EMERGENCY_GREEN = 200  # ~10 s minimum GREEN for ambulance (if speed high)

def set_phase(phase_idx):
    """
    Normal phase control — one direction green, the other red.
      Phase 0 → A = Green,  B = Red
      Phase 1 → A = Yellow, B = Red
      Phase 2 → A = Red,    B = Green
    Lights are already frozen; set_state() is permanent until next call.
    """
    def _light_pressure(light):
        arm = _arm_from_location(light.get_location())
        q = latest_arm_stats['arm_queues'].get(arm, 0)
        w = latest_arm_stats['arm_avg_waits'].get(arm, 0.0)
        return q + 0.2 * w

    def _set_group_state(group, state):
        for light in group:
            light.set_state(state)

    if phase_idx == 0:
        # Single active head in A group (highest-pressure arm), all others RED.
        best_idx = 0
        best_p = -1.0
        for i, l in enumerate(group_A):
            p = _light_pressure(l)
            if p > best_p:
                best_p = p
                best_idx = i
        if best_p <= 0.05:
            print("[SIGNAL] Phase 0: no demand in group_A, holding RED")
            _set_group_state(group_A, carla.TrafficLightState.Red)
            _set_group_state(group_B, carla.TrafficLightState.Red)
            last_green_phase[0] = 0
            last_green_head[0] = None
        else:
            print(f"[SIGNAL] Phase 0: Single GREEN in group_A[{best_idx}] (pressure={best_p:.2f}), others RED")
            for i, l in enumerate(group_A):
                st = carla.TrafficLightState.Green if i == best_idx else carla.TrafficLightState.Red
                l.set_state(st)
            _set_group_state(group_B, carla.TrafficLightState.Red)
            last_green_phase[0] = 0
            last_green_head[0] = ('A', best_idx)
    elif phase_idx == 1:
        # Yellow only on the previously active movement.
        active = last_green_head[0]
        if active is None:
            print("[SIGNAL] Phase 1: no active movement, holding RED")
            _set_group_state(group_A, carla.TrafficLightState.Red)
            _set_group_state(group_B, carla.TrafficLightState.Red)
        else:
            active_group, active_idx = active
            if active_group == 'A':
                print(f"[SIGNAL] Phase 1: group_A[{active_idx}] YELLOW transition, group_B RED")
                for i, l in enumerate(group_A):
                    l.set_state(carla.TrafficLightState.Yellow if i == active_idx else carla.TrafficLightState.Red)
                _set_group_state(group_B, carla.TrafficLightState.Red)
            else:
                print(f"[SIGNAL] Phase 1: group_B[{active_idx}] YELLOW transition, group_A RED")
                _set_group_state(group_A, carla.TrafficLightState.Red)
                for i, l in enumerate(group_B):
                    l.set_state(carla.TrafficLightState.Yellow if i == active_idx else carla.TrafficLightState.Red)
    else:  # phase_idx == 2 or other
        # Single active head in B group (highest-pressure arm), all others RED.
        best_idx = 0
        best_p = -1.0
        for i, l in enumerate(group_B):
            p = _light_pressure(l)
            if p > best_p:
                best_p = p
                best_idx = i
        if best_p <= 0.05:
            print("[SIGNAL] Phase 2: no demand in group_B, holding RED")
            _set_group_state(group_A, carla.TrafficLightState.Red)
            _set_group_state(group_B, carla.TrafficLightState.Red)
            last_green_phase[0] = 2
            last_green_head[0] = None
        else:
            print(f"[SIGNAL] Phase 2: Single GREEN in group_B[{best_idx}] (pressure={best_p:.2f}), others RED")
            _set_group_state(group_A, carla.TrafficLightState.Red)
            for i, l in enumerate(group_B):
                st = carla.TrafficLightState.Green if i == best_idx else carla.TrafficLightState.Red
                l.set_state(st)
            last_green_phase[0] = 2
            last_green_head[0] = ('B', best_idx)

def set_all_red():
    """Set all signals to RED — used during pre-clearance phase."""
    for l in group_A:
        l.set_state(carla.TrafficLightState.Red)
    for l in group_B:
        l.set_state(carla.TrafficLightState.Red)

def verify_light_states():
    """Debug function: print current state of ALL lights to verify control."""
    print("[DEBUG] Current light states:")
    green_count = 0
    yellow_count = 0
    red_count = 0
    green_a = 0
    green_b = 0
    for i, l in enumerate(group_A):
        state = l.get_state()
        state_name = "GREEN" if state == carla.TrafficLightState.Green else \
                     "YELLOW" if state == carla.TrafficLightState.Yellow else "RED"
        print(f"  group_A[{i}] @ ({l.get_location().x:.1f}, {l.get_location().y:.1f}): {state_name}")
        if state == carla.TrafficLightState.Green:
            green_count += 1
            green_a += 1
        elif state == carla.TrafficLightState.Yellow: yellow_count += 1
        else: red_count += 1
    for i, l in enumerate(group_B):
        state = l.get_state()
        state_name = "GREEN" if state == carla.TrafficLightState.Green else \
                     "YELLOW" if state == carla.TrafficLightState.Yellow else "RED"
        print(f"  group_B[{i}] @ ({l.get_location().x:.1f}, {l.get_location().y:.1f}): {state_name}")
        if state == carla.TrafficLightState.Green:
            green_count += 1
            green_b += 1
        elif state == carla.TrafficLightState.Yellow: yellow_count += 1
        else: red_count += 1
    print(f"[DEBUG] Total: {green_count} GREEN, {yellow_count} YELLOW, {red_count} RED")
    if green_a > 0 and green_b > 0:
        print(f"[ALERT] WARNING! Both groups have GREEN (A={green_a}, B={green_b})")
    if green_count > 1:
        print(f"[ALERT] WARNING! More than one GREEN head active ({green_count})")
    return green_count, yellow_count, red_count

# Apply startup phase using single-head selection logic to avoid 2-green startup.
set_phase(initial_green_phase)

def set_signals_for_emergency(emg_vehicles):
    """
    Priority: sort vehicles by distance to intersection — closest gets green.
    Compute the angular difference between the vehicle's bearing and road_yaw,
    then fold it so both forward AND backward along the same arm map to diff<45.
    Green that arm's group, red the other.
    Returns 'A' or 'B' — the arm that was given green — so the caller can
    set phases[0] to match (0=A green, 2=B green).
    """
    if not emg_vehicles:
        return None

    emg_vehicles = sorted(emg_vehicles,
                          key=lambda v: v.get_location().distance(center))
    priority_vehicle = emg_vehicles[0]

    loc = priority_vehicle.get_location()
    dx  = loc.x - center.x
    dy  = loc.y - center.y
    vehicle_angle = math.degrees(math.atan2(dy, dx))
    diff = abs((vehicle_angle - road_yaw + 180) % 360 - 180)
    diff = min(diff, abs(diff - 180))   # fold: both ends of axis → diff < 45

    priority_arm = 'A' if diff < 45 else 'B'

    # Keep exactly one GREEN head during emergency.
    if priority_arm == 'A':
        best_idx = 0
        best_dist = float('inf')
        for i, l in enumerate(group_A):
            d = l.get_location().distance(loc)
            if d < best_dist:
                best_dist = d
                best_idx = i
        for i, l in enumerate(group_A):
            l.set_state(carla.TrafficLightState.Green if i == best_idx else carla.TrafficLightState.Red)
        for l in group_B:
            l.set_state(carla.TrafficLightState.Red)
    else:
        best_idx = 0
        best_dist = float('inf')
        for i, l in enumerate(group_B):
            d = l.get_location().distance(loc)
            if d < best_dist:
                best_dist = d
                best_idx = i
        for l in group_A:
            l.set_state(carla.TrafficLightState.Red)
        for i, l in enumerate(group_B):
            l.set_state(carla.TrafficLightState.Green if i == best_idx else carla.TrafficLightState.Red)
    return priority_arm

# ── Emergency spawner ──────────────────────────────────────────────────────
emergency_vehicle = None

def _safe_destroy_actor(actor, tracked_list=None):
    """Best-effort destroy that avoids CARLA 'actor not found' teardown noise."""
    if actor is None:
        return False
    try:
        if not actor.is_alive:
            return False
    except RuntimeError:
        return False

    try:
        if world.get_actor(actor.id) is None:
            return False
    except RuntimeError:
        return False

    try:
        actor.destroy()
        if tracked_list is not None:
            try:
                tracked_list.remove(actor)
            except ValueError:
                pass
        return True
    except RuntimeError:
        return False

def _clear_spawn_point(sp, radius=5.0):
    """Destroy any vehicle blocking a spawn point so the spot is free."""
    for other in world.get_actors().filter("vehicle.*"):
        if other.is_alive and other.get_location().distance(sp.location) < radius:
            _safe_destroy_actor(other, tracked_list=vehicles)

def spawn_emergency():
    """
    Spawn an emergency vehicle at 30–65 m from the intersection so it is
    immediately visible in BOTH CAM 1 (20 m back) and CAM 2 (55 m overhead).
    Blueprint priority: ambulance → firetruck → police → any 4-wheel vehicle.

        Strategy (most-reliable first):
      1. Use pre-cached approach-arm candidates, clearing any blocking vehicle first.
            2. Widen to 30–75 m on the same approach arm, clearing blockers.
      3. Global last resort across all map spawn points.
    """
    lib = world.get_blueprint_library()
    ambulance_bps = list(lib.filter("vehicle.*ambulance*"))
    fallback_bps  = (list(lib.filter("vehicle.*firetruck*")) +
                     list(lib.filter("vehicle.*police*")))
    generic_bps   = [bp for bp in lib.filter("vehicle.*")
                     if int(bp.get_attribute('number_of_wheels').as_int()) == 4]
    bps = ambulance_bps if ambulance_bps else fallback_bps
    if not bps:
        bps = generic_bps
    if not bps:
        print("[EMERGENCY] No vehicle blueprints found — cannot spawn.")
        return None

    def _try_spawn(sp):
        bp_choice = random.choice(bps)
        if bp_choice.has_attribute('role_name'):
            bp_choice.set_attribute('role_name', 'emergency')
        return world.try_spawn_actor(bp_choice, sp)

    v = None

    # ── PRIMARY: approach-arm candidates at 30–65 m ──────────────────────
    # Clear any blocking vehicle first, then spawn.
    for sp in _approach_spawn_candidates:
        _clear_spawn_point(sp)
        v = _try_spawn(sp)
        if v:
            break

    # ── SECONDARY: widen slightly but stay within overview camera footprint ─
    if not v:
        all_sps = world.get_map().get_spawn_points()
        # Sort farthest-first while still visible in CAM 2.
        arm_sps = sorted(
            [sp for sp in all_sps if 30 < sp.location.distance(center) < 75],
            key=lambda sp: sp.location.distance(center), reverse=True
        )
        for sp in arm_sps:
            _clear_spawn_point(sp)
            v = _try_spawn(sp)
            if v:
                break

    # ── LAST RESORT: fallback candidates near intersection for visibility ─
    if not v:
        fallback_sps = list(_approach_spawn_candidates)
        random.shuffle(fallback_sps)
        for sp in fallback_sps:
            _clear_spawn_point(sp)
            v = _try_spawn(sp)
            if v:
                break

    if not v:
        print("[EMERGENCY] Spawn failed even after clearing — skipping.")
        return None

    # Force the vehicle to face the intersection regardless of spawn-point yaw.
    # CARLA spawn points have fixed orientations that often point away from
    # the junction — without this the vehicle drives off in the wrong direction.
    t   = v.get_transform()
    dx  = center.x - t.location.x
    dy  = center.y - t.location.y
    t.rotation.yaw = math.degrees(math.atan2(dy, dx))
    v.set_transform(t)
    world.tick()   # let physics settle before enabling autopilot

    loc = v.get_location()
    print(f"[EMERGENCY] Spawned {v.type_id} at "
          f"({loc.x:.1f}, {loc.y:.1f})  dist={loc.distance(center):.0f} m  "
          f"yaw={t.rotation.yaw:.1f}°")

    v.set_autopilot(True)
    traffic_manager.ignore_lights_percentage(v, 100)
    traffic_manager.ignore_signs_percentage(v, 100)
    traffic_manager.ignore_vehicles_percentage(v, 100)
    traffic_manager.force_lane_change(v, False)
    traffic_manager.auto_lane_change(v, False)
    traffic_manager.vehicle_percentage_speed_difference(v, -100)  # max speed
    traffic_manager.distance_to_leading_vehicle(v, 0)

    # Clear the entire lane between spawn point and intersection so physics
    # cannot stop the vehicle.  20 m was too short — now covers the full arm.
    emg_loc = v.get_location()
    emg_fwd = v.get_transform().get_forward_vector()
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == v.id or not other.is_alive:
            continue
        o_loc = other.get_location()
        if emg_loc.distance(o_loc) > 70:   # was 20 m — clear full approach arm
            continue
        dx2, dy2 = o_loc.x - emg_loc.x, o_loc.y - emg_loc.y
        dot = dx2 * emg_fwd.x + dy2 * emg_fwd.y
        if dot > 0:    # only vehicles ahead
            _safe_destroy_actor(other, tracked_list=vehicles)

    return v

# ── Dashboard ──────────────────────────────────────────────────────────────
DASH_W   = 420
HIST_LEN = 120

def draw_dashboard(phases, yolo_counts, stats_list, system_mode,
                   wait_history, tick_count,
                   yolo_conf=0.0, emg_count=0, fb_count=0):
    H, W  = 760, DASH_W
    panel = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)

    PH_TXT = ['GREEN', 'YELLOW', 'RED']
    PH_COL = [(0, 210, 0), (0, 210, 230), (0, 0, 230)]

    # ── Title bar ─────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (W, 54), (28, 28, 50), -1)
    cv2.putText(panel, "TRAFFIC INTELLIGENCE SYSTEM",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 1)
    cv2.putText(panel, f"Tick {tick_count:>6d}   |   Det. range {ROAD_DETECTION_DIST} m",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 160), 1)
    cv2.line(panel, (0, 54), (W, 54), (60, 60, 100), 1)

    # ── Sensing / view legend ─────────────────────────────────────────────
    cv2.rectangle(panel, (4, 58), (W-4, 100), (22, 22, 40), -1)
    cv2.circle(panel,  (14, 70), 5, (0, 220, 130), -1)
    cv2.putText(panel, "CONTROL INPUT: Ground Sensors (N/S/E/W)",
                (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 130), 1)
    cv2.circle(panel,  (14, 88), 5, (180, 255, 100), -1)
    cv2.putText(panel, "CAM 2  Intersection Overview (Flow)",
                (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 100), 1)
    cv2.line(panel, (0, 100), (W, 100), (50, 50, 80), 1)

    # ── Intersection card ─────────────────────────────────────────────────
    idx   = 0
    stats = stats_list[0]
    ph    = phases[0]
    sy    = 106

    cv2.rectangle(panel, (4, sy), (W-4, sy+358), (24, 24, 42), -1)
    cv2.rectangle(panel, (4, sy), (W-4, sy+358), (50, 50, 80),  1)

    # Header
    cv2.rectangle(panel, (4, sy), (W-4, sy+28), (38, 38, 65), -1)
    cv2.putText(panel, "  INTERSECTION",
                (6, sy+21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 255), 1)

    # ── Mode badge (covers all 6 states) ─────────────────────────────────
    _MODE_BADGE = {
        'EMERGENCY': ("EMERGENCY OVERRIDE",  (120, 120, 255), ( 50,   0, 100)),
        'PRE_CLEAR': ("PRE-CLEARANCE",        (255, 120,   0), (100,  50,   0)),
        'GRACE':     ("GRACE PERIOD",         (  0, 215, 255), (  0,  40,  70)),
        'RECOVERY':  ("POST-EMRG RECOVERY",   (  0, 200, 230), (  0,  45,  70)),
        'FALLBACK':  ("FIXED-TIME FALLBACK",  (  0, 180, 255), (  0,  40,  80)),
    }
    mtxt, mcol, mbg = _MODE_BADGE.get(system_mode,
                                      ("RL CONTROL", (0, 230, 110), (0, 50, 20)))
    cv2.rectangle(panel, (6, sy+30), (W-6, sy+52), mbg, -1)
    cv2.putText(panel, f"MODE: {mtxt}",
                (10, sy+46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mcol, 1)

    # ── Signal lights ─────────────────────────────────────────────────────
    for pi in range(3):
        cx  = 28 + pi * 44
        col = PH_COL[pi] if pi == ph else (28, 28, 28)
        cv2.circle(panel, (cx, sy + 90), 18, col, -1)
        if pi == ph:
            cv2.circle(panel, (cx, sy + 90), 18, (255, 255, 255), 1)
    cv2.putText(panel, PH_TXT[ph],
                (168, sy + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PH_COL[ph], 2)

    # ── Stats rows ────────────────────────────────────────────────────────
    q, aw, yc = stats['queue_length'], stats['avg_waiting_time'], yolo_counts[idx]
    tp = stats.get('throughput_vpm', 0.0)
    aw_col = (0, 70, 220) if aw > 15 else (0, 200, 100)
    for ri, (lbl, val, col) in enumerate([
        ("Vehicles (Sensors)",  f"{yc}",              (180, 180, 180)),
        ("Queue Length",     f"{q}",               (180, 180, 180)),
        ("Avg Wait",         f"{aw:.1f} s",         aw_col),
        ("Throughput",       f"{tp:.1f} vpm",      (150, 220, 150)),
        ("Det. Range",       f"{ROAD_DETECTION_DIST} m", (150, 200, 255)),
    ]):
        ry = sy + 126 + ri * 25
        cv2.putText(panel, lbl, (10, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 120, 150), 1)
        cv2.putText(panel, val, (W - 80, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

    # ── Queue bar ─────────────────────────────────────────────────────────
    bx0, bx1, by = 10, W-10, sy + 256
    fill = int((bx1 - bx0) * min(q, 20) / 20)
    cv2.rectangle(panel, (bx0, by), (bx1, by+12), (35, 35, 50), -1)
    if fill > 0:
        cv2.rectangle(panel, (bx0, by), (bx0+fill, by+12),
                      (0, 60, 210) if q > 10 else (0, 185, 80), -1)
    cv2.putText(panel, "QUEUE", (bx1-52, by+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 100), 1)

    # ── Sensor confidence bar ─────────────────────────────────────────────
    conf_col = (0,200,80) if yolo_conf > 0.5 else (0,180,220) if yolo_conf > 0.35 else (0,60,220)
    cv2.putText(panel, f"Sensor Conf: {yolo_conf:.2f}",
                (10, sy+280), cv2.FONT_HERSHEY_SIMPLEX, 0.41, conf_col, 1)
    bar_w = int((W-28) * max(0.0, min(1.0, yolo_conf)))
    cv2.rectangle(panel, (10, sy+284), (W-18, sy+293), (40, 40, 55), -1)
    cv2.rectangle(panel, (10, sy+284), (10+bar_w, sy+293), conf_col, -1)
    thresh_x = 10 + int((W-28) * 0.35)
    cv2.line(panel, (thresh_x, sy+282), (thresh_x, sy+295), (0, 100, 200), 1)
    cv2.putText(panel, "0.35", (thresh_x-8, sy+308),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 140, 200), 1)

    # ── State detail badge ────────────────────────────────────────────────
    by2 = sy + 312
    if system_mode == 'EMERGENCY':
        cv2.rectangle(panel, (6, by2), (W-6, by2+46), (70, 0, 0), -1)
        cv2.putText(panel, "!! EMERGENCY ON APPROACH ROAD !!",
                    (8, by2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 90, 90), 1)
        cv2.putText(panel, "ROAD SIGNAL: GREEN  (others: RED)",
                    (8, by2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 150, 150), 1)
    elif system_mode == 'FALLBACK':
        cv2.rectangle(panel, (6, by2), (W-6, by2+46), (0, 35, 65), -1)
        cv2.putText(panel, "SENSOR DEGRADED — FIXED-TIME SAFE MODE",
                    (8, by2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 180, 255), 1)
        cv2.putText(panel, "30s GREEN  5s YELLOW  30s RED per arm",
                    (8, by2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
    elif system_mode == 'PRE_CLEAR':
        cv2.rectangle(panel, (6, by2), (W-6, by2+46), (60, 40, 0), -1)
        cv2.putText(panel, "  PRE-CLEARANCE IN PROGRESS",
                    (8, by2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 180, 255), 1)
        cv2.putText(panel, "  All signals RED — clearing intersection",
                    (8, by2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 200, 255), 1)
    elif system_mode == 'GRACE':
        cv2.rectangle(panel, (6, by2), (W-6, by2+46), (0, 40, 80), -1)
        cv2.putText(panel, "  GRACE — mid-crossing vehicles clear",
                    (8, by2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 215, 255), 1)
        cv2.putText(panel, "  Pre-clearance phase pending ...",
                    (8, by2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 200, 255), 1)
    elif system_mode == 'RECOVERY':
        cv2.rectangle(panel, (6, by2), (W-6, by2+46), (0, 40, 55), -1)
        cv2.putText(panel, "  RECOVERY — draining post-emrg backlog",
                    (8, by2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 230), 1)
        cv2.putText(panel, "  RL resumes after backlog clears",
                    (8, by2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (100, 210, 240), 1)

    # ── Event counters ────────────────────────────────────────────────────
    ec_y = sy + 364
    cv2.line(panel, (0, ec_y - 5), (W, ec_y - 5), (50, 50, 80), 1)
    cv2.putText(panel, f"Emergency events: {emg_count}",
                (10, ec_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (150, 100, 255), 1)
    cv2.putText(panel, f"Fallback events:  {fb_count}",
                (10, ec_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 160, 255), 1)

    # ── Live wait-time graph ───────────────────────────────────────────────
    GT  = sy + 418
    GX0, GY0 = 8,   GT
    GX1, GY1 = W-8, min(H-32, GT + 220)
    GH, GW   = GY1 - GY0, GX1 - GX0

    cv2.rectangle(panel, (0, GT-18), (W, GT), (28, 28, 50), -1)
    cv2.putText(panel, "AVG WAIT TIME  (last 120 ticks)",
                (8, GT-4), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 130, 200), 1)
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
        cv2.circle(panel, (GX0 + 6, GY0 + 12), 4, (0, 220, 110), -1)
        cv2.putText(panel, "Avg Wait",
                    (GX0 + 13, GY0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (0, 220, 110), 1)

    cv2.putText(panel, f"{max_val:.0f}s",
                (GX0, GY0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 120), 1)
    cv2.putText(panel, "0s",
                (GX0, GY1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 120), 1)

    # ── Status bar ────────────────────────────────────────────────────────
    is_emg = system_mode == 'EMERGENCY'
    is_fb  = system_mode == 'FALLBACK'
    if is_emg:
        sb_bg, sb_col = (50, 0, 0),   (120, 80, 255)
        sb_txt = "EMERGENCY PRIORITY ACTIVE"
    elif is_fb:
        sb_bg, sb_col = (0, 35, 65),  (0, 160, 255)
        sb_txt = "FALLBACK: FIXED-TIME SAFE MODE"
    else:
        sb_bg, sb_col = (0, 28, 0),   (70, 200, 70)
        sb_txt = "SYSTEM: OPERATIONAL  |  DQN ACTIVE"
    cv2.rectangle(panel, (0, 729), (W, 760), sb_bg, -1)
    cv2.putText(panel, sb_txt, (6, 750),
                cv2.FONT_HERSHEY_SIMPLEX, 0.39, sb_col, 1)

    return panel

# ── Display window ─────────────────────────────────────────────────────────
WIN_NAME = "Traffic Intelligence System  |  CAM 2: Intersection Overview  |  Q = quit"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 760 + DASH_W, 760)  # Overview | Dashboard

# ── Main loop ──────────────────────────────────────────────────────────────
print(f"\nDemo running.")
print(f"  CAM 2 (Intersection / Overview): above centre, height 30 m, pitch -90° (true top-down), "
      f"yaw=0 (axis-aligned '+'), live flow + signal dots")
print(f"  Control input = Ground sensors (N/S/E/W per arm)")
print(f"  Display = CAM 2 overview + dashboard")
print(f"  Press Q to quit.\n")

YOLO_INTERVAL      = 3
EMERGENCY_INTERVAL = 800    # ~40 s between events (longer demo, clearer view)
EMERGENCY_LIFETIME = 420    # ~21 s for vehicle to transit
EMERGENCY_POST_GAP = 160    # ~8 s quiet period after clearance

tick_count     = 0
emg_counter    = EMERGENCY_INTERVAL
emg_age        = 0
emg_post_gap   = 0
emg_stuck_ticks = 0   # consecutive ticks the emergency vehicle was near-stationary
overview_frame = None   # latest annotated overview frame
wait_history   = [[] for _ in range(NUM_INT)]

try:
    while True:
        try:
            world.tick()
        except RuntimeError as e:
            print(f"[WARN] Simulation tick failed: {e}")
            break
        tick_count += 1

        # ── Emergency vehicle lifecycle ────────────────────────────────────
        if emergency_vehicle and emergency_vehicle.is_alive:
            ev_vel   = emergency_vehicle.get_velocity()
            ev_speed = math.sqrt(ev_vel.x**2 + ev_vel.y**2 + ev_vel.z**2)
            if ev_speed < 0.5:
                emg_stuck_ticks += 1
                if emg_stuck_ticks >= 5:
                    # Step 1: direct throttle override — bypasses TM for this tick.
                    emergency_vehicle.apply_control(carla.VehicleControl(
                        throttle=1.0, steer=0.0, brake=0.0,
                        hand_brake=False, reverse=False
                    ))
                    # Re-issue TM commands in case TM reset them.
                    traffic_manager.ignore_vehicles_percentage(emergency_vehicle, 100)
                    traffic_manager.vehicle_percentage_speed_difference(emergency_vehicle, -100)
                if emg_stuck_ticks >= 20:
                    # Step 2: truly stuck — teleport as last resort.
                    wp = carla_map.get_waypoint(emergency_vehicle.get_location(),
                                                project_to_road=True,
                                                lane_type=carla.LaneType.Driving)
                    if wp:
                        nexts = wp.next(8.0)
                        if nexts:
                            tf = nexts[0].transform
                            tf.location.z += 0.5
                            emergency_vehicle.set_transform(tf)
                            world.tick()
                    emg_stuck_ticks = 0
            else:
                emg_stuck_ticks = 0
            emg_age += 1
            if emg_age >= EMERGENCY_LIFETIME:
                _safe_destroy_actor(emergency_vehicle, tracked_list=vehicles)
                emergency_vehicle = None
                emg_age      = 0
                emg_counter  = 0
                emg_post_gap = EMERGENCY_POST_GAP
        else:
            if emg_post_gap > 0:
                emg_post_gap -= 1
            else:
                emg_counter += 1
                if emg_counter >= EMERGENCY_INTERVAL:
                    emergency_vehicle = spawn_emergency()
                    emg_age     = 0
                    # If spawn failed, retry after 60 ticks (3 s), not 1200
                    emg_counter = 0 if emergency_vehicle else EMERGENCY_INTERVAL - 60

        timestamp  = world.get_snapshot().timestamp.elapsed_seconds
        all_actors = list(world.get_actors().filter("vehicle.*"))

        # Ground-sensor control input (decision backend).
        emergency_inputs = ([emergency_vehicle]
                    if emergency_vehicle is not None and emergency_vehicle.is_alive
                    else [])
        sensor_result = ground_sensor.update(all_actors, emergency_inputs)
        latest_sensor_counts = dict(sensor_result['arm_counts'])
        control_count = sum(sensor_result['arm_counts'].values())
        control_conf = 0.95

        # ── Continuous vehicle replenishment ──────────────────────────────
        # Check every 200 ticks. If the nearby vehicle count has dropped
        # (emergency clear-path destroys up to 70 m of traffic), spawn fresh
        # vehicles from random approach-arm points so flow never stops.
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
                    if added >= need:
                        break
                    d = sp.location.distance(center)
                    if 30 < d < REPLENISH_RADIUS_M:
                        bp = random.choice(car_bps)
                        nv = world.try_spawn_actor(bp, sp)
                        if nv:
                            nv.set_autopilot(True)
                            vehicles.append(nv)
                            added += 1

        wait_trackers[0].update(all_actors, center, tick_count)
        gt_count, avg_speed = compute_gt(center)
        _wt_stats = wait_trackers[0].get_stats()
        latest_arm_stats = {
            'arm_queues': dict(latest_sensor_counts),
            'arm_avg_waits': dict(_wt_stats['arm_avg_waits']),
        }

        # ── Simulated low-confidence window (Scenario 4 — FALLBACK demo) ──
        # Triggers periodically when in DQN mode to demonstrate the safety
        # fallback without requiring actual YOLO sensor degradation.
        if system_mode == 'DQN':
            _sim_lc_timer += 1
        if not _sim_lc_active and _sim_lc_timer >= SIM_LC_INTERVAL and system_mode == 'DQN':
            _sim_lc_active  = True
            _sim_lc_dur_ctr = 0
            _sim_lc_timer   = 0
            print(f"[DEMO] tick {tick_count}: Simulating sensor degradation — "
                  f"sensor confidence forced low for {SIM_LC_DURATION} ticks.")
        if _sim_lc_active:
            _sim_lc_dur_ctr += 1
            if _sim_lc_dur_ctr >= SIM_LC_DURATION:
                _sim_lc_active  = False
                _sim_lc_dur_ctr = 0
        effective_control_conf = 0.10 if _sim_lc_active else control_conf

        # Update FallbackController every tick (pass emergency_flag=0 — emergency
        # is handled separately so confidence and emergency don't interfere).
        fallback_mode = fallback_ctrl.update(effective_control_conf, control_count, 0)

        # ── Refresh display frames at interval ─────────────────────────────
        if tick_count % YOLO_INTERVAL == 0:
            yolo_counts[0] = control_count
            yolo_conf = effective_control_conf

            # Overview camera: intersection flow frame
            ov = draw_overview_annotated(
                phases[0], system_mode == 'EMERGENCY', gt_count,
                emg_vehicle=emergency_vehicle,
                system_mode=system_mode
            )
            if ov is not None:
                overview_frame = ov

            # Verify light states every 50 ticks to catch multiple-green issues
            if tick_count % 50 == 0:
                verify_light_states()

            # Update rolling wait-time history
            s = wait_trackers[0].get_stats()
            wait_history[0].append(s['avg_waiting_time'])
            if len(wait_history[0]) > HIST_LEN:
                wait_history[0].pop(0)

        # ── State machine signal control ──────────────────────────────────
        # Step 1: detect every tick regardless of current state.
        emg_vehicles = get_emergency_vehicles(center)

        if system_mode == 'DQN':
            # ── SCENARIO 1: NORMAL — DQN adaptive control ─────────────────
            if emg_vehicles:
                # SCENARIO 2: EMERGENCY — detected on approach
                names = [v.type_id.split('.')[-1] for v in emg_vehicles]
                print(f"[EMERGENCY] tick {tick_count}: {len(emg_vehicles)} vehicle(s) "
                      f"({', '.join(names)}) — grace period ({GRACE_TICKS} ticks).")
                system_mode          = 'GRACE'
                grace_counter        = GRACE_TICKS
                emergency_crossed_center = False
                emergency_event_count += 1
            elif fallback_mode == ControlMode.FIXED_TIME:
                # SCENARIO 4: FALLBACK — low YOLO confidence
                system_mode          = 'FALLBACK'
                fallback_event_count += 1
                phases[0]            = 0
                counters[0]          = 0
                phase_durations[0]   = 600   # 30 s per phase
                set_phase(phases[0])
                print(f"[FALLBACK] tick {tick_count}: Low confidence "
                        f"({effective_control_conf:.2f}) — entering FIXED-TIME mode.")
            else:
                # Normal DQN control (3-phase for demo)
                arm_stats = wait_trackers[0].get_stats()
                arm_stats['arm_queues'] = dict(latest_sensor_counts)
                state = build_state(
                    arm_n_queue     = arm_stats['arm_queues']['N'],
                    arm_s_queue     = arm_stats['arm_queues']['S'],
                    arm_e_queue     = arm_stats['arm_queues']['E'],
                    arm_w_queue     = arm_stats['arm_queues']['W'],
                    arm_n_wait      = arm_stats['arm_avg_waits']['N'],
                    arm_s_wait      = arm_stats['arm_avg_waits']['S'],
                    arm_e_wait      = arm_stats['arm_avg_waits']['E'],
                    arm_w_wait      = arm_stats['arm_avg_waits']['W'],
                    current_phase   = phases[0],
                    phase_counter   = counters[0],
                    emergency_flag  = 0,
                    elapsed_seconds = timestamp
                )
                action = agents[0].act(state)

                # Map DQN 4-action output to the two movement groups in demo mode.
                desired_phase = 0 if action in (0, 1) else 2
                best_phase = _pick_best_green_phase(arm_stats, default_phase=desired_phase)

                # Complete yellow transition to pending green.
                if phases[0] == 1:
                    counters[0] += 1
                    if counters[0] >= phase_durations[0]:
                        phases[0] = pending_green_phase[0]
                        counters[0] = 0
                        phase_durations[0] = 100
                        set_phase(phases[0])
                else:
                    counters[0] += 1
                    current_pressure = _phase_pressure(arm_stats, phases[0])
                    desired_pressure = _phase_pressure(arm_stats, desired_phase)
                    best_pressure = _phase_pressure(arm_stats, best_phase)
                    ns_demand, ew_demand = _axis_demands(arm_stats)

                    if phases[0] == 0:
                        ns_starve_ticks = 0
                        ew_starve_ticks = ew_starve_ticks + 1 if ew_demand > DEMAND_EPS else 0
                    else:  # phases[0] == 2
                        ew_starve_ticks = 0
                        ns_starve_ticks = ns_starve_ticks + 1 if ns_demand > DEMAND_EPS else 0

                    # If agent picks a weak phase while another is clearly busier,
                    # bias toward the busier arm for demo clarity and fairness.
                    if best_pressure > DEMAND_EPS and desired_pressure < best_pressure * 0.75:
                        desired_phase = best_phase
                        desired_pressure = best_pressure

                    # If neither arm has demand, keep one stable green (avoid long all-red).
                    if current_pressure <= DEMAND_EPS and desired_pressure <= DEMAND_EPS:
                        hold_phase = phases[0] if phases[0] in (0, 2) else last_green_phase[0]
                        if hold_phase not in (0, 2):
                            hold_phase = 0
                        if phases[0] != hold_phase:
                            phases[0] = hold_phase
                            set_phase(phases[0])
                        counters[0] = 0
                        phase_durations[0] = 60
                        continue

                    # If the current arm is empty, do not keep serving it.
                    if current_pressure <= DEMAND_EPS and desired_pressure > DEMAND_EPS:
                        phases[0] = desired_phase
                        counters[0] = 0
                        phase_durations[0] = 100
                        set_phase(desired_phase)
                        continue

                    # Adaptive green cap: don't over-hold any one side.
                    if current_pressure < 1.0:
                        max_green = 30
                    elif current_pressure < 4.0:
                        max_green = 60
                    elif current_pressure < 8.0:
                        max_green = 90
                    else:
                        max_green = 130

                    min_green = 18 if current_pressure < 2.0 else 32
                    should_switch = False

                    # Anti-starvation: force service if an axis waits too long with demand.
                    if ew_starve_ticks >= 60 and ew_demand > DEMAND_EPS and counters[0] >= 12:
                        desired_phase = 2
                        desired_pressure = ew_demand
                        should_switch = True
                    elif ns_starve_ticks >= 60 and ns_demand > DEMAND_EPS and counters[0] >= 12:
                        desired_phase = 0
                        desired_pressure = ns_demand
                        should_switch = True

                    if (best_phase != phases[0] and
                            best_pressure > max(DEMAND_EPS, current_pressure * 1.20) and
                            counters[0] >= 12):
                        desired_phase = best_phase
                        should_switch = True

                    if desired_pressure > DEMAND_EPS and desired_phase != phases[0] and counters[0] >= min_green:
                        if desired_pressure > current_pressure * 1.05 or current_pressure < 1.0:
                            should_switch = True

                    if counters[0] >= max_green and desired_pressure > DEMAND_EPS:
                        should_switch = True
                        if desired_phase == phases[0]:
                            desired_phase = 2 if phases[0] == 0 else 0

                    if should_switch:
                        pending_green_phase[0] = desired_phase
                        phases[0] = 1
                        counters[0] = 0
                        phase_durations[0] = 18
                        set_phase(1)

        elif system_mode == 'FALLBACK':
            # ── SCENARIO 4: FALLBACK — fixed 30s cycle, safety fallback ───
            if emg_vehicles:
                # Emergency always overrides fallback immediately
                names = [v.type_id.split('.')[-1] for v in emg_vehicles]
                print(f"[EMERGENCY overrides FALLBACK] tick {tick_count}: "
                      f"{', '.join(names)} detected.")
                system_mode          = 'GRACE'
                grace_counter        = GRACE_TICKS
                emergency_crossed_center = False
                emergency_event_count += 1
            elif fallback_mode == ControlMode.DQN:
                # Confidence recovered — resume DQN
                system_mode = 'DQN'
                print(f"[FALLBACK→DQN] tick {tick_count}: "
                        f"Confidence recovered ({effective_control_conf:.2f}) — "
                      f"resuming RL control.")
            else:
                # Fixed-time cycling: 30 s green A, 30 s green B
                counters[0] += 1
                if counters[0] >= phase_durations[0]:
                    counters[0] = 0
                    phases[0]   = (phases[0] + 1) % len(phase_states)
                    phase_durations[0] = 20 if phases[0] == 1 else 600
                    set_phase(phases[0])

        elif system_mode == 'GRACE':
            # ── Hold signals — let mid-crossing vehicles clear ─────────────
            grace_counter -= 1
            if grace_counter <= 0:
                if emg_vehicles:
                    system_mode      = 'PRE_CLEAR'
                    preclear_counter = PRECLEAR_TICKS
                    set_all_red()
                    verify_light_states()  # Confirm all RED during pre-clear
                    print(f"[PRE_CLEAR] Grace expired — all RED for "
                          f"{PRECLEAR_TICKS} ticks.")
                    phases[0] = 2
                else:
                    system_mode      = 'RECOVERY'
                    recovery_counter = RECOVERY_TICKS
                    phases[0]        = 2
                    counters[0]      = 0
                    phase_durations[0] = 100
                    set_phase(2)
                    print(f"[EMERGENCY] Cleared during grace — RECOVERY.")

        elif system_mode == 'PRE_CLEAR':
            # ── All RED — ensure intersection empty before ambulance GREEN ─
            preclear_counter -= 1
            if preclear_counter <= 0:
                if emg_vehicles:
                    system_mode = 'EMERGENCY'
                    emg_speed = math.sqrt(
                        emg_vehicles[0].get_velocity().x**2 +
                        emg_vehicles[0].get_velocity().y**2 +
                        emg_vehicles[0].get_velocity().z**2
                    )
                    emergency_timeout = (int(MIN_EMERGENCY_GREEN * 1.5)
                                         if emg_speed < 8.0
                                         else MIN_EMERGENCY_GREEN)
                    if emg_speed < 8.0:
                        print(f"[EMERGENCY] Speed {emg_speed:.1f} m/s — "
                              f"extending GREEN to {emergency_timeout} ticks.")
                    arm = set_signals_for_emergency(emg_vehicles)
                    emergency_crossed_center = False
                    phases[0] = 0 if arm == 'A' else 2
                    counters[0] = 0
                    phase_durations[0] = 9999
                    print(f"[EMERGENCY] Override ACTIVE (arm {arm}): "
                          f"approach GREEN, all others RED.")
                    verify_light_states()  # Confirm emergency signals applied
                else:
                    system_mode = 'DQN'
                    print(f"[PRE_CLEAR] Vehicle cleared — back to DQN.")

        elif system_mode == 'EMERGENCY':
            # ── SCENARIO 2: EMERGENCY — hold GREEN for ambulance ──────────
            tracked_emg = emergency_vehicle if (emergency_vehicle and emergency_vehicle.is_alive) else None
            priority_emg = emg_vehicles
            if (not priority_emg and tracked_emg is not None and
                    tracked_emg.get_location().distance(center) <= ROAD_DETECTION_DIST + 35):
                priority_emg = [tracked_emg]

            if priority_emg:
                closest = min(priority_emg,
                              key=lambda v: v.get_location().distance(center))
                if closest.get_location().distance(center) < 12.0:
                    emergency_crossed_center = True

                arm = set_signals_for_emergency(priority_emg)
                phases[0] = 0 if arm == 'A' else 2
                counters[0] = 0
                phase_durations[0] = 9999
                emergency_timeout -= 1
                if emergency_timeout <= 0:
                    system_mode = 'RECOVERY'
                    recovery_counter = RECOVERY_TICKS
                    phases[0] = 2; counters[0] = 0; phase_durations[0] = 100
                    set_phase(2)
                    print(f"[EMERGENCY→RECOVERY] Timeout — resuming fixed cycle.")
            else:
                # If center crossing has not been observed yet, tolerate short dropouts.
                if not emergency_crossed_center and emergency_timeout > 0:
                    emergency_timeout -= 1
                    if tracked_emg is not None:
                        arm = set_signals_for_emergency([tracked_emg])
                        phases[0] = 0 if arm == 'A' else 2
                        counters[0] = 0
                        phase_durations[0] = 9999
                    continue
                print(f"[EMERGENCY] Cleared at tick {tick_count}. "
                      f"RECOVERY for {RECOVERY_TICKS} ticks.")
                system_mode = 'RECOVERY'
                recovery_counter = RECOVERY_TICKS
                phases[0] = 2; counters[0] = 0; phase_durations[0] = 100
                set_phase(2)

        elif system_mode == 'RECOVERY':
            # ── SCENARIO 3: RECOVERY — fixed cycle drains backlog ─────────
            recovery_counter -= 1
            counters[0] += 1
            if counters[0] >= phase_durations[0]:
                counters[0] = 0
                phases[0] = (phases[0] + 1) % len(phase_states)
                phase_durations[0] = 20 if phases[0] == 1 else 100
                set_phase(phases[0])
            if recovery_counter <= 0:
                system_mode = 'DQN'
                phase_durations[0] = 100
                print(f"[RECOVERY] Complete at tick {tick_count}. Resuming DQN.")

        # ── Siren ──────────────────────────────────────────────────────────
        if system_mode in ('GRACE', 'EMERGENCY'):
            start_siren()
        else:
            stop_siren()

        # ── Render — Overview | Dashboard ──────────────────────────────────
        if overview_frame is not None:
            stats_now = [wait_trackers[0].get_stats()]
            dash = draw_dashboard(phases, yolo_counts, stats_now,
                                  system_mode, wait_history, tick_count,
                                  yolo_conf=effective_control_conf,
                                  emg_count=emergency_event_count,
                                  fb_count=fallback_event_count)
            cv2.imshow(WIN_NAME, np.hstack([overview_frame, dash]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
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
