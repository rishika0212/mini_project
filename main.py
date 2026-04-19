"""
main.py
=======
Complete integrated system:
  CARLA → RGB Camera → YOLOv8 → Confidence Check
        → DQN Agent (4-phase intelligent switching)
    → Signal Control (GREEN → YELLOW → ALL_RED → GREEN)
    → Waiting Time Tracking (per-arm)
    → CSV Logging

Signal phases (DQN chooses next phase):
  Phase 0: NS Straight + Right  (N & S green, E & W red)
  Phase 1: NS Left turns        (N & S green protected, E & W red)
  Phase 2: EW Straight + Right  (E & W green, N & S red)
  Phase 3: EW Left turns        (E & W green protected, N & S red)

Transitions (hardcoded, not controlled by DQN):
  GREEN (5–30 s) → YELLOW (3 s) → ALL_RED (1 s) → GREEN (next phase)

Usage:
    python main.py
    python main.py --model vehicle_detector.pt  (custom trained model)
"""

import carla
import random
import os
import csv
import math
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
import threading

from dqn_agent    import DQNAgent, EpisodeTracker, build_state, compute_reward
from waiting_time import IntersectionWaitingTimeTracker
from fallback     import get_yolo_confidence
from system_controller import TrafficSystemController, SystemMode

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolov8n.pt',
                    help='YOLO model path (default: yolov8n.pt)')
parser.add_argument('--train', action='store_true',
                    help='Train DQN (default: evaluation mode)')
parser.add_argument('--no-yolo', action='store_true',
                    help='Skip YOLO (faster training, use ground-truth vehicle counts)')
args = parser.parse_args()

print(f"\n{'='*55}")
print(f"  Traffic Intelligence System — 4-Phase DQN")
print(f"  Mode:  {'TRAINING' if args.train else 'EVALUATION'}")
print(f"  YOLO:  {args.model}")
print(f"{'='*55}\n")

# ── Connect ────────────────────────────────────────────────────────────────
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world  = client.get_world()

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode   = False
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.global_percentage_speed_difference(-20)  # 20% faster than default
traffic_manager.set_global_distance_to_leading_vehicle(2.0)  # 2m safe distance
print("Connected to CARLA.")

# ── YOLOv8 ─────────────────────────────────────────────────────────────────
print(f"Loading YOLO model: {args.model}")
model           = YOLO(args.model)
# Enable GPU (CUDA:0) if available; falls back to CPU automatically
if not args.no_yolo:
    model.to('cuda')
VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
EMERGENCY_KW    = ['ambulance','firetruck','police']
print("YOLO ready." if not args.no_yolo else "YOLO disabled (using ground-truth counts).")

# ── DQN agents ─────────────────────────────────────────────────────────────
NUM_INT = 3
agents, trackers, controllers = [], [], []

for i in range(NUM_INT):
    agent   = DQNAgent()
    tracker = EpisodeTracker(episode_length=500)
    tracker.set_int_id(i+1)

    if not args.train:
        agent.epsilon = 0.0   # greedy during evaluation

    agent.load(f"data/dqn_weights_int{i+1}.json")

    if not args.train:
        agent.epsilon = 0.0

    # Create system controller for this intersection
    controller = TrafficSystemController(intersection_id=i+1, dqn_agent=agent, num_phases=4)

    agents.append(agent)
    trackers.append(tracker)
    controllers.append(controller)

print(f"DQN agents ready ({'training' if args.train else 'greedy eval'} mode).")

# ── Shared frames ──────────────────────────────────────────────────────────
latest_frames = {}
frame_locks   = {}
frame_counts  = {}

# ── Intersections ──────────────────────────────────────────────────────────
def group_lights(lights, threshold=45):
    groups = []
    for light in lights:
        loc = light.get_location()
        added = False
        for group in groups:
            if loc.distance(group[0].get_location()) < threshold:
                group.append(light); added = True; break
        if not added:
            groups.append([light])
    return groups

traffic_lights       = world.get_actors().filter("traffic.traffic_light")
groups               = group_lights(traffic_lights, threshold=45)
valid_groups         = [g for g in groups if len(g) >= 3]
intersections        = valid_groups[:NUM_INT]
intersection_centers = []

for i, group in enumerate(intersections):
    x = sum(l.get_location().x for l in group)/len(group)
    y = sum(l.get_location().y for l in group)/len(group)
    z = sum(l.get_location().z for l in group)/len(group)
    intersection_centers.append(carla.Location(x=x, y=y, z=z))
    print(f"  Int {i+1}: ({x:.1f}, {y:.1f})")

ROI_RADIUS    = 50
wait_trackers = [IntersectionWaitingTimeTracker(i+1, ROI_RADIUS, intersection_centers[i])
                 for i in range(NUM_INT)]

# ── Vehicles ───────────────────────────────────────────────────────────────
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                and not any(kw in bp.id.lower() for kw in EMERGENCY_KW)]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

for sp in spawn_points[:180]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v: v.set_autopilot(True); vehicles.append(v)

for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center)<80]
    n = 0
    for sp in nearby:
        if n >= 80: break
        bp = random.choice(car_bps)
        v  = world.try_spawn_actor(bp, sp)
        if v: v.set_autopilot(True); vehicles.append(v); n += 1

print(f"Spawned {len(vehicles)} vehicles.")
for _ in range(20): world.tick()

# ── Cameras ────────────────────────────────────────────────────────────────
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x','640')
camera_bp.set_attribute('image_size_y','640')
camera_bp.set_attribute('fov','90')
camera_bp.set_attribute('sensor_tick','0.1')

cameras, camera_int_map = [], []

for i, center in enumerate(intersection_centers):
    transform = carla.Transform(
        carla.Location(x=center.x-20, y=center.y, z=center.z+12),
        carla.Rotation(pitch=-45, yaw=0)
    )
    cam = world.spawn_actor(camera_bp, transform)
    cameras.append(cam)
    camera_int_map.append(i)
    latest_frames[i] = None
    frame_locks[i]   = threading.Lock()
    frame_counts[i]  = 0

def on_image(image, idx):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:,:,:3][:,:,::-1].copy()
    with frame_locks[idx]:
        latest_frames[idx] = rgb; frame_counts[idx] += 1

for idx, cam in enumerate(cameras):
    cam.listen(lambda img, i=idx: on_image(img, i))

print("Cameras attached. Warming up...")
for _ in range(30): world.tick()
print(f"  Frames: {[frame_counts[i] for i in range(len(cameras))]}")

# ── YOLO with confidence ───────────────────────────────────────────────────
yolo_counts = [0]*NUM_INT
yolo_confs  = [0.0]*NUM_INT

YOLO_SNAPSHOT_INTERVAL = 3000

def run_yolo(camera_idx, tick):
    with frame_locks[camera_idx]:
        frame = latest_frames[camera_idx]
    if frame is None:
        return 0, 0.0, None

    results  = model(frame, verbose=False, conf=0.25)
    avg_conf, count = get_yolo_confidence(results)
    annotated = frame[:,:,::-1].copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(annotated,
                            f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}",
                            (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,(0,255,0),1)

    cv2.putText(annotated, f"Vehicles:{count} Conf:{avg_conf:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

    if tick % YOLO_SNAPSHOT_INTERVAL == 0:
        os.makedirs("data/snapshots", exist_ok=True)
        cv2.imwrite(f"data/snapshots/int{camera_idx+1}_t{tick:07d}.png",
                    annotated)

    return count, avg_conf, annotated

# ── Ground truth avg speed ─────────────────────────────────────────────────
def compute_avg_speed(center, all_actors):
    c, ss = 0, 0.0
    for v in all_actors:
        if v.get_location().distance(center) < ROI_RADIUS:
            c += 1
            vel = v.get_velocity()
            ss += math.sqrt(vel.x**2+vel.y**2+vel.z**2)
    return ss/c if c > 0 else 0.0

def is_emergency_vehicle(v):
    role = str(v.attributes.get('role_name', '')).lower()
    return role == 'emergency' or any(kw in v.type_id.lower() for kw in EMERGENCY_KW)

# ── Emergency ──────────────────────────────────────────────────────────────
def find_emergency_in_roi(center):
    for v in world.get_actors().filter("vehicle.*"):
        if not v.is_alive:
            continue
        if v.get_location().distance(center) < ROI_RADIUS:
            if is_emergency_vehicle(v):
                return v
    return None

def set_all_red(idx, intersection_arms):
    """Set all signals to RED for intersection idx — used during pre-clearance."""
    arm_ns, arm_ew = intersection_arms[idx]
    for l in arm_ns + arm_ew:
        l.set_state(carla.TrafficLightState.Red)

def set_emergency_phase(idx, emg_vehicle):
    """Give GREEN to the arm where emergency vehicle is approaching from."""
    if emg_vehicle is None or not emg_vehicle.is_alive:
        return

    arm_ns, arm_ew = intersection_arms[idx]
    center = intersection_centers[idx]
    emg_loc = emg_vehicle.get_location()

    # Determine which arm the ambulance is on by angular difference
    dx = emg_loc.x - center.x
    dy = emg_loc.y - center.y
    emg_angle = math.degrees(math.atan2(dy, dx))

    # Find which arm is closest to ambulance direction
    ns_diffs = []
    ew_diffs = []

    for l in arm_ns:
        light_loc = l.get_location()
        ldx = light_loc.x - center.x
        ldy = light_loc.y - center.y
        light_angle = math.degrees(math.atan2(ldy, ldx))
        diff = abs((emg_angle - light_angle + 180) % 360 - 180)
        ns_diffs.append(diff)

    for l in arm_ew:
        light_loc = l.get_location()
        ldx = light_loc.x - center.x
        ldy = light_loc.y - center.y
        light_angle = math.degrees(math.atan2(ldy, ldx))
        diff = abs((emg_angle - light_angle + 180) % 360 - 180)
        ew_diffs.append(diff)

    # Whichever arm is closer gets GREEN
    ns_is_closer = min(ns_diffs) if ns_diffs else 180 < min(ew_diffs) if ew_diffs else True

    if ns_is_closer:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Green)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Red)
    else:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Red)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Green)

# ── Rush hour scenario ──────────────────────────────────────────────────────
RUSH_HOUR_INTERVAL  = 2400      # ~2 minutes — realistic traffic surge
RUSH_HOUR_BATCH     = 35        # vehicles added per surge event
rush_vehicles       = []
rush_hour_counter   = 0

def spawn_rush_hour():
    added = 0
    all_sps = world.get_map().get_spawn_points()
    for center in intersection_centers:
        nearby = [sp for sp in all_sps if sp.location.distance(center) < 90]
        random.shuffle(nearby)
        n = 0
        for sp in nearby:
            if n >= RUSH_HOUR_BATCH:
                break
            bp = random.choice(car_bps)
            v  = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True)
                rush_vehicles.append(v)
                n += 1; added += 1
    if added:
        print(f"  [Rush Hour] Spawned {added} extra vehicles across all intersections")

emergency_vehicle = None

def spawn_emergency():
    lib = world.get_blueprint_library()
    ambulance_bps = list(lib.filter("vehicle.*ambulance*"))
    fallback_bps  = (list(lib.filter("vehicle.*firetruck*")) +
                     list(lib.filter("vehicle.*police*")))
    generic_bps   = [bp for bp in lib.filter("vehicle.*")
                     if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
    bps = ambulance_bps if ambulance_bps else fallback_bps
    if not bps:
        bps = generic_bps
    if not bps:
        return None

    center = random.choice(intersection_centers)
    all_sps = world.get_map().get_spawn_points()
    nearby = [sp for sp in all_sps if sp.location.distance(center) < 180]
    candidates = nearby if nearby else all_sps
    random.shuffle(candidates)

    v = None
    for sp in candidates[:150]:
        bp = random.choice(bps)
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'emergency')
        v = world.try_spawn_actor(bp, sp)
        if v:
            break

    if v:
        v.set_autopilot(True)
        traffic_manager.ignore_lights_percentage(v, 100)
        traffic_manager.ignore_signs_percentage(v, 100)
        traffic_manager.ignore_vehicles_percentage(v, 100)
        traffic_manager.auto_lane_change(v, True)
        traffic_manager.vehicle_percentage_speed_difference(v, -50)  # Reduced from -80 to -50
        traffic_manager.distance_to_leading_vehicle(v, 0.5)
    return v

# ── Signal phases ───────────────────────────────────────────────────────────
PHASE_NS_STRAIGHT = 0   # N & S straight + right  (NS green, EW red)
PHASE_NS_LEFT     = 1   # N & S protected left     (NS green, EW red)
PHASE_EW_STRAIGHT = 2   # E & W straight + right  (EW green, NS red)
PHASE_EW_LEFT     = 3   # E & W protected left     (EW green, NS red)

PHASE_NAMES = ['NS_St', 'NS_Lt', 'EW_St', 'EW_Lt']

# Transition timings (hardcoded — DQN does NOT control these)
YELLOW_TICKS   = 60    # 3 s at 0.05 s/tick
ALL_RED_TICKS  = 20    # 1 s at 0.05 s/tick
MIN_GREEN_TICKS = 100  # 5 s minimum green before a switch is allowed
MAX_GREEN_TICKS = 600  # 30 s maximum green before a forced switch

def _angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)

def split_intersection_arms(group):
    if not group:
        return [], []
    ref_yaw = group[0].get_transform().rotation.yaw
    arm_ns, arm_ew = [], []
    for light in group:
        lyaw = light.get_transform().rotation.yaw
        diff = _angle_diff(lyaw, ref_yaw)
        if diff < 45 or diff > 135:
            arm_ns.append(light)
        else:
            arm_ew.append(light)
    if not arm_ns or not arm_ew:
        mid = max(1, len(group) // 2)
        arm_ns = list(group[:mid])
        arm_ew = list(group[mid:])
    return arm_ns, arm_ew

intersection_arms = [split_intersection_arms(g) for g in intersections]

def apply_phase(idx, phase):
    """Apply a green phase: phases 0/1 → NS green, phases 2/3 → EW green."""
    arm_ns, arm_ew = intersection_arms[idx]
    if phase in (PHASE_NS_STRAIGHT, PHASE_NS_LEFT):
        for l in arm_ns: l.set_state(carla.TrafficLightState.Green)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Red)
    else:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Red)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Green)

def _apply_yellow(idx, from_phase):
    """Yellow on the arm that just had green; other arm stays red."""
    arm_ns, arm_ew = intersection_arms[idx]
    if from_phase in (PHASE_NS_STRAIGHT, PHASE_NS_LEFT):
        for l in arm_ns: l.set_state(carla.TrafficLightState.Yellow)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Red)
    else:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Red)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Yellow)

def _apply_all_red(idx):
    arm_ns, arm_ew = intersection_arms[idx]
    for l in arm_ns + arm_ew:
        l.set_state(carla.TrafficLightState.Red)

# Per-intersection transition state machine
# trans_state: 'GREEN' | 'YELLOW' | 'ALL_RED'
# emergency_state: 'NORMAL' | 'GRACE' | 'PRE_CLEAR' | 'EMERGENCY' | 'RECOVERY'
active_phases       = [PHASE_NS_STRAIGHT] * NUM_INT  # current green phase
phase_counters      = [0] * NUM_INT                  # ticks in current green
trans_state         = ['GREEN'] * NUM_INT
trans_counter       = [0] * NUM_INT                  # countdown for yellow/all_red
pending_phase       = [PHASE_NS_STRAIGHT] * NUM_INT  # next phase after transition
emergency_state     = ['NORMAL'] * NUM_INT           # emergency handling state
grace_counter       = [0] * NUM_INT                  # grace period countdown
preclear_counter    = [0] * NUM_INT                  # pre-clearance countdown
recovery_counter    = [0] * NUM_INT                  # recovery period countdown
emergency_timeout   = [0] * NUM_INT                  # adaptive GREEN time for ambulance

GRACE_TICKS         = 40     # ~2 s   — detect emergency, prepare
PRECLEAR_TICKS      = 40     # ~2 s   — all RED to ensure intersection clear
RECOVERY_TICKS      = 200    # ~10 s  — rebalance traffic
MIN_EMERGENCY_GREEN = 200    # ~10 s minimum GREEN for ambulance

for i in range(NUM_INT):
    apply_phase(i, PHASE_NS_STRAIGHT)

def _start_transition(idx, target_phase):
    """Initiate GREEN→YELLOW→ALL_RED→GREEN(target) sequence."""
    pending_phase[idx]  = target_phase
    trans_state[idx]    = 'YELLOW'
    trans_counter[idx]  = YELLOW_TICKS
    _apply_yellow(idx, active_phases[idx])

def _highest_pressure_phase(arm_stats):
    """Return the straight phase (0 or 2) with highest combined queue+wait pressure."""
    aq = arm_stats['arm_queues']
    aw = arm_stats['arm_avg_waits']
    ns = aq['N'] + aq['S'] + aw['N'] + aw['S']
    ew = aq['E'] + aq['W'] + aw['E'] + aw['W']
    return PHASE_NS_STRAIGHT if ns >= ew else PHASE_EW_STRAIGHT

# ── Green wave ──────────────────────────────────────────────────────────────
_int_order = sorted(range(NUM_INT),
                    key=lambda i: intersection_centers[i].x)

def _wave_offset(from_i, to_i):
    dist = intersection_centers[from_i].distance(intersection_centers[to_i])
    return max(100, int(dist / 8.3 / 0.05))

green_wave_due = [0] * NUM_INT

def trigger_green_wave(from_idx):
    pos = _int_order.index(from_idx)
    for step, downstream in enumerate(_int_order[pos+1:], start=1):
        if green_wave_due[downstream] == 0:
            green_wave_due[downstream] = _wave_offset(from_idx, downstream) * step

# ── CSV ────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
csv_file   = open("data/rl_states_final.csv","w",newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp","intersection_id",
    "yolo_count","yolo_confidence","vehicles_cleared","avg_speed",
    "active_phase","trans_state","action","reward",
    "control_mode","emergency_flag","system_mode",
    "fallback_mode","grace_counter","preclear_counter","recovery_counter",
    "emergency_timeout",
    "avg_waiting_time","max_waiting_time","queue_length",
    "throughput_vpm","epsilon","episode"
])

# ── Main loop ──────────────────────────────────────────────────────────────
print("\nSystem running. Press Ctrl+C to stop.\n")

YOLO_INTERVAL      = 10
EMERGENCY_INTERVAL = 240    # ~12 seconds between emergency events (realistic demo)
EMERGENCY_LIFETIME = 300    # ~15 seconds for vehicle to fully transit

tick_count       = 0
emg_counter      = 0
emg_age          = 0
prev_states      = [None]*NUM_INT
prev_actions     = [0]*NUM_INT
prev_throughput  = [0]*NUM_INT
modes            = ["DQN"] * NUM_INT

try:
    while True:
        world.tick()
        tick_count += 1

        # Rush hour surge
        rush_hour_counter += 1
        if rush_hour_counter >= RUSH_HOUR_INTERVAL:
            rush_hour_counter = 0
            spawn_rush_hour()

        # Emergency lifecycle
        emg_counter += 1
        if emergency_vehicle and emergency_vehicle.is_alive:
            emg_age += 1
            if emg_age >= EMERGENCY_LIFETIME:
                emergency_vehicle.destroy()
                emergency_vehicle = None; emg_age = 0
        if emg_counter >= EMERGENCY_INTERVAL:
            if emergency_vehicle is None:
                emergency_vehicle = spawn_emergency()
                emg_age = 0
            emg_counter = 0

        timestamp  = world.get_snapshot().timestamp.elapsed_seconds
        all_actors = list(world.get_actors().filter("vehicle.*"))

        # Update waiting time trackers (per-arm)
        for idx, center in enumerate(intersection_centers):
            wait_trackers[idx].update(all_actors, center, tick_count)

        # YOLO every 10 ticks (skip during training for speed)
        if tick_count % YOLO_INTERVAL == 0 and not args.no_yolo:
            for cam_i in range(len(cameras)):
                cnt, conf, _ = run_yolo(cam_i, tick_count)
                int_i        = camera_int_map[cam_i]
                yolo_counts[int_i] = cnt
                yolo_confs[int_i]  = conf

        # Per-intersection control
        for idx, center in enumerate(intersection_centers):
            avg_speed  = compute_avg_speed(center, all_actors)
            emg_v      = find_emergency_in_roi(center)
            emergency_flag = 1 if emg_v is not None else 0
            arm_stats  = wait_trackers[idx].get_stats()

            # Get current system mode from controller
            controllers[idx].set_active_phase(active_phases[idx])

            # Vehicles cleared (throughput delta)
            cur_throughput   = arm_stats['throughput_vpm']
            vehicles_cleared = max(0, round(cur_throughput - prev_throughput[idx]))
            prev_throughput[idx] = cur_throughput

            # Build 12-element state vector
            state = build_state(
                arm_n_queue     = arm_stats['arm_queues']['N'],
                arm_s_queue     = arm_stats['arm_queues']['S'],
                arm_e_queue     = arm_stats['arm_queues']['E'],
                arm_w_queue     = arm_stats['arm_queues']['W'],
                arm_n_wait      = arm_stats['arm_avg_waits']['N'],
                arm_s_wait      = arm_stats['arm_avg_waits']['S'],
                arm_e_wait      = arm_stats['arm_avg_waits']['E'],
                arm_w_wait      = arm_stats['arm_avg_waits']['W'],
                current_phase   = active_phases[idx],
                phase_counter   = phase_counters[idx],
                emergency_flag  = emergency_flag,
                elapsed_seconds = timestamp,
            )

            switching = False

            # Update system controller with confidence and emergency flag
            action_suggestion, sys_mode, signal_override = controllers[idx].update(
                state=state,
                yolo_confidence=yolo_confs[idx],
                yolo_count=yolo_counts[idx],
                emergency_flag=emergency_flag,
                emg_vehicle=emg_v
            )

            # Sync controller mode to ground-truth emergency state machine
            controllers[idx].sync_mode_from_main(emergency_state[idx])
            mode = controllers[idx].system_mode.value
            modes[idx] = mode

            # Get fallback controller status for diagnostics
            fb_status = controllers[idx].fallback.get_status()

            # Per-intersection emergency state machine
            # NOTE: NORMAL state falls through to the `else` DQN branch below;
            # emergency detection is handled first then control falls through.
            if emergency_state[idx] == 'NORMAL' and emergency_flag == 1:
                # Emergency detected — enter GRACE period
                emergency_state[idx] = 'GRACE'
                grace_counter[idx] = GRACE_TICKS

            elif emergency_state[idx] == 'GRACE':
                # Hold current state while mid-crossing vehicles clear
                grace_counter[idx] -= 1
                if grace_counter[idx] <= 0:
                    if emergency_flag == 1:
                        # Transition to PRE_CLEAR: all RED for 2-3 seconds
                        emergency_state[idx] = 'PRE_CLEAR'
                        preclear_counter[idx] = PRECLEAR_TICKS
                        set_all_red(idx, intersection_arms)
                        trans_state[idx] = 'GREEN'  # mark state
                    else:
                        # Vehicle cleared before grace expired
                        emergency_state[idx] = 'NORMAL'

            elif emergency_state[idx] == 'PRE_CLEAR':
                # All RED phase — ensure intersection is empty before giving green
                preclear_counter[idx] -= 1
                if preclear_counter[idx] <= 0:
                    if emergency_flag == 1:
                        # Pre-clearance complete — activate emergency override
                        emergency_state[idx] = 'EMERGENCY'
                        # Measure ambulance speed for adaptive GREEN time
                        if emg_v is not None and emg_v.is_alive:
                            ev = emg_v.get_velocity()
                            emg_speed = math.sqrt(ev.x**2 + ev.y**2 + ev.z**2)
                            # Bonus: Slower vehicles get extended GREEN time
                            if emg_speed < 8.0:
                                emergency_timeout[idx] = int(MIN_EMERGENCY_GREEN * 1.5)
                                controllers[idx].set_emergency_timeout(emergency_timeout[idx])
                            else:
                                emergency_timeout[idx] = MIN_EMERGENCY_GREEN
                                controllers[idx].set_emergency_timeout(emergency_timeout[idx])
                        set_emergency_phase(idx, emg_v)
                        active_phases[idx] = PHASE_NS_STRAIGHT
                        phase_counters[idx] = 0
                        trans_state[idx] = 'GREEN'
                    else:
                        # Vehicle cleared during pre-clear
                        emergency_state[idx] = 'NORMAL'

            elif emergency_state[idx] == 'EMERGENCY':
                # Give priority to emergency vehicle
                if emergency_flag == 1 and emg_v is not None and emg_v.is_alive:
                    set_emergency_phase(idx, emg_v)
                    emergency_timeout[idx] -= 1
                    if emergency_timeout[idx] <= 0:
                        # Timeout — force transition to recovery with proper signal sequence
                        emergency_state[idx] = 'RECOVERY'
                        recovery_counter[idx] = RECOVERY_TICKS
                        trans_state[idx] = 'YELLOW'  # Start transition sequence
                        trans_counter[idx] = YELLOW_TICKS
                        pending_phase[idx] = PHASE_NS_STRAIGHT
                        _apply_yellow(idx, active_phases[idx])
                else:
                    # Vehicle exited — transition to recovery with proper signal sequence
                    emergency_state[idx] = 'RECOVERY'
                    recovery_counter[idx] = RECOVERY_TICKS
                    trans_state[idx] = 'YELLOW'  # Start transition sequence
                    trans_counter[idx] = YELLOW_TICKS
                    pending_phase[idx] = PHASE_NS_STRAIGHT
                    _apply_yellow(idx, active_phases[idx])
                action = active_phases[idx]

            elif emergency_state[idx] == 'RECOVERY':
                # Fixed-time cycling to drain backed-up traffic
                # Use proper YELLOW→ALL_RED→GREEN transitions
                recovery_counter[idx] -= 1

                ts = trans_state[idx]

                if ts == 'GREEN':
                    phase_counters[idx] += 1
                    # Switch phase every 100 ticks (5 seconds) during recovery
                    if phase_counters[idx] >= 100:
                        target = (active_phases[idx] + 1) % 4
                        if target != active_phases[idx]:
                            _start_transition(idx, target)

                elif ts == 'YELLOW':
                    trans_counter[idx] -= 1
                    if trans_counter[idx] <= 0:
                        trans_state[idx] = 'ALL_RED'
                        trans_counter[idx] = ALL_RED_TICKS
                        _apply_all_red(idx)

                elif ts == 'ALL_RED':
                    trans_counter[idx] -= 1
                    if trans_counter[idx] <= 0:
                        active_phases[idx] = pending_phase[idx]
                        phase_counters[idx] = 0
                        trans_state[idx] = 'GREEN'
                        apply_phase(idx, active_phases[idx])

                # Exit recovery when counter expires
                if recovery_counter[idx] <= 0:
                    emergency_state[idx] = 'NORMAL'
                    active_phases[idx] = PHASE_NS_STRAIGHT
                    phase_counters[idx] = 0
                    trans_state[idx] = 'GREEN'
                    apply_phase(idx, PHASE_NS_STRAIGHT)

                action = active_phases[idx]

            else:  # NORMAL state — DQN or FIXED_TIME fallback
                # Check if FallbackController has switched to FIXED_TIME
                if fb_status.get('mode') == 'FIXED_TIME':
                    should_switch, next_phase = controllers[idx].fallback.get_fixed_time_action()
                    if should_switch and trans_state[idx] == 'GREEN':
                        target = next_phase % 4
                        if target != active_phases[idx]:
                            _start_transition(idx, target)
                            switching = True
                    action = active_phases[idx]
                else:  # DQN — adaptive 4-phase control
                    action = agents[idx].act(state)   # desired next phase (0-3)

                ts = trans_state[idx]

                if ts == 'GREEN':
                    phase_counters[idx] += 1

                    # Green wave override: force NS phase when wave arrives
                    if green_wave_due[idx] > 0:
                        green_wave_due[idx] -= 1
                        if (green_wave_due[idx] == 0 and
                                active_phases[idx] not in (PHASE_NS_STRAIGHT, PHASE_NS_LEFT)):
                            active_phases[idx]  = PHASE_NS_STRAIGHT
                            phase_counters[idx] = 0
                            apply_phase(idx, PHASE_NS_STRAIGHT)

                    # Forced switch when max green reached
                    if phase_counters[idx] >= MAX_GREEN_TICKS:
                        target = _highest_pressure_phase(arm_stats)
                        if target != active_phases[idx]:
                            _start_transition(idx, target)
                            switching = True
                    # Agent-requested switch (only after min green elapsed)
                    elif (action != active_phases[idx] and
                          phase_counters[idx] >= MIN_GREEN_TICKS):
                        _start_transition(idx, action)
                        switching = True

                elif ts == 'YELLOW':
                    trans_counter[idx] -= 1
                    if trans_counter[idx] <= 0:
                        trans_state[idx]   = 'ALL_RED'
                        trans_counter[idx] = ALL_RED_TICKS
                        _apply_all_red(idx)

                elif ts == 'ALL_RED':
                    trans_counter[idx] -= 1
                    if trans_counter[idx] <= 0:
                        active_phases[idx]  = pending_phase[idx]
                        phase_counters[idx] = 0
                        trans_state[idx]    = 'GREEN'
                        apply_phase(idx, active_phases[idx])
                        if active_phases[idx] in (PHASE_NS_STRAIGHT, PHASE_NS_LEFT):
                            trigger_green_wave(idx)

            reward = compute_reward(
                avg_speed        = avg_speed,
                emergency_flag   = emergency_flag,
                switching        = switching,
                avg_waiting_time = arm_stats['avg_waiting_time'],
                queue_length     = arm_stats['queue_length'],
                vehicles_cleared = vehicles_cleared,
            )

            # Train if in training mode
            if args.train and prev_states[idx] is not None:
                done = trackers[idx].is_done()
                agents[idx].remember(prev_states[idx], prev_actions[idx],
                                     reward, state, done)
                loss = agents[idx].replay()
                trackers[idx].update(reward, loss)
                if done:
                    print(f"\n[Int {idx+1}]", end=" ")
                    trackers[idx].next_episode(agents[idx])

            prev_states[idx]  = state
            prev_actions[idx] = action

            csv_writer.writerow([
                round(timestamp,3), idx+1,
                yolo_counts[idx], round(yolo_confs[idx],3),
                vehicles_cleared, round(avg_speed,3),
                active_phases[idx], trans_state[idx],
                action, round(reward,3),
                mode, emergency_flag,
                emergency_state[idx],
                fb_status.get('mode', 'DQN'),
                grace_counter[idx],
                preclear_counter[idx],
                recovery_counter[idx],
                emergency_timeout[idx],
                arm_stats['avg_waiting_time'],
                arm_stats['max_waiting_time'],
                arm_stats['queue_length'],
                arm_stats['throughput_vpm'],
                round(agents[idx].epsilon,4),
                trackers[idx].episode
            ])

        # Status every 500 ticks
        if tick_count % 500 == 0:
            print(f"\nTick {tick_count} | t={timestamp:.0f}s")
            for idx in range(NUM_INT):
                ws = wait_trackers[idx].get_stats()
                aq = ws['arm_queues']
                print(f"  Int {idx+1}: "
                      f"YOLO={yolo_counts[idx]:2d} "
                      f"conf={yolo_confs[idx]:.2f} "
                      f"Mode={modes[idx]:10s} "
                      f"Wait={ws['avg_waiting_time']:.2f}s "
                      f"Q(N{aq['N']} S{aq['S']} E{aq['E']} W{aq['W']}) "
                      f"Phase={PHASE_NAMES[active_phases[idx]]}"
                      f"[{trans_state[idx][0]}]")

except KeyboardInterrupt:
    print("\nStopping...")
    if args.train:
        for i, agent in enumerate(agents):
            agent.save(f"data/dqn_weights_int{i+1}.json")
        print("Weights saved.")

finally:
    csv_file.close()
    for cam in cameras:
        cam.stop(); cam.destroy()
    for v in vehicles + rush_vehicles:
        if v.is_alive: v.destroy()
    if emergency_vehicle and emergency_vehicle.is_alive:
        emergency_vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Done.")
