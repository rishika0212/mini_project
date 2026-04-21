"""
evaluate.py — DQN-only evaluation with intelligent phase switching.

Usage:
    python evaluate.py --policy dqn

Results saved to: data/eval_dqn.csv
"""

import carla
import random
import os
import csv
import math
import time
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
import threading

from dqn_agent   import DQNAgent, build_state
from waiting_time import IntersectionWaitingTimeTracker

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--policy',   default='dqn',
                    choices=['dqn', 'fixed', 'random'],
                    help='Policy to evaluate: dqn | fixed | random')
parser.add_argument('--episodes', default=30, type=int,
                    help='Number of evaluation episodes')
parser.add_argument('--ep_len',   default=500, type=int,
                    help='Ticks per episode')
args = parser.parse_args()

print(f"\n{'='*50}")
print(f"  Evaluating policy: {args.policy.upper()}")
print(f"  Episodes: {args.episodes}  |  Length: {args.ep_len} ticks")
print(f"{'='*50}\n")

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

print("Connected to CARLA.")

# ── YOLOv8 ─────────────────────────────────────────────────────────────────
print("Loading YOLOv8...")
model           = YOLO('yolov8n.pt')
VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
EMERGENCY_KW    = ['ambulance','firetruck','police']
print("YOLOv8 ready.")

# ── DQN agents (only used if policy=dqn) ───────────────────────────────────
NUM_INT = 3
agents  = []
if args.policy == 'dqn':
    print("Loading DQN weights...")
    for i in range(NUM_INT):
        agent         = DQNAgent()
        agent.epsilon = 0.0   # NO exploration during evaluation
        agent.load(f"data/dqn_weights_int{i+1}.json")
        agent.epsilon = 0.0   # force greedy after load
        agents.append(agent)
    print("DQN agents loaded (epsilon=0, greedy mode).")

# ── Shared frame storage ───────────────────────────────────────────────────
latest_frames = {}
frame_locks   = {}
frame_counts  = {}

# ── Intersections ──────────────────────────────────────────────────────────
def group_lights(lights, threshold=45):
    groups = []
    for light in lights:
        loc   = light.get_location()
        added = False
        for group in groups:
            if loc.distance(group[0].get_location()) < threshold:
                group.append(light)
                added = True
                break
        if not added:
            groups.append([light])
    return groups

traffic_lights = world.get_actors().filter("traffic.traffic_light")
groups         = group_lights(traffic_lights, threshold=45)
valid_groups   = [g for g in groups if len(g) >= 3]
intersections  = valid_groups[:NUM_INT]

intersection_centers = []
for i, group in enumerate(intersections):
    x = sum(l.get_location().x for l in group)/len(group)
    y = sum(l.get_location().y for l in group)/len(group)
    z = sum(l.get_location().z for l in group)/len(group)
    intersection_centers.append(carla.Location(x=x,y=y,z=z))
    print(f"  Int {i+1}: ({x:.1f}, {y:.1f})")

ROI_RADIUS = 50

# ── Waiting time trackers ──────────────────────────────────────────────────
wait_trackers = [IntersectionWaitingTimeTracker(i+1, ROI_RADIUS, intersection_centers[i])
                 for i in range(NUM_INT)]

# ── Spawn vehicles ─────────────────────────────────────────────────────────
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                and not any(kw in bp.id.lower() for kw in EMERGENCY_KW)]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

for sp in spawn_points[:120]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)
        vehicles.append(v)

for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center)<80]
    n = 0
    for sp in nearby:
        if n >= 35: break
        bp = random.choice(car_bps)
        v  = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True)
            vehicles.append(v)
            n += 1

print(f"Spawned {len(vehicles)} vehicles.")
for _ in range(20):
    world.tick()

# ── Cameras ────────────────────────────────────────────────────────────────
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x','640')
camera_bp.set_attribute('image_size_y','640')
camera_bp.set_attribute('fov','90')
camera_bp.set_attribute('sensor_tick','0.1')

cameras        = []
camera_int_map = []

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
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb   = array[:,:,:3][:,:,::-1].copy()
    with frame_locks[idx]:
        latest_frames[idx]  = rgb
        frame_counts[idx]  += 1

for idx, cam in enumerate(cameras):
    cam.listen(lambda img, i=idx: on_image(img, i))

print("Cameras attached. Warming up...")
for _ in range(30):
    world.tick()
print(f"  Frames: {[frame_counts[i] for i in range(len(cameras))]}")

# ── YOLO ───────────────────────────────────────────────────────────────────
def run_yolo(camera_idx):
    with frame_locks[camera_idx]:
        frame = latest_frames[camera_idx]
    if frame is None:
        return 0
    results = model(frame, verbose=False, conf=0.25)
    count   = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in VEHICLE_CLASSES:
                count += 1
    return count

# ── Ground truth ───────────────────────────────────────────────────────────
def compute_gt(center):
    actors    = world.get_actors().filter("vehicle.*")
    count     = 0
    speed_sum = 0.0
    for v in actors:
        if v.get_location().distance(center) < ROI_RADIUS:
            count += 1
            vel    = v.get_velocity()
            speed_sum += math.sqrt(vel.x**2+vel.y**2+vel.z**2)
    return count, (speed_sum/count if count>0 else 0.0)

def get_arm_counts(actors, center):
    """Count vehicles per approach arm (N/S/E/W). +X=East, +Y=South in CARLA."""
    n = s = e = w = 0
    for v in actors:
        loc = v.get_location()
        if loc.distance(center) < ROI_RADIUS:
            dx = loc.x - center.x
            dy = loc.y - center.y
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

def find_emergency_in_roi(center):
    for v in world.get_actors().filter("vehicle.*"):
        if not v.is_alive:
            continue
        if v.get_location().distance(center) < ROI_RADIUS:
            if is_emergency_vehicle(v):
                return v
    return None

def set_emergency_phase(group, emg_vehicle):
    emg_loc = emg_vehicle.get_location()
    closest = min(group, key=lambda l: l.get_location().distance(emg_loc))
    for light in group:
        if light == closest:
            light.set_state(carla.TrafficLightState.Green)
        else:
            light.set_state(carla.TrafficLightState.Red)

# ── Signal control ─────────────────────────────────────────────────────────
PHASE_NS_GREEN = 0
PHASE_YELLOW   = 1
PHASE_EW_GREEN = 2

YELLOW_TICKS = 20  # ~1s at 0.05s tick

phase_states      = [PHASE_NS_GREEN] * NUM_INT
phase_counters    = [0] * NUM_INT
yellow_remaining  = [0] * NUM_INT
yellow_targets    = [PHASE_EW_GREEN] * NUM_INT
intersection_arms = []

def _angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)

def split_intersection_arms(group):
    """Split each intersection's lights into two opposing movement arms."""
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

for g in intersections:
    intersection_arms.append(split_intersection_arms(g))

def apply_phase(idx, phase):
    arm_ns, arm_ew = intersection_arms[idx]
    if phase == PHASE_NS_GREEN:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Green)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Red)
    elif phase == PHASE_YELLOW:
        if yellow_targets[idx] == PHASE_EW_GREEN:
            for l in arm_ns: l.set_state(carla.TrafficLightState.Yellow)
            for l in arm_ew: l.set_state(carla.TrafficLightState.Red)
        else:
            for l in arm_ns: l.set_state(carla.TrafficLightState.Red)
            for l in arm_ew: l.set_state(carla.TrafficLightState.Yellow)
    else:
        for l in arm_ns: l.set_state(carla.TrafficLightState.Red)
        for l in arm_ew: l.set_state(carla.TrafficLightState.Green)

for i in range(NUM_INT):
    apply_phase(i, PHASE_NS_GREEN)

# ── CSV output ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path   = f"data/eval_{args.policy}.csv"
csv_file   = open(out_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "episode","intersection_id","tick",
    "yolo_count","gt_count","avg_speed",
    "signal_phase","action",
    "avg_waiting_time","max_waiting_time",
    "queue_length","throughput_vpm",
    "emergency_flag"
])

# ── Emergency vehicle ──────────────────────────────────────────────────────
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
        traffic_manager.vehicle_percentage_speed_difference(v, -80)
        traffic_manager.distance_to_leading_vehicle(v, 0.5)
    return v

# ── Main evaluation loop ───────────────────────────────────────────────────
print(f"\nStarting evaluation — {args.episodes} episodes × {args.ep_len} ticks\n")

YOLO_INTERVAL      = 10
EMERGENCY_INTERVAL = 120
EMERGENCY_LIFETIME = 260
FIXED_PHASE_LEN    = 100   # ticks per green phase for fixed policy

yolo_counts     = [0] * NUM_INT

# Episode-level result storage
episode_results = []

try:
    for episode in range(1, args.episodes + 1):
        print(f"Episode {episode}/{args.episodes} ...", end=" ", flush=True)

        # Reset trackers
        for wt in wait_trackers:
            wt.reset_episode()

        ep_tick             = 0
        emergency_counter   = 0
        emergency_age       = 0
        emergency_vehicle   = None

        while ep_tick < args.ep_len:
            world.tick()
            ep_tick        += 1
            emergency_counter += 1

            # Emergency lifecycle
            if emergency_vehicle and emergency_vehicle.is_alive:
                emergency_age += 1
                if emergency_age >= EMERGENCY_LIFETIME:
                    emergency_vehicle.destroy()
                    emergency_vehicle = None
                    emergency_age     = 0

            if emergency_counter >= EMERGENCY_INTERVAL:
                if emergency_vehicle is None:
                    emergency_vehicle = spawn_emergency()
                    emergency_age     = 0
                emergency_counter = 0

            # YOLO
            if ep_tick % YOLO_INTERVAL == 0:
                for cam_i in range(len(cameras)):
                    yolo_counts[camera_int_map[cam_i]] = run_yolo(cam_i)

            # Update waiting time trackers
            all_actors = list(world.get_actors().filter("vehicle.*"))
            for idx, center in enumerate(intersection_centers):
                wait_trackers[idx].update(all_actors, center, ep_tick)

            # Per-intersection control
            for idx, center in enumerate(intersection_centers):
                gt_count, avg_speed = compute_gt(center)
                arm_n, arm_s, arm_e, arm_w = get_arm_counts(all_actors, center)
                emg_v          = find_emergency_in_roi(center)
                emergency_flag = 1 if emg_v is not None else 0

                wait_stats_now = wait_trackers[idx].get_stats()

                # Compute pressures: pressure = queue + 1.5 * avg_wait
                pressures = {
                    'N': wait_stats_now['arm_queues']['N'] + 1.5 * wait_stats_now['arm_avg_waits']['N'],
                    'S': wait_stats_now['arm_queues']['S'] + 1.5 * wait_stats_now['arm_avg_waits']['S'],
                    'E': wait_stats_now['arm_queues']['E'] + 1.5 * wait_stats_now['arm_avg_waits']['E'],
                    'W': wait_stats_now['arm_queues']['W'] + 1.5 * wait_stats_now['arm_avg_waits']['W'],
                }

                # Map phase to arm: PHASE_NS_GREEN=0 -> 'N' or 'S', PHASE_EW_GREEN=2 -> 'E' or 'W'
                if phase_states[idx] == PHASE_NS_GREEN:
                    current_arm = 'N'  # Default; actual arm doesn't matter for pressure calc
                elif phase_states[idx] == PHASE_EW_GREEN:
                    current_arm = 'E'  # Default; actual arm doesn't matter for pressure calc
                else:
                    current_arm = 'N'  # Yellow phase, use 'N' as default

                state = build_state(
                    pressures       = pressures,
                    current_arm     = current_arm,
                    ticks_in_phase  = phase_counters[idx],
                    emergency_flag  = emergency_flag,
                )

                if emergency_flag == 1 and emg_v.is_alive:
                    set_emergency_phase(intersections[idx], emg_v)
                    phase_states[idx] = PHASE_NS_GREEN
                    phase_counters[idx] = 0
                    yellow_remaining[idx] = 0
                    yellow_targets[idx] = PHASE_EW_GREEN
                    action = 0
                else:
                    if args.policy == 'dqn':
                        action = agents[idx].act(state)
                    elif args.policy == 'fixed':
                        action = 1 if phase_counters[idx] >= FIXED_PHASE_LEN else 0
                    else:  # random
                        action = random.randint(0, 1)

                    if phase_states[idx] == PHASE_YELLOW:
                        yellow_remaining[idx] = max(0, yellow_remaining[idx] - 1)
                        if yellow_remaining[idx] == 0:
                            phase_states[idx] = yellow_targets[idx]
                            phase_counters[idx] = 0
                            apply_phase(idx, phase_states[idx])
                    else:
                        if action == 1:
                            prev_phase = phase_states[idx]
                            phase_states[idx] = PHASE_YELLOW
                            phase_counters[idx] = 0
                            yellow_remaining[idx] = YELLOW_TICKS
                            yellow_targets[idx] = (PHASE_EW_GREEN
                                                   if prev_phase == PHASE_NS_GREEN
                                                   else PHASE_NS_GREEN)
                            apply_phase(idx, PHASE_YELLOW)

                phase_counters[idx] += 1

                # Log every 10 ticks
                if ep_tick % 10 == 0:
                    stats = wait_trackers[idx].get_stats()
                    csv_writer.writerow([
                        episode, idx+1, ep_tick,
                        yolo_counts[idx], gt_count,
                        round(avg_speed,3), phase_states[idx], action,
                        stats['avg_waiting_time'],
                        stats['max_waiting_time'],
                        stats['queue_length'],
                        stats['throughput_vpm'],
                        emergency_flag
                    ])

        # Episode summary
        ep_stats = {}
        for idx in range(NUM_INT):
            s = wait_trackers[idx].get_stats()
            ep_stats[idx+1] = s

        avg_wait_all = np.mean([ep_stats[i+1]['avg_waiting_time']
                                for i in range(NUM_INT)])
        print(f"Avg wait: {avg_wait_all:.2f}s")
        episode_results.append({
            'episode': episode,
            'avg_wait': avg_wait_all,
            'stats': ep_stats
        })

finally:
    # Print final summary
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE — Policy: {args.policy.upper()}")
    print(f"{'='*50}")

    if episode_results:
        all_waits = [r['avg_wait'] for r in episode_results]
        print(f"Episodes completed : {len(episode_results)}")
        print(f"Avg waiting time   : {np.mean(all_waits):.3f}s")
        print(f"Best episode wait  : {min(all_waits):.3f}s")
        print(f"Worst episode wait : {max(all_waits):.3f}s")
        print(f"Results saved to   : {out_path}")

    csv_file.close()

    for cam in cameras:
        cam.stop()
        cam.destroy()
    for v in vehicles:
        if v.is_alive:
            v.destroy()
    if emergency_vehicle and emergency_vehicle.is_alive:
        emergency_vehicle.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Done.")
