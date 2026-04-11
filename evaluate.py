"""
evaluate.py — Clean evaluation of 3 policies
Runs each policy for 30 episodes and records real waiting time metrics.

Usage:
    python evaluate.py --policy dqn      # trained DQN agent
    python evaluate.py --policy fixed    # fixed 30s per phase
    python evaluate.py --policy random   # random phase selection

Results saved to: data/eval_{policy}.csv
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

from dqn_agent   import DQNAgent, build_state, dqn_phase_duration
from waiting_time import IntersectionWaitingTimeTracker

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--policy',   default='dqn',
                    choices=['dqn','fixed','random'],
                    help='Policy to evaluate')
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
wait_trackers = [IntersectionWaitingTimeTracker(i+1, ROI_RADIUS)
                 for i in range(NUM_INT)]

# ── Spawn vehicles ─────────────────────────────────────────────────────────
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

for sp in spawn_points[:30]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)
        vehicles.append(v)

for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center)<80]
    n = 0
    for sp in nearby:
        if n >= 15: break
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

def check_emergency(center):
    for v in world.get_actors().filter("vehicle.*"):
        if v.get_location().distance(center) < ROI_RADIUS:
            if any(kw in v.type_id.lower() for kw in EMERGENCY_KW):
                return 1
    return 0

# ── Signal control ─────────────────────────────────────────────────────────
phase_states = [carla.TrafficLightState.Green,
                carla.TrafficLightState.Yellow,
                carla.TrafficLightState.Red]

FIXED_DURATION = int(30 / 0.05)   # 30 seconds in ticks = 600

def set_phase(group, state):
    for light in group:
        light.set_state(state)

def get_action(policy, idx, state, yolo_count):
    if policy == 'dqn':
        return agents[idx].act(state)
    elif policy == 'fixed':
        return 0   # always keep current phase (duration handles switching)
    elif policy == 'random':
        return random.randint(0, 1)

def get_duration(policy, action, yolo_count):
    if policy == 'fixed':
        return FIXED_DURATION
    elif policy == 'random':
        return random.choice([40, 80, 120])
    else:
        return dqn_phase_duration(action, yolo_count)

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
    bps = list(lib.filter("vehicle.*ambulance*")) + \
          list(lib.filter("vehicle.*firetruck*"))
    if not bps: return None
    center = random.choice(intersection_centers)
    nearby = [sp for sp in world.get_map().get_spawn_points()
              if sp.location.distance(center) < 100]
    sp = random.choice(nearby) if nearby else \
         random.choice(world.get_map().get_spawn_points())
    v  = world.try_spawn_actor(random.choice(bps), sp)
    if v:
        v.set_autopilot(True)
    return v

# ── Main evaluation loop ───────────────────────────────────────────────────
print(f"\nStarting evaluation — {args.episodes} episodes × {args.ep_len} ticks\n")

YOLO_INTERVAL      = 10
EMERGENCY_INTERVAL = 400
EMERGENCY_LIFETIME = 300

phases          = [0] * NUM_INT
counters        = [0] * NUM_INT
phase_durations = [FIXED_DURATION if args.policy=='fixed' else 100] * NUM_INT
yolo_counts     = [0] * NUM_INT
prev_states     = [None] * NUM_INT

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
                emergency_flag      = check_emergency(center)

                state = build_state(
                    yolo_count      = yolo_counts[idx],
                    gt_count        = gt_count,
                    avg_speed       = avg_speed,
                    current_phase   = phases[idx],
                    phase_counter   = counters[idx],
                    phase_duration  = phase_durations[idx],
                    emergency_flag  = emergency_flag,
                    elapsed_seconds = ep_tick * 0.05
                )

                if emergency_flag == 1:
                    set_phase(intersections[idx], carla.TrafficLightState.Green)
                    phases[idx]   = 0
                    counters[idx] = 0
                    action        = 0
                else:
                    action = get_action(args.policy, idx, state, yolo_counts[idx])

                counters[idx] += 1
                if counters[idx] >= phase_durations[idx]:
                    counters[idx]        = 0
                    phases[idx]          = (phases[idx]+1) % len(phase_states)
                    phase_durations[idx] = get_duration(args.policy,
                                                        action,
                                                        yolo_counts[idx])
                    set_phase(intersections[idx], phase_states[phases[idx]])

                # Log every 10 ticks
                if ep_tick % 10 == 0:
                    stats = wait_trackers[idx].get_stats()
                    csv_writer.writerow([
                        episode, idx+1, ep_tick,
                        yolo_counts[idx], gt_count,
                        round(avg_speed,3), phases[idx], action,
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
