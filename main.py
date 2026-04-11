"""
main.py
=======
Complete integrated system:
  CARLA → RGB Camera → YOLOv8 → Confidence Check
    → DQN Agent (normal) OR Fixed-Time (fallback)
    → Signal Control
    → Waiting Time Tracking
    → CSV Logging

This is the final version incorporating all phases:
  ✅ Phase 1: Simulation + data collection
  ✅ Phase 2: DQN agent
  ✅ Phase 3: YOLOv8 detection
  ✅ Phase 4: Full pipeline integration
  ✅ Phase 5: Emergency priority + safety fallback + time-of-day
  ✅ Phase 6: Waiting time metrics

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
from pathlib import Path

from dqn_agent    import DQNAgent, EpisodeTracker, build_state, \
                         compute_reward, dqn_phase_duration
from waiting_time import IntersectionWaitingTimeTracker
from fallback     import FallbackController, ControlMode, get_yolo_confidence

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolov8n.pt',
                    help='YOLO model path (default: yolov8n.pt)')
parser.add_argument('--train', action='store_true',
                    help='Train DQN (default: evaluation mode)')
args = parser.parse_args()

print(f"\n{'='*55}")
print(f"  Traffic Intelligence System — Final Version")
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
print("Connected to CARLA.")

# ── YOLOv8 ─────────────────────────────────────────────────────────────────
print(f"Loading YOLO model: {args.model}")
model           = YOLO(args.model)
VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
EMERGENCY_KW    = ['ambulance','firetruck','police']
print("YOLO ready.")

# ── DQN agents ─────────────────────────────────────────────────────────────
NUM_INT = 3
agents, trackers, fallbacks = [], [], []

for i in range(NUM_INT):
    agent   = DQNAgent()
    tracker = EpisodeTracker(episode_length=500)
    tracker.set_int_id(i+1)
    fb      = FallbackController(intersection_id=i+1)

    if not args.train:
        agent.epsilon = 0.0   # greedy during evaluation

    agent.load(f"data/dqn_weights_int{i+1}.json")

    if not args.train:
        agent.epsilon = 0.0

    agents.append(agent)
    trackers.append(tracker)
    fallbacks.append(fb)

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
wait_trackers = [IntersectionWaitingTimeTracker(i+1, ROI_RADIUS)
                 for i in range(NUM_INT)]

# ── Vehicles ───────────────────────────────────────────────────────────────
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

for sp in spawn_points[:30]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v: v.set_autopilot(True); vehicles.append(v)

for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center)<80]
    n = 0
    for sp in nearby:
        if n >= 15: break
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

# ── Ground truth ───────────────────────────────────────────────────────────
def compute_gt(center):
    actors = world.get_actors().filter("vehicle.*")
    c, ss  = 0, 0.0
    for v in actors:
        if v.get_location().distance(center) < ROI_RADIUS:
            c += 1
            vel = v.get_velocity()
            ss += math.sqrt(vel.x**2+vel.y**2+vel.z**2)
    return c, (ss/c if c>0 else 0.0)

# ── Emergency ──────────────────────────────────────────────────────────────
def check_emergency(center):
    for v in world.get_actors().filter("vehicle.*"):
        if v.get_location().distance(center) < ROI_RADIUS:
            if any(kw in v.type_id.lower() for kw in EMERGENCY_KW):
                return 1
    return 0

emergency_vehicle = None

def spawn_emergency():
    lib = world.get_blueprint_library()
    bps = list(lib.filter("vehicle.*ambulance*")) + \
          list(lib.filter("vehicle.*firetruck*"))
    if not bps: return None
    center = random.choice(intersection_centers)
    nearby = [sp for sp in world.get_map().get_spawn_points()
              if sp.location.distance(center)<100]
    sp = random.choice(nearby) if nearby else \
         random.choice(world.get_map().get_spawn_points())
    v  = world.try_spawn_actor(random.choice(bps), sp)
    if v: v.set_autopilot(True)
    return v

# ── Signal ─────────────────────────────────────────────────────────────────
phase_states    = [carla.TrafficLightState.Green,
                   carla.TrafficLightState.Yellow,
                   carla.TrafficLightState.Red]
phases          = [0]*NUM_INT
counters        = [0]*NUM_INT
phase_durations = [100]*NUM_INT

def set_phase(group, state):
    for light in group: light.set_state(state)

# ── CSV ────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
csv_file   = open("data/rl_states_final.csv","w",newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp","intersection_id",
    "yolo_count","yolo_confidence","gt_count","avg_speed",
    "signal_phase","phase_duration","action","reward",
    "control_mode","emergency_flag",
    "avg_waiting_time","max_waiting_time","queue_length",
    "throughput_vpm","epsilon","episode"
])

# ── Main loop ──────────────────────────────────────────────────────────────
print("\nSystem running. Press Ctrl+C to stop.\n")

YOLO_INTERVAL      = 10
EMERGENCY_INTERVAL = 400
EMERGENCY_LIFETIME = 300

tick_count   = 0
emg_counter  = 0
emg_age      = 0
prev_states  = [None]*NUM_INT
prev_actions = [0]*NUM_INT

try:
    while True:
        world.tick()
        tick_count += 1

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

        # Update waiting time trackers
        for idx, center in enumerate(intersection_centers):
            wait_trackers[idx].update(all_actors, center, tick_count)

        # YOLO every 10 ticks
        if tick_count % YOLO_INTERVAL == 0:
            for cam_i in range(len(cameras)):
                cnt, conf, _ = run_yolo(cam_i, tick_count)
                int_i        = camera_int_map[cam_i]
                yolo_counts[int_i] = cnt
                yolo_confs[int_i]  = conf

        # Per-intersection control
        for idx, center in enumerate(intersection_centers):
            gt_count, avg_speed = compute_gt(center)
            emergency_flag      = check_emergency(center)
            wait_stats          = wait_trackers[idx].get_stats()

            # Fallback check
            mode = fallbacks[idx].update(
                yolo_confidence = yolo_confs[idx],
                yolo_count      = yolo_counts[idx],
                emergency_flag  = emergency_flag
            )

            # Build state
            state = build_state(
                yolo_count=yolo_counts[idx], gt_count=gt_count,
                avg_speed=avg_speed, current_phase=phases[idx],
                phase_counter=counters[idx],
                phase_duration=phase_durations[idx],
                emergency_flag=emergency_flag,
                elapsed_seconds=timestamp
            )

            # Select action based on mode
            if mode == ControlMode.EMERGENCY:
                set_phase(intersections[idx], carla.TrafficLightState.Green)
                phases[idx]=0; counters[idx]=0
                action=0; reward=2.0

            elif mode == ControlMode.FIXED_TIME:
                should_switch, new_phase = \
                    fallbacks[idx].get_fixed_time_action()
                if should_switch:
                    phases[idx] = new_phase
                    set_phase(intersections[idx], phase_states[phases[idx]])
                action=0
                reward = compute_reward(yolo_counts[idx], avg_speed,
                                        phases[idx], emergency_flag, 0)

            else:  # DQN
                action = agents[idx].act(state)
                reward = compute_reward(yolo_counts[idx], avg_speed,
                                        phases[idx], emergency_flag, action)

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

            # Phase switching (DQN mode only)
            if mode == ControlMode.DQN:
                counters[idx] += 1
                if counters[idx] >= phase_durations[idx]:
                    counters[idx]        = 0
                    phases[idx]          = (phases[idx]+1) % len(phase_states)
                    phase_durations[idx] = dqn_phase_duration(action,
                                                              yolo_counts[idx])
                    set_phase(intersections[idx], phase_states[phases[idx]])

            # Log
            csv_writer.writerow([
                round(timestamp,3), idx+1,
                yolo_counts[idx], round(yolo_confs[idx],3),
                gt_count, round(avg_speed,3),
                phases[idx], phase_durations[idx],
                action, round(reward,3),
                mode.value, emergency_flag,
                wait_stats['avg_waiting_time'],
                wait_stats['max_waiting_time'],
                wait_stats['queue_length'],
                wait_stats['throughput_vpm'],
                round(agents[idx].epsilon,4),
                trackers[idx].episode
            ])

        # Status
        if tick_count % 500 == 0:
            print(f"\nTick {tick_count} | t={timestamp:.0f}s")
            for idx in range(NUM_INT):
                ws = wait_trackers[idx].get_stats()
                fb = fallbacks[idx].get_status()
                print(f"  Int {idx+1}: "
                      f"YOLO={yolo_counts[idx]:2d} "
                      f"conf={yolo_confs[idx]:.2f} "
                      f"Mode={fb['mode']:10s} "
                      f"Wait={ws['avg_waiting_time']:.2f}s "
                      f"Queue={ws['queue_length']:2d} "
                      f"Phase={['G','Y','R'][phases[idx]]}")

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
    for v in vehicles:
        if v.is_alive: v.destroy()
    if emergency_vehicle and emergency_vehicle.is_alive:
        emergency_vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Done.")