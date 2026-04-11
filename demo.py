"""
demo.py — Live visual demo of trained DQN agent
Shows YOLO bounding boxes + signal state + waiting time overlay
on screen in real time. Record your screen while this runs.

Usage:
    python demo.py
"""

import carla
import random
import os
import math
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import time

from dqn_agent    import DQNAgent, build_state, dqn_phase_duration
from waiting_time import IntersectionWaitingTimeTracker

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
model           = YOLO('yolov8n.pt')
VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
EMERGENCY_KW    = ['ambulance','firetruck','police']

# ── DQN agents ─────────────────────────────────────────────────────────────
NUM_INT = 3
agents  = []
for i in range(NUM_INT):
    agent         = DQNAgent()
    agent.epsilon = 0.0
    agent.load(f"data/dqn_weights_int{i+1}.json")
    agent.epsilon = 0.0
    agents.append(agent)

print("DQN agents loaded (greedy mode).")

# ── Intersections ──────────────────────────────────────────────────────────
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

traffic_lights       = world.get_actors().filter("traffic.traffic_light")
groups               = group_lights(traffic_lights, threshold=45)
valid_groups         = [g for g in groups if len(g) >= 3]
intersections        = valid_groups[:NUM_INT]
intersection_centers = []

for i, group in enumerate(intersections):
    x = sum(l.get_location().x for l in group)/len(group)
    y = sum(l.get_location().y for l in group)/len(group)
    z = sum(l.get_location().z for l in group)/len(group)
    intersection_centers.append(carla.Location(x=x,y=y,z=z))

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
camera_bp.set_attribute('sensor_tick','0.05')   # faster for demo

latest_frames = {}
frame_locks   = {}
frame_counts  = {}
cameras       = []
camera_int_map= []

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
        latest_frames[idx] = rgb; frame_counts[idx] += 1

for idx, cam in enumerate(cameras):
    cam.listen(lambda img, i=idx: on_image(img, i))

print("Cameras ready. Warming up...")
for _ in range(30): world.tick()

# ── YOLO with annotations ──────────────────────────────────────────────────
PHASE_COLORS = {
    0: (0, 255, 0),    # Green
    1: (0, 215, 255),  # Yellow
    2: (0, 0, 255),    # Red
}
PHASE_NAMES = {0:'GREEN', 1:'YELLOW', 2:'RED'}

def run_yolo_annotated(camera_idx, phase, wait_stats, yolo_count):
    with frame_locks[camera_idx]:
        frame = latest_frames[camera_idx]
    if frame is None:
        return None, 0

    results    = model(frame, verbose=False, conf=0.25)
    annotated  = frame[:,:,::-1].copy()
    count      = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES:
                count += 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(annotated, label, (x1,max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),1)

    # ── HUD overlay ────────────────────────────────────────────────────────
    int_id     = camera_int_map[camera_idx] + 1
    ph_color   = PHASE_COLORS[phase]
    ph_name    = PHASE_NAMES[phase]
    avg_wait   = wait_stats['avg_waiting_time']
    queue      = wait_stats['queue_length']

    # Dark banner at top
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0,0),(640,90),(0,0,0),-1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Intersection title
    cv2.putText(annotated, f"INTERSECTION {int_id}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    # Signal phase indicator
    cv2.circle(annotated, (580,30), 20, ph_color, -1)
    cv2.putText(annotated, ph_name,
                (490,75), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                ph_color, 2)

    # Stats
    cv2.putText(annotated, f"YOLO: {count} vehicles",
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200,200,200), 1)
    cv2.putText(annotated, f"Avg Wait: {avg_wait:.1f}s  Queue: {queue}",
                (10,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200,200,200), 1)

    return annotated, count

# ── Ground truth + emergency ───────────────────────────────────────────────
def compute_gt(center):
    actors = world.get_actors().filter("vehicle.*")
    c, ss  = 0, 0.0
    for v in actors:
        if v.get_location().distance(center) < ROI_RADIUS:
            c += 1
            vel = v.get_velocity()
            ss += math.sqrt(vel.x**2+vel.y**2+vel.z**2)
    return c, (ss/c if c>0 else 0.0)

def check_emergency(center):
    for v in world.get_actors().filter("vehicle.*"):
        if v.get_location().distance(center) < ROI_RADIUS:
            if any(kw in v.type_id.lower() for kw in EMERGENCY_KW):
                return 1
    return 0

# ── Signal ─────────────────────────────────────────────────────────────────
phase_states    = [carla.TrafficLightState.Green,
                   carla.TrafficLightState.Yellow,
                   carla.TrafficLightState.Red]
phases          = [0]*NUM_INT
counters        = [0]*NUM_INT
phase_durations = [100]*NUM_INT
yolo_counts     = [0]*NUM_INT

def set_phase(group, state):
    for light in group: light.set_state(state)

# ── Emergency spawner ──────────────────────────────────────────────────────
emergency_vehicle = None

def spawn_emergency():
    lib = world.get_blueprint_library()
    bps = list(lib.filter("vehicle.*ambulance*"))+\
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

# ── Display window ─────────────────────────────────────────────────────────
WIN_NAME = "Traffic Intelligence Demo — Press Q to quit"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1920, 680)

# ── Main loop ──────────────────────────────────────────────────────────────
print(f"\nDemo running. Press Q in the window to quit.\n")

YOLO_INTERVAL      = 5    # faster for demo
EMERGENCY_INTERVAL = 400
EMERGENCY_LIFETIME = 300

tick_count        = 0
emg_counter       = 0
emg_age           = 0
annotated_frames  = [None]*NUM_INT

try:
    while True:
        world.tick()
        tick_count += 1

        # Emergency
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

        # YOLO + annotate every YOLO_INTERVAL ticks
        if tick_count % YOLO_INTERVAL == 0:
            for cam_i in range(len(cameras)):
                int_i      = camera_int_map[cam_i]
                stats      = wait_trackers[int_i].get_stats()
                frame, cnt = run_yolo_annotated(cam_i, phases[int_i],
                                                stats, yolo_counts[int_i])
                if frame is not None:
                    annotated_frames[int_i] = frame
                    yolo_counts[int_i]      = cnt

        # DQN control
        for idx, center in enumerate(intersection_centers):
            gt_count, avg_speed = compute_gt(center)
            emergency_flag      = check_emergency(center)

            state = build_state(
                yolo_count=yolo_counts[idx], gt_count=gt_count,
                avg_speed=avg_speed, current_phase=phases[idx],
                phase_counter=counters[idx],
                phase_duration=phase_durations[idx],
                emergency_flag=emergency_flag,
                elapsed_seconds=timestamp
            )

            if emergency_flag == 1:
                set_phase(intersections[idx], carla.TrafficLightState.Green)
                phases[idx]=0; counters[idx]=0; action=0
            else:
                action = agents[idx].act(state)

            counters[idx] += 1
            if counters[idx] >= phase_durations[idx]:
                counters[idx]        = 0
                phases[idx]          = (phases[idx]+1) % len(phase_states)
                phase_durations[idx] = dqn_phase_duration(action, yolo_counts[idx])
                set_phase(intersections[idx], phase_states[phases[idx]])

        # Build display — 3 camera feeds side by side
        valid = [f for f in annotated_frames if f is not None]
        if len(valid) == NUM_INT:
            display = np.hstack(valid)
            cv2.imshow(WIN_NAME, display)

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print stats every 200 ticks
        if tick_count % 200 == 0:
            print(f"Tick {tick_count} | t={timestamp:.0f}s")
            for idx in range(NUM_INT):
                s = wait_trackers[idx].get_stats()
                print(f"  Int {idx+1}: "
                      f"YOLO={yolo_counts[idx]:2d} "
                      f"AvgWait={s['avg_waiting_time']:.2f}s "
                      f"Queue={s['queue_length']} "
                      f"Phase={['GREEN','YELLOW','RED'][phases[idx]]}")

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    for cam in cameras:
        cam.stop(); cam.destroy()
    for v in vehicles:
        if v.is_alive: v.destroy()
    if emergency_vehicle and emergency_vehicle.is_alive:
        emergency_vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Demo ended.")
