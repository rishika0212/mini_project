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
import winsound

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
NUM_INT = 2
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
EMERGENCY_BP_KW = ['ambulance', 'firetruck', 'police']
car_bps         = [bp for bp in blueprints
                   if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                   and not any(kw in bp.id.lower() for kw in EMERGENCY_BP_KW)]
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
cameras          = []
camera_int_map   = []
camera_transforms= []

for i, center in enumerate(intersection_centers):
    transform = carla.Transform(
        carla.Location(x=center.x-20, y=center.y, z=center.z+12),
        carla.Rotation(pitch=-45, yaw=0)
    )
    cam = world.spawn_actor(camera_bp, transform)
    cameras.append(cam)
    camera_int_map.append(i)
    camera_transforms.append(transform)
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

# ── Camera projection helpers ──────────────────────────────────────────────
_CAM_W, _CAM_H, _CAM_FOV = 640, 640, 90
_focal = _CAM_W / (2.0 * math.tan(math.radians(_CAM_FOV / 2.0)))
CAM_K  = np.array([[_focal, 0,      _CAM_W / 2.0],
                   [0,      _focal, _CAM_H / 2.0],
                   [0,      0,      1.0          ]], dtype=np.float64)

def _world_to_cam(transform):
    """Return 4x4 world-to-camera matrix for a carla.Transform."""
    cy = math.cos(math.radians(transform.rotation.yaw))
    sy = math.sin(math.radians(transform.rotation.yaw))
    cr = math.cos(math.radians(transform.rotation.roll))
    sr = math.sin(math.radians(transform.rotation.roll))
    cp = math.cos(math.radians(transform.rotation.pitch))
    sp = math.sin(math.radians(transform.rotation.pitch))
    tx, ty, tz = transform.location.x, transform.location.y, transform.location.z
    # Camera-to-world (columns = local axes in world space)
    c2w = np.array([
        [cp*cy,  cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr, tx],
        [cp*sy,  sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr, ty],
        [sp,    -cp*sr,             cp*cr,             tz],
        [0,      0,                 0,                 1 ]
    ], dtype=np.float64)
    return np.linalg.inv(c2w)

def project_actor(actor, cam_idx):
    """
    Project an actor's centre to pixel coords for camera cam_idx.
    Returns (u, v, depth) or None if behind the camera or off-screen.
    """
    loc  = actor.get_location()
    w2c  = _world_to_cam(camera_transforms[cam_idx])
    p    = w2c @ np.array([loc.x, loc.y, loc.z, 1.0])
    if p[0] <= 0:           # behind camera
        return None
    u = int(CAM_K[0, 0] * p[1] / p[0] + CAM_K[0, 2])
    v = int(CAM_K[1, 1] * (-p[2]) / p[0] + CAM_K[1, 2])
    if not (0 <= u < _CAM_W and 0 <= v < _CAM_H):
        return None
    return u, v, p[0]       # pixel x, pixel y, depth

# ── Siren ──────────────────────────────────────────────────────────────────
_siren_on = False

def _siren_loop():
    while _siren_on:
        winsound.Beep(1400, 180)
        winsound.Beep(900,  180)

def start_siren():
    global _siren_on
    if not _siren_on:
        _siren_on = True
        threading.Thread(target=_siren_loop, daemon=True).start()

def stop_siren():
    global _siren_on
    _siren_on = False

# ── YOLO with annotations ──────────────────────────────────────────────────
PHASE_COLORS = {
    0: (0, 255, 0),    # Green
    1: (0, 215, 255),  # Yellow
    2: (0, 0, 255),    # Red
}
PHASE_NAMES = {0:'GREEN', 1:'YELLOW', 2:'RED'}

def run_yolo_annotated(camera_idx, phase, wait_stats, yolo_count, is_emergency=False):
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

    # ── HUD overlay (compact — full stats live in the dashboard) ─────────
    int_id   = camera_int_map[camera_idx] + 1
    ph_color = PHASE_COLORS[phase]
    ph_name  = PHASE_NAMES[phase]

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (640, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    cv2.putText(annotated, f"INTERSECTION {int_id}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated, f"YOLO: {count} vehicles",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.circle(annotated, (580, 28), 17, ph_color, -1)
    cv2.circle(annotated, (580, 28), 17, (255, 255, 255), 1)
    cv2.putText(annotated, ph_name,
                (492, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ph_color, 2)

    # ── Emergency vehicle red bounding boxes (CARLA actor projection) ────
    int_center = intersection_centers[camera_int_map[camera_idx]]
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(int_center) < ROI_RADIUS:
            if any(kw in actor.type_id.lower() for kw in EMERGENCY_KW):
                pt = project_actor(actor, camera_idx)
                if pt is None:
                    continue
                u, v, depth = pt
                ext = actor.bounding_box.extent
                hw  = max(22, int(CAM_K[0, 0] * ext.y / depth))
                hh  = max(22, int(CAM_K[1, 1] * ext.z / depth))
                x1, y1 = max(0, u - hw), max(0, v - hh)
                x2, y2 = min(_CAM_W-1, u + hw), min(_CAM_H-1, v + hh)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                name = actor.type_id.split('.')[-1].upper()
                cv2.putText(annotated, f"EMERGENCY: {name}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if is_emergency:
        cv2.rectangle(annotated, (0, 65), (640, 100), (0, 0, 180), -1)
        cv2.putText(annotated, "!! EMERGENCY OVERRIDE — SIGNAL GREEN !!",
                    (10, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

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

# ── Signal & Emergency State ────────────────────────────────────────────────
phase_states       = [carla.TrafficLightState.Green,
                      carla.TrafficLightState.Yellow,
                      carla.TrafficLightState.Red]
phases             = [0] * NUM_INT
counters           = [0] * NUM_INT
phase_durations    = [100] * NUM_INT
yolo_counts        = [0] * NUM_INT
emergency_active   = [False] * NUM_INT  # True while holding green for emergency
emergency_cooldown = [0]     * NUM_INT  # ticks of fixed-time buffer after clearance

COOLDOWN_TICKS = 200  # ~2 signal cycles before RL resumes after an emergency

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

# ── Dashboard ──────────────────────────────────────────────────────────────
DASH_W   = 420
HIST_LEN = 120   # rolling window for the wait-time graph

def draw_dashboard(phases, yolo_counts, stats_list, emg_active, emg_cooldown,
                   wait_history, tick_count):
    H, W = 640, DASH_W
    panel = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)

    # Title bar
    cv2.rectangle(panel, (0, 0), (W, 46), (28, 28, 50), -1)
    cv2.putText(panel, "TRAFFIC INTELLIGENCE SYSTEM",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
    cv2.putText(panel, f"Tick {tick_count:>6d}   |   {NUM_INT} Intersections Active",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 160), 1)
    cv2.line(panel, (0, 46), (W, 46), (60, 60, 100), 1)

    PH_TXT = ['GREEN', 'YELLOW', 'RED']
    PH_COL = [(0, 210, 0), (0, 210, 230), (0, 0, 230)]

    section_tops = [54, 232]   # fixed y-start for each intersection card

    for idx in range(NUM_INT):
        stats = stats_list[idx]
        ph    = phases[idx]
        sy    = section_tops[idx]

        # Card background + border
        cv2.rectangle(panel, (4, sy),      (W-4, sy+174), (24, 24, 42), -1)
        cv2.rectangle(panel, (4, sy),      (W-4, sy+174), (50, 50, 80),  1)

        # Intersection header
        cv2.rectangle(panel, (4, sy),      (W-4, sy+22),  (38, 38, 65), -1)
        cv2.putText(panel, f"  INTERSECTION {idx+1}",
                    (6, sy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1)

        # Mode badge
        if emg_active[idx]:
            mtxt, mcol, mbg = "EMERGENCY OVERRIDE", (110, 110, 255), (50,  0, 100)
        elif emg_cooldown[idx] > 0:
            mtxt, mcol, mbg = "POST-EMERGENCY",     (0,  200, 230), ( 0, 45,  70)
        else:
            mtxt, mcol, mbg = "RL CONTROL",         (0,  230, 110), ( 0, 50,  20)
        cv2.rectangle(panel, (6, sy+24), (W-6, sy+41), mbg, -1)
        cv2.putText(panel, f"MODE: {mtxt}",
                    (10, sy+37), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mcol, 1)

        # Signal lights (3 circles)
        for pi in range(3):
            cx  = 26 + pi * 36
            col = PH_COL[pi] if pi == ph else (28, 28, 28)
            cv2.circle(panel, (cx, sy+58), 13, col, -1)
            if pi == ph:
                cv2.circle(panel, (cx, sy+58), 13, (255, 255, 255), 1)
        cv2.putText(panel, PH_TXT[ph],
                    (130, sy+64), cv2.FONT_HERSHEY_SIMPLEX, 0.68, PH_COL[ph], 2)

        # Stats rows
        q, aw, yc = (stats['queue_length'], stats['avg_waiting_time'],
                     yolo_counts[idx])
        aw_col = (0, 70, 220) if aw > 15 else (0, 200, 100)
        for ri, (lbl, val, col) in enumerate([
            ("Vehicles (YOLO)", f"{yc}",       (180, 180, 180)),
            ("Queue Length",    f"{q}",         (180, 180, 180)),
            ("Avg Wait",        f"{aw:.1f}s",   aw_col),
        ]):
            ry = sy + 82 + ri * 18
            cv2.putText(panel, lbl, (10, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 140), 1)
            cv2.putText(panel, val, (W-68, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # Queue bar
        bx0, bx1, by = 10, W-10, sy+138
        fill = int((bx1-bx0) * min(q, 20) / 20)
        cv2.rectangle(panel, (bx0, by), (bx1, by+9), (35, 35, 50), -1)
        if fill > 0:
            cv2.rectangle(panel, (bx0, by), (bx0+fill, by+9),
                          (0, 60, 210) if q > 10 else (0, 185, 80), -1)
        cv2.putText(panel, "QUEUE", (bx1-44, by+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 100), 1)

        # Emergency in-ROI badge (conditional)
        if emg_active[idx]:
            cv2.rectangle(panel, (6, sy+152), (W-6, sy+170), (60, 0, 0), -1)
            cv2.putText(panel, "!! EMERGENCY VEHICLE IN ROI !!",
                        (8, sy+165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 90, 90), 1)

    # ── Live wait-time graph ───────────────────────────────────────────────
    GT, GX0, GY0, GX1, GY1 = 416, 8, 434, W-8, 622
    GH, GW = GY1-GY0, GX1-GX0

    cv2.rectangle(panel, (0, GT-18), (W, GT), (28, 28, 50), -1)
    cv2.putText(panel, "AVG WAIT TIME HISTORY  (last 120 readings)",
                (8, GT-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 200), 1)
    cv2.rectangle(panel, (GX0, GY0), (GX1, GY1), (22, 22, 38), -1)
    cv2.rectangle(panel, (GX0, GY0), (GX1, GY1), (55, 55, 85),  1)
    for gi in range(1, 4):
        gy = GY0 + int(GH * gi / 4)
        cv2.line(panel, (GX0, gy), (GX1, gy), (38, 38, 60), 1)

    all_vals = [v for h in wait_history for v in h]
    max_val  = max(max(all_vals) if all_vals else 30, 30)

    LINE_COLS = [(0, 220, 110), (80, 150, 255)]
    for idx in range(NUM_INT):
        hist = wait_history[idx]
        if len(hist) < 2:
            continue
        n   = len(hist)
        pts = [(GX0 + int(GW*i/(n-1)),
                max(GY0, min(GY1, GY1 - int(GH * min(v, max_val) / max_val))))
               for i, v in enumerate(hist)]
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i-1], pts[i], LINE_COLS[idx], 1)
        lx = GX0 + 6 + idx * 90
        cv2.circle(panel, (lx+4, GY0+11), 3, LINE_COLS[idx], -1)
        cv2.putText(panel, f"Int {idx+1}", (lx+10, GY0+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, LINE_COLS[idx], 1)

    cv2.putText(panel, f"{max_val:.0f}s", (GX0, GY0+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 120), 1)
    cv2.putText(panel, "0s", (GX0, GY1-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 120), 1)

    # Status bar
    any_emg = any(emg_active)
    cv2.rectangle(panel, (0, 623), (W, 640),
                  (50, 0, 0) if any_emg else (0, 28, 0), -1)
    cv2.putText(panel,
                "EMERGENCY PRIORITY ACTIVE" if any_emg
                else "SYSTEM: OPERATIONAL  |  DQN ACTIVE",
                (6, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (120, 80, 255) if any_emg else (70, 200, 70), 1)

    return panel

# ── Display window ─────────────────────────────────────────────────────────
WIN_NAME = "Traffic Intelligence Demo — Press Q to quit"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280 + DASH_W, 640)

# ── Main loop ──────────────────────────────────────────────────────────────
print(f"\nDemo running. Press Q in the window to quit.\n")

YOLO_INTERVAL         = 5
EMERGENCY_INTERVAL    = 1200   # ~60 s between emergency events (realistic)
EMERGENCY_LIFETIME    = 600    # ~30 s for vehicle to pass through
EMERGENCY_POST_GAP    = 800    # ~40 s quiet period after one clears before next

tick_count        = 0
emg_counter       = EMERGENCY_INTERVAL
emg_age           = 0
emg_post_gap      = 0
annotated_frames  = [None] * NUM_INT
wait_history      = [[]   for _ in range(NUM_INT)]

try:
    while True:
        world.tick()
        tick_count += 1

        # Emergency spawner — one vehicle at a time, realistic cadence
        if emergency_vehicle and emergency_vehicle.is_alive:
            emg_age += 1
            if emg_age >= EMERGENCY_LIFETIME:
                emergency_vehicle.destroy()
                emergency_vehicle = None
                emg_age     = 0
                emg_counter = 0
                emg_post_gap = EMERGENCY_POST_GAP   # enforce quiet gap
        else:
            if emg_post_gap > 0:
                emg_post_gap -= 1
            else:
                emg_counter += 1
                if emg_counter >= EMERGENCY_INTERVAL:
                    emergency_vehicle = spawn_emergency()
                    emg_age     = 0
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
                                                stats, yolo_counts[int_i],
                                                is_emergency=emergency_active[int_i])
                if frame is not None:
                    annotated_frames[int_i] = frame
                    yolo_counts[int_i]      = cnt
            # Update rolling wait-time history for the graph
            for idx in range(NUM_INT):
                s = wait_trackers[idx].get_stats()
                wait_history[idx].append(s['avg_waiting_time'])
                if len(wait_history[idx]) > HIST_LEN:
                    wait_history[idx].pop(0)

        # ── Emergency + DQN signal control ────────────────────────────────
        for idx, center in enumerate(intersection_centers):
            gt_count, avg_speed = compute_gt(center)
            emg                 = check_emergency(center)

            if emg:
                # STEP 2-4: Emergency present — hold GREEN, force RED elsewhere
                if not emergency_active[idx]:
                    print(f"[EMERGENCY] Detected at Intersection {idx+1}! Overriding RL.")
                emergency_active[idx]   = True
                emergency_cooldown[idx] = COOLDOWN_TICKS
                phases[idx]   = 0
                counters[idx] = 0
                set_phase(intersections[idx], carla.TrafficLightState.Green)
                for other in range(NUM_INT):
                    if other != idx and not emergency_active[other]:
                        phases[other]   = 2
                        counters[other] = 0
                        set_phase(intersections[other], carla.TrafficLightState.Red)

            elif emergency_active[idx]:
                # STEP 5: Emergency just left ROI — set RED, begin cooldown
                print(f"[EMERGENCY] Cleared at Intersection {idx+1}. "
                      f"Setting RED, resuming RL in {COOLDOWN_TICKS} ticks.")
                emergency_active[idx] = False
                phases[idx]   = 2
                counters[idx] = 0
                set_phase(intersections[idx], carla.TrafficLightState.Red)

            elif emergency_cooldown[idx] > 0:
                # STEP 6: Post-emergency fixed-time buffer (let traffic rebalance)
                emergency_cooldown[idx] -= 1
                counters[idx] += 1
                if counters[idx] >= phase_durations[idx]:
                    counters[idx] = 0
                    phases[idx]   = (phases[idx] + 1) % len(phase_states)
                    set_phase(intersections[idx], phase_states[phases[idx]])

            else:
                # STEP 7: Normal RL control
                state = build_state(
                    yolo_count=yolo_counts[idx], gt_count=gt_count,
                    avg_speed=avg_speed, current_phase=phases[idx],
                    phase_counter=counters[idx],
                    phase_duration=phase_durations[idx],
                    emergency_flag=0,
                    elapsed_seconds=timestamp
                )
                action = agents[idx].act(state)
                counters[idx] += 1
                if counters[idx] >= phase_durations[idx]:
                    counters[idx]        = 0
                    phases[idx]          = (phases[idx] + 1) % len(phase_states)
                    phase_durations[idx] = dqn_phase_duration(action, yolo_counts[idx])
                    set_phase(intersections[idx], phase_states[phases[idx]])

        # Siren — on while any intersection has an active emergency
        if any(emergency_active):
            start_siren()
        else:
            stop_siren()

        # Build display — 2 camera feeds + dashboard
        valid = [f for f in annotated_frames if f is not None]
        if len(valid) == NUM_INT:
            stats_now = [wait_trackers[i].get_stats() for i in range(NUM_INT)]
            dash      = draw_dashboard(phases, yolo_counts, stats_now,
                                       emergency_active, emergency_cooldown,
                                       wait_history, tick_count)
            cv2.imshow(WIN_NAME, np.hstack(valid + [dash]))

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
