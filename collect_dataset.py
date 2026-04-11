"""
collect_dataset.py
==================
Collects a YOLO-format training dataset from CARLA automatically.

Uses two cameras per intersection:
  - RGB camera      → training images
  - Semantic camera → auto-generates bounding box annotations

No manual labeling needed. CARLA's semantic segmentation gives us
perfect ground truth bounding boxes for free.

Output structure:
  dataset/
    images/
      train/   (80%)
      val/     (20%)
    labels/
      train/
      val/
    data.yaml  (YOLO config file)

Usage:
    python collect_dataset.py
"""

import carla
import random
import os
import cv2
import numpy as np
import threading
import math
import yaml
import shutil
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
IMAGES_PER_INTERSECTION = 300    # collect 300 images per intersection
TARGET_TOTAL            = 900    # 900 total = decent dataset for fine-tuning
TRAIN_SPLIT             = 0.8
IMG_SIZE                = 640
CAPTURE_INTERVAL        = 15     # capture every 15 ticks
MIN_BOX_SIZE            = 10     # ignore tiny detections (pixels)

# CARLA semantic segmentation class IDs
# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
VEHICLE_SEM_ID = 10   # vehicles in CARLA semantic

# YOLO class mapping
YOLO_CLASSES = {
    'car':        0,
    'truck':      1,
    'bus':        2,
    'motorcycle': 3,
    'emergency':  4,
}

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

# ── Dataset folders ────────────────────────────────────────────────────────
BASE = Path("dataset")
for split in ["train", "val"]:
    (BASE / "images" / split).mkdir(parents=True, exist_ok=True)
    (BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

print("Dataset folders created.")

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
intersections        = valid_groups[:3]
intersection_centers = []

for i, group in enumerate(intersections):
    x = sum(l.get_location().x for l in group)/len(group)
    y = sum(l.get_location().y for l in group)/len(group)
    z = sum(l.get_location().z for l in group)/len(group)
    intersection_centers.append(carla.Location(x=x, y=y, z=z))
    print(f"  Int {i+1}: ({x:.1f}, {y:.1f})")

# ── Spawn vehicles (dense traffic for good dataset) ────────────────────────
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

# Spawn densely near intersections
for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center) < 80]
    for sp in nearby[:20]:
        bp = random.choice(car_bps)
        v  = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True)
            vehicles.append(v)

# Background traffic
for sp in spawn_points[:30]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)
        vehicles.append(v)

# Spawn some emergency vehicles too
lib          = world.get_blueprint_library()
emg_bps      = list(lib.filter("vehicle.*ambulance*")) + \
               list(lib.filter("vehicle.*firetruck*"))
emg_vehicles = []
for center in intersection_centers:
    nearby = [sp for sp in spawn_points if sp.location.distance(center) < 60]
    if nearby and emg_bps:
        sp = random.choice(nearby)
        v  = world.try_spawn_actor(random.choice(emg_bps), sp)
        if v:
            v.set_autopilot(True)
            emg_vehicles.append(v)
            vehicles.append(v)

print(f"Spawned {len(vehicles)} vehicles ({len(emg_vehicles)} emergency).")

# Let traffic settle
for _ in range(40):
    world.tick()

# ── Camera setup ───────────────────────────────────────────────────────────
# RGB camera
rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', str(IMG_SIZE))
rgb_bp.set_attribute('image_size_y', str(IMG_SIZE))
rgb_bp.set_attribute('fov', '90')
rgb_bp.set_attribute('sensor_tick', '0.1')

# Semantic segmentation camera (same position)
sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
sem_bp.set_attribute('image_size_x', str(IMG_SIZE))
sem_bp.set_attribute('image_size_y', str(IMG_SIZE))
sem_bp.set_attribute('fov', '90')
sem_bp.set_attribute('sensor_tick', '0.1')

# ── Shared storage ─────────────────────────────────────────────────────────
rgb_frames = {}
sem_frames = {}
rgb_locks  = {}
sem_locks  = {}
rgb_counts = {}

cameras = []

for i, center in enumerate(intersection_centers):
    rgb_frames[i] = None
    sem_frames[i] = None
    rgb_locks[i]  = threading.Lock()
    sem_locks[i]  = threading.Lock()
    rgb_counts[i] = 0

    transform = carla.Transform(
        carla.Location(x=center.x-20, y=center.y, z=center.z+12),
        carla.Rotation(pitch=-45, yaw=0)
    )

    rgb_cam = world.spawn_actor(rgb_bp, transform)
    sem_cam = world.spawn_actor(sem_bp, transform)
    cameras.extend([rgb_cam, sem_cam])

    def on_rgb(image, idx=i):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:,:,:3].copy()
        with rgb_locks[idx]:
            rgb_frames[idx] = bgr
            rgb_counts[idx] += 1

    def on_sem(image, idx=i):
        # Convert semantic to class ID array
        image.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        with sem_locks[idx]:
            sem_frames[idx] = arr[:,:,:3].copy()

    rgb_cam.listen(lambda img, f=on_rgb: f(img))
    sem_cam.listen(lambda img, f=on_sem: f(img))

print("Cameras attached (RGB + Semantic).")

# Warmup
for _ in range(30):
    world.tick()
print("Cameras warmed up.")

# ── Bounding box extraction from semantic ──────────────────────────────────

# CARLA CityScapes palette colors for vehicles
# Vehicle class = tag 10 → rendered as (0, 0, 142) in CityScapes palette
VEHICLE_COLOR = np.array([142, 0, 0])    # BGR in OpenCV

def get_vehicle_color(sem_frame, int_idx):
    """
    Extract vehicle bounding boxes from semantic frame.
    Returns list of (x_center, y_center, width, height) normalized 0-1
    """
    h, w = sem_frame.shape[:2]

    # Mask pixels that match vehicle color (with tolerance)
    diff    = np.abs(sem_frame.astype(int) - VEHICLE_COLOR)
    mask    = (diff.sum(axis=2) < 60).astype(np.uint8) * 255

    # Find connected components (each = one vehicle)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    boxes = []
    for label in range(1, num_labels):  # skip background (label 0)
        x, y, bw, bh, area = stats[label]

        # Filter tiny boxes
        if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
            continue
        if area < MIN_BOX_SIZE * MIN_BOX_SIZE:
            continue

        # Normalize to 0-1 for YOLO format
        x_center = (x + bw/2) / w
        y_center = (y + bh/2) / h
        norm_w   = bw / w
        norm_h   = bh / h

        # Clamp to [0,1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        norm_w   = max(0, min(1, norm_w))
        norm_h   = max(0, min(1, norm_h))

        boxes.append((x_center, y_center, norm_w, norm_h))

    return boxes

def classify_vehicle(actor):
    """Classify vehicle as YOLO class index."""
    type_id = actor.type_id.lower()
    if 'ambulance' in type_id or 'firetruck' in type_id:
        return YOLO_CLASSES['emergency']
    elif 'truck' in type_id:
        return YOLO_CLASSES['truck']
    elif 'bus' in type_id:
        return YOLO_CLASSES['bus']
    elif 'motorcycle' in type_id or 'bike' in type_id:
        return YOLO_CLASSES['motorcycle']
    else:
        return YOLO_CLASSES['car']

# ── Collect dataset ────────────────────────────────────────────────────────
print(f"\nCollecting {IMAGES_PER_INTERSECTION} images per intersection...")
print("This will take a few minutes. Do not close CARLA.\n")

collected    = {i: 0 for i in range(len(intersection_centers))}
total        = 0
tick_count   = 0
img_index    = 0

try:
    while total < TARGET_TOTAL:
        world.tick()
        tick_count += 1

        if tick_count % CAPTURE_INTERVAL != 0:
            continue

        for idx in range(len(intersection_centers)):
            if collected[idx] >= IMAGES_PER_INTERSECTION:
                continue

            with rgb_locks[idx]:
                rgb = rgb_frames[idx]
            with sem_locks[idx]:
                sem = sem_frames[idx]

            if rgb is None or sem is None:
                continue

            # Get bounding boxes from semantic
            boxes = get_vehicle_color(sem, idx)

            # Skip frames with no vehicles
            if len(boxes) == 0:
                continue

            # Determine train/val split
            split = "train" if (img_index % 10) < 8 else "val"

            # Save image
            img_name = f"carla_int{idx+1}_{img_index:06d}"
            img_path = BASE / "images" / split / f"{img_name}.png"
            cv2.imwrite(str(img_path), rgb)

            # Save YOLO label
            lbl_path = BASE / "labels" / split / f"{img_name}.txt"
            with open(lbl_path, 'w') as f:
                for (xc, yc, bw, bh) in boxes:
                    # Use car class (0) for all — semantic doesn't distinguish
                    # vehicle subtypes easily from color alone
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            collected[idx] += 1
            total          += 1
            img_index      += 1

            if total % 50 == 0:
                print(f"  Collected {total}/{TARGET_TOTAL} images "
                      f"({[collected[i] for i in range(len(intersection_centers))]})")

except KeyboardInterrupt:
    print("\nCollection stopped early.")

finally:
    # Count what we have
    train_imgs = len(list((BASE/"images"/"train").glob("*.png")))
    val_imgs   = len(list((BASE/"images"/"val").glob("*.png")))
    print(f"\nDataset collected:")
    print(f"  Train: {train_imgs} images")
    print(f"  Val:   {val_imgs} images")

    # Write data.yaml for YOLO training
    yaml_data = {
        'path':  str(BASE.absolute()),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    1,
        'names': ['vehicle']
    }
    with open(BASE / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"  Saved: dataset/data.yaml")

    # Cleanup
    for cam in cameras:
        cam.stop()
        cam.destroy()
    for v in vehicles:
        if v.is_alive:
            v.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("\nDataset collection complete.")
    print("Next step: python train_yolo.py")
