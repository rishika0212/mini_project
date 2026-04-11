"""
train_yolo.py
=============
Fine-tunes YOLOv8n on your CARLA dataset.

Run AFTER collect_dataset.py has finished.

Usage:
    python train_yolo.py

Output:
    runs/detect/carla_traffic/weights/best.pt  ← your custom model
"""

from ultralytics import YOLO
from pathlib import Path
import os

DATASET_YAML = "dataset/data.yaml"
BASE_MODEL   = "yolov8n.pt"       # start from pretrained weights
EPOCHS       = 50                  # 50 epochs is enough for fine-tuning
IMG_SIZE     = 640
BATCH        = 8                   # reduce to 4 if you get OOM errors
PROJECT      = "runs/detect"
NAME         = "carla_traffic"

print("="*50)
print("Fine-tuning YOLOv8n on CARLA dataset")
print("="*50)

# Check dataset exists
if not Path(DATASET_YAML).exists():
    print("ERROR: dataset/data.yaml not found.")
    print("Run collect_dataset.py first.")
    exit(1)

# Count images
train_count = len(list(Path("dataset/images/train").glob("*.png")))
val_count   = len(list(Path("dataset/images/val").glob("*.png")))
print(f"Train images: {train_count}")
print(f"Val images:   {val_count}")
print(f"Epochs:       {EPOCHS}")
print(f"Batch size:   {BATCH}")
print()

if train_count < 50:
    print("WARNING: Very few training images. Run collect_dataset.py longer.")
    print("Continuing anyway...")

# Load base model
model = YOLO(BASE_MODEL)

# Fine-tune on your data
print("Starting training...")
results = model.train(
    data      = DATASET_YAML,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    project   = PROJECT,
    name      = NAME,
    exist_ok  = True,
    patience  = 10,           # early stopping
    save      = True,
    plots     = True,         # generates training graphs
    verbose   = True,
    device    = 0,            # GPU 0 — change to 'cpu' if no GPU
)

print()
print("="*50)
print("Training complete!")
print("="*50)

best_model = f"{PROJECT}/{NAME}/weights/best.pt"
if Path(best_model).exists():
    print(f"Best model saved: {best_model}")
    print()
    print("Update main.py to use your custom model:")
    print(f'  model = YOLO("{best_model}")')
    print()
    print("Or copy it to your project folder:")
    print(f'  copy {best_model} C:\\CARLA\\Traffic_project\\vehicle_detector.pt')
else:
    print("Warning: best.pt not found. Check training output.")
