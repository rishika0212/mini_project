# YOLOv8 + Deep Q-Network Based Adaptive Traffic Signal Control
### with Emergency Vehicle Prioritization Using CARLA Simulation

**Dayananda Sagar College of Engineering**
Department of Robotics and Artificial Intelligence — 22RI66 Mini Project II

---

## Project Overview

A vision-driven adaptive traffic signal control system that:
- Detects vehicles in real time using **YOLOv8n** from overhead RGB cameras
- Controls traffic signals intelligently using a **Deep Q-Network (DQN)** agent
- Prioritizes **emergency vehicles** (ambulance, fire truck) automatically
- Falls back to **fixed-time control** when camera confidence drops below threshold
- Simulated in **CARLA v0.9.x** across 3 intersections simultaneously

### Key Results (30-episode evaluation)
| Metric | DQN Agent | Fixed-Time | Random |
|--------|-----------|------------|--------|
| Avg Waiting Time | **7.15s** | 8.82s | 10.16s |
| Throughput (veh/min) | **14.24** | 6.03 | 4.10 |
| Avg Speed (m/s) | **2.01** | 0.94 | 0.65 |
| vs Fixed-Time | **18.9% better** | baseline | — |

---

## Project Structure

```
Traffic_project/
├── main.py                 # Full integrated system entrypoint
├── dqn_agent.py            # DQN neural network + training logic
├── waiting_time.py         # Per-vehicle waiting time tracker
├── fallback.py             # Safety fallback controller
├── evaluate.py             # Policy evaluation (DQN vs Fixed vs Random)
├── collect_dataset.py      # CARLA dataset collection for YOLO training
├── train_yolo.py           # Fine-tune YOLOv8 on CARLA data
├── demo.py                 # Live visual demo
├── yolov8n.pt              # Base YOLO model
├── data/
│   ├── dqn_weights_int1.json
│   ├── dqn_weights_int2.json
│   ├── dqn_weights_int3.json
│   ├── rl_states.csv
│   └── snapshots/
└── dataset/                # Created by collect_dataset.py
```

---

## Setup

### Requirements
- Python 3.8+
- CARLA 0.9.x
- NVIDIA GPU (recommended)

### Install dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
```

---

## How to Run

### 1. Start CARLA
```bash
CarlaUE4.exe -quality-level=Low
```

### 2. Run the full system (evaluation mode)
```bash
python main.py
```

### 3. Run with training mode
```bash
python main.py --train
```

### 4. Collect YOLO training data
```bash
python collect_dataset.py
```

### 5. Fine-tune YOLO on your data
```bash
python train_yolo.py
```

### 6. Use custom YOLO model
```bash
python main.py --model vehicle_detector.pt
```

### 7. Evaluate policies
```bash
python evaluate.py --policy dqn
python evaluate.py --policy fixed
python evaluate.py --policy random
```

### 8. Run live demo
```bash
python demo.py
```

---

## System Architecture

```
CARLA Simulation
      │
      ▼
RGB Camera (640×640, 90° FOV, 12m height, -45° pitch)
      │
      ▼
YOLOv8n Detection (conf threshold: 0.25)
      │
      ├─── Confidence ≥ 0.35 ──→ DQN Agent ──→ Signal Decision
      │                               │
      │                    State: [yolo_count, gt_count,
      │                            avg_speed, phase,
      │                            phase_timer, time_of_day,
      │                            emergency_flag]
      │
      ├─── Confidence < 0.35 ──→ Fixed-Time Fallback (30s/phase)
      │
      └─── Emergency detected ─→ Force Green (immediate)
```

---

## DQN Architecture

- **State space**: 7 features (normalized)
- **Action space**: 2 actions (keep phase / switch phase)
- **Network**: 7 → 32 → 32 → 2 (ReLU, numpy implementation)
- **Training**: Experience replay (buffer: 10,000), epsilon-greedy (ε: 1.0→0.1)
- **Episodes trained**: 114

---

## Team

| Name | USN |
|------|-----|
| Dhruv Khanna | 1DS23RI018 |
| Rishika | 1DS23RI040 |
| Saurav Kumar | IDS23RI045 |
| Yajat Kanaskar | IDS23RI060 |

**Guide**: Nishchitha M H, Assistant Professor, Robotics and AI

---

## References

1. Azfar et al. (2025) — Traffic Co-Simulation with CARLA + SUMO + RL
2. Li et al. (2021) — Deep RL for Traffic Signal Control, IEEE Access
3. Shankaran & Rajendran (2021) — Camera-Based Adaptive Traffic Control
4. VLMLight (2024) — Vision-Language Traffic Signal Control
