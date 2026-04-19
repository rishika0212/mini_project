# Traffic Intelligence System - Complete Workflow Guide

## Quick Reference: Command Sequences

### 🎯 Option 1: Just See It Working (5-10 min)
```bash
# Terminal 1: Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# Terminal 2: Run interactive demo
python demo.py
# Press Q to quit after 2-3 minutes
```

### 🚀 Option 2: Train Then Demo (3-4 hours + demo time)
```bash
# Terminal 1: Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# Terminal 2: Train DQN agent
python train_orchestrator.py
# Trains 350 episodes (sparse) + 150 episodes (dense)
# Creates: data/dqn_weights_int1.json

# Terminal 3: Run demo with trained agent
python demo.py
```

### 📊 Option 3: Evaluate & Compare (30-45 min + plots)
```bash
# Terminal 1: Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# Terminal 2: Evaluate DQN policy
python evaluate.py --policy dqn --episodes 30

# Terminal 3: Evaluate Fixed-Time baseline
python evaluate.py --policy fixed --episodes 30

# Terminal 4: Evaluate Random baseline
python evaluate.py --policy random --episodes 30

# Terminal 5: Generate plots
python plot_results.py
# Creates: data/plots/avg_wait_comparison.png, etc.
```

### 🧪 Option 4: Full Development Workflow (3-5 hours + demo)
```bash
# PHASE 1: Train with sparse traffic (1-2 hours)
python train_orchestrator.py

# PHASE 2: Run live demo
python demo.py

# PHASE 3: Evaluate performance
python evaluate.py --policy dqn --episodes 30
python evaluate.py --policy fixed --episodes 30

# PHASE 4: Visualize results
python plot_results.py

# PHASE 5: Test specific scenarios
python test_all_scenarios.py
```

---

## Detailed Command Reference

### DEMO (Interactive Visualization)

```bash
python demo.py
```

**What it does:**
- Runs real-time interactive demo with overhead camera
- Shows DQN, Emergency, Fallback, Recovery scenarios
- Emergency vehicles spawn every 40s
- Displays RED boxes for ambulances

**Output:**
- Live window: Overhead camera (760px) + Dashboard (420px)
- Console: Mode switches, emergency events, diagnostics

**Duration:** Run for 3-5 minutes to see all scenarios
**Controls:** Q = quit

---

### MAIN (Full System with YOLO)

#### Run in Evaluation Mode (Default)
```bash
python main.py
```

**What it does:**
- Runs complete system with YOLO detection
- Uses pre-trained DQN agent (greedy mode, no exploration)
- Logs detailed metrics to CSV
- Shows camera views + HUD

**Options:**
```bash
# Use custom YOLO model
python main.py --model yolov8s.pt

# Disable YOLO, use ground-truth vehicle counts (faster)
python main.py --no-yolo

# Both options
python main.py --model yolov8s.pt --no-yolo --train
```

**Output:**
- CSV log: `data/main_int1_episode_X.csv`
- Console: Vehicle counts, wait times, mode changes
- Window: Live traffic visualization

**Duration:** Runs indefinitely until interrupted (Ctrl+C)

---

#### Train DQN Agent
```bash
python main.py --train
```

**What it does:**
- Trains DQN agent from scratch (or continues existing)
- 1 episode = 600 simulation ticks = 30 seconds
- Uses YOLO for detection (slower, more realistic)
- Logs rewards and metrics

**Key Parameters (inside main.py to modify):**
- `EPISODES = 500` - number of training episodes
- `RUSH_HOUR_INTERVAL = 2400` - vehicle spawn frequency
- `EMERGENCY_INTERVAL = 240` - ambulance spawn interval

**Output:**
- Trained weights: `data/dqn_weights_int1.json`
- CSV log: `data/main_int1_episode_X.csv`
- Checkpoint every 10 episodes

**Duration:** ~1 hour per 100 episodes (with YOLO)
**To Speed Up:** Add `--no-yolo` flag for 30-40% faster training

---

### TRAIN ORCHESTRATOR (Recommended Training)

```bash
python train_orchestrator.py
```

**What it does:**
- Two-phase automated training with optimization
- Phase 1: 350 episodes, sparse traffic (135 vehicles) = ~1.5 hours
- Phase 2: 150 episodes, dense traffic (420 vehicles) = ~1-2 hours
- Automatically disables YOLO during training (fast mode)
- Real-time progress output

**Training Schedule:**
```
Phase 1: Sparse (135 vehicles)
├─ Episodes: 0-350
├─ Time: ~90 minutes
├─ Focus: Learn basic phase transitions
└─ Output: Intermediate weights

Phase 2: Dense (420 vehicles)
├─ Episodes: 351-500
├─ Time: ~60 minutes
├─ Focus: Handle congestion & emergencies
└─ Output: Final weights (data/dqn_weights_int1.json)
```

**Output:**
- Final trained agent: `data/dqn_weights_int1.json`
- Training logs: `data/training_*.csv`
- Real-time console output with reward tracking

**Duration:** ~3-4 hours total
**Status:** Saves progress periodically, can resume if interrupted

---

### EVALUATE (Test Different Policies)

```bash
python evaluate.py --policy dqn --episodes 30
```

**Policies Available:**
- `dqn` - Trained neural network agent (intelligent)
- `fixed` - Fixed 30-30-5 cycle (baseline)
- `random` - Random phase selection (worst case)

**Options:**
```bash
python evaluate.py --policy dqn --episodes 50
python evaluate.py --policy fixed --episodes 30
python evaluate.py --policy random --episodes 30
```

**Metrics Logged:**
- Average wait time per episode
- Queue length variations
- Throughput (vehicles/min)
- Emergency response time
- Total reward (for DQN)

**Output:**
- CSV files: `data/eval_dqn.csv`, `data/eval_fixed.csv`, `data/eval_random.csv`
- Console: Episode-by-episode metrics

**Duration:** ~30 seconds per episode
- DQN: ~15 min for 30 episodes
- Fixed: ~15 min for 30 episodes
- Random: ~15 min for 30 episodes
- Total for all 3: ~45 min

---

### PLOT RESULTS (Visualization)

**Prerequisites:** Must run evaluate.py first for each policy

```bash
# Step 1: Generate CSV data (do this first)
python evaluate.py --policy dqn --episodes 30
python evaluate.py --policy fixed --episodes 30
python evaluate.py --policy random --episodes 30

# Step 2: Create plots
python plot_results.py
```

**Generates:**
- `data/plots/avg_wait_comparison.png` - Average waiting time vs policy
- `data/plots/queue_comparison.png` - Queue length over time
- `data/plots/throughput_comparison.png` - Vehicles per minute
- `data/plots/reward_curve.png` - DQN training reward history

**Output Folder:** `data/plots/`

---

### TEST SCENARIOS (Development & Validation)

```bash
python test_all_scenarios.py
```

**Tests:**
1. Normal DQN operation (300 ticks, no emergencies)
2. Emergency vehicle scenario (ambulance appears at 150 ticks)
3. Fallback mode activation (low confidence triggers fixed-time)
4. Recovery phase after emergency

**Output:**
- Console: Pass/Fail for each scenario
- Assertion errors if something breaks
- Metrics for each test case

**Run After:** Any code changes to verify system still works

---

### LOGIC TESTS (Unit Testing)

```bash
python test_system_logic.py
```

**Tests:**
- FallbackController state transitions
- Signal phase transitions
- Emergency state machine logic
- Confidence calculation

**Output:**
- Unit test results
- Pass/Fail counts

---

## Recommended Workflows

### 👀 Workflow A: Just Want to See It
```bash
# 1. Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# 2. Run demo
python demo.py

# Time: 5-10 minutes
```

### 🎓 Workflow B: Understand the System
```bash
# 1. Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# 2. Run demo to see all scenarios
python demo.py  # 5 min

# 3. Run tests to validate logic
python test_all_scenarios.py  # 5 min

# 4. Check baseline metrics
python evaluate.py --policy fixed --episodes 10  # 5 min

# Time: 15 minutes
```

### 🚀 Workflow C: Full Evaluation
```bash
# 1. Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# 2. Train a new agent
python train_orchestrator.py  # 3-4 hours

# 3. Run demo with trained agent
python demo.py  # 5 min

# 4. Evaluate all policies
python evaluate.py --policy dqn --episodes 30  # 15 min
python evaluate.py --policy fixed --episodes 30  # 15 min
python evaluate.py --policy random --episodes 30  # 15 min

# 5. Generate comparison plots
python plot_results.py  # 2 min

# Time: ~4.5 hours
```

### 🔬 Workflow D: Development & Testing
```bash
# 1. Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# 2. Make code changes...

# 3. Run logic tests
python test_system_logic.py  # 2 min

# 4. Run scenario tests
python test_all_scenarios.py  # 10 min

# 5. Run demo to verify visually
python demo.py  # 3 min

# 6. If all good, evaluate performance
python evaluate.py --policy dqn --episodes 10  # 5 min

# Time: 30 minutes per iteration
```

---

## File Outputs

### Data Directory Structure
```
data/
├── dqn_weights_int1.json          # Trained DQN agent weights
├── main_int1_episode_1.csv         # Main system logs
├── main_int1_episode_2.csv
├── eval_dqn.csv                    # Evaluation results (DQN policy)
├── eval_fixed.csv                  # Evaluation results (Fixed-time)
├── eval_random.csv                 # Evaluation results (Random)
├── training_log.txt                # Training progress
├── plots/
│   ├── avg_wait_comparison.png
│   ├── queue_comparison.png
│   ├── throughput_comparison.png
│   └── reward_curve.png
└── checkpoints/                    # Training checkpoints
    └── dqn_weights_int1_checkpoint_X.json
```

### CSV Columns (main_int1_episode_X.csv)
```
tick, mode, phase, trans_state, vehicles, queue_length, avg_waiting_time,
throughput_vpm, yolo_confidence, low_conf_count, emergency_state,
emergency_flag, emergency_timeout, grace_counter, preclear_counter,
recovery_counter, phase_counter, signal_ns, signal_ew
```

---

## Performance Expectations

| Operation | Time | Hardware |
|-----------|------|----------|
| Demo (5 min) | ~5 min | Any |
| Training (500 episodes) | 3-4 hours | GPU recommended |
| Training (per 100 episodes) | 45-60 min | GPU: 30-40 min |
| Evaluate (30 episodes) | 15-20 min | GPU recommended |
| Plot generation | <1 min | Any |
| Test suite | 5-10 min | Any |

**GPU Acceleration:**
- With GPU: 3-4 hours for full training
- Without GPU: 6-8 hours for full training
- YOLO inference: 10x faster with GPU

---

## Debugging Commands

### Check CARLA Connection
```bash
python -c "import carla; client=carla.Client('localhost',2000); print('Connected to CARLA:', client.get_server_version())"
```

### Verify All Dependencies
```bash
python -c "
import carla
import cv2
import numpy as np
from ultralytics import YOLO
from dqn_agent import DQNAgent
from ground_sensors import IntersectionGroundSensors
print('All dependencies OK')
"
```

### Quick Demo Check
```bash
python demo.py --no-render  # (if supported - checks setup without window)
```

### Check Trained Weights
```bash
python -c "
import json
with open('data/dqn_weights_int1.json') as f:
    w = json.load(f)
    print(f'Trained weights loaded: {len(w)} keys')
    print('Layers:', list(w.keys())[:5], '...')
"
```

---

## Choosing the Right Command

### I want to...

**See the system working immediately**
```bash
python demo.py
```

**Train a new DQN agent**
```bash
python train_orchestrator.py
```

**Compare DQN vs Fixed-Time vs Random**
```bash
python evaluate.py --policy dqn --episodes 30
python evaluate.py --policy fixed --episodes 30
python evaluate.py --policy random --episodes 30
python plot_results.py
```

**Test that everything still works after code changes**
```bash
python test_all_scenarios.py
```

**Run the full system with detailed logging**
```bash
python main.py
```

**Run the full system and train the agent**
```bash
python main.py --train
```

**Debug why something isn't working**
```bash
python test_system_logic.py
python test_all_scenarios.py
python demo.py  # Watch console output carefully
```

---

## Common Issues & Fixes

| Problem | Command to Try | Fix |
|---------|---|---|
| "Cannot reach CARLA" | Ensure CARLA running on localhost:2000 | Start CARLA first |
| No emergency vehicles | Check console for "[EMERGENCY]" messages | Run for longer, happens every 40s |
| Demo window is frozen | Check console for errors | May be waiting for CARLA tick |
| Training too slow | Add `--no-yolo` flag | Use ground sensors instead of YOLO |
| Out of memory | Reduce EPISODES or MAX_VEHICLES | Lower training intensity |
| Plots not generating | Verify CSV files exist in data/ | Run evaluate.py first |
| DQN stuck in FALLBACK | Check confidence calculation | May need more training |

---

## Next Steps After Running Commands

### After `demo.py`
- Note emergency vehicle RED box behavior
- Watch mode switches (DQN → FALLBACK → RECOVERY)
- Check dashboard metrics update live

### After `train_orchestrator.py`
- Check `data/dqn_weights_int1.json` created
- Run demo.py to see trained agent in action
- Compare with `evaluate.py --policy dqn`

### After `evaluate.py` (all 3 policies)
- Run `plot_results.py` to see comparison charts
- Note DQN typically beats Fixed-Time (lower wait times)
- Random policy shows why structured control is needed

### After `plot_results.py`
- Review generated PNG files in `data/plots/`
- Shows DQN performance advantage visually
- Use for reports/presentations

---

**Summary**: Start with `python demo.py` for quick visualization, then `python train_orchestrator.py` to get a trained agent, then use `evaluate.py` + `plot_results.py` for detailed performance analysis.
