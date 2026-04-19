# Quick Command Reference

## Start CARLA (Terminal 1)
```bash
CarlaUE4.exe /Game/Carla/Maps/Town03
```

---

## 5-Minute Demo
```bash
python demo.py
# Shows: Overhead camera + Emergency RED boxes + DQN control
# Press Q to quit
```

---

## Training & Evaluation

### Option A: Fast Training (3-4 hours)
```bash
python train_orchestrator.py
# Auto 2-phase: sparse (350ep) → dense (150ep)
# Creates: data/dqn_weights_int1.json
```

### Option B: Full System Training (1-2 hours per policy)
```bash
python main.py --train
# Modify EPISODES variable in main.py to control length
```

---

## Evaluation & Plotting

### Test Different Policies
```bash
python evaluate.py --policy dqn --episodes 30
python evaluate.py --policy fixed --episodes 30
python evaluate.py --policy random --episodes 30
```

### Generate Comparison Plots
```bash
python plot_results.py
# Creates PNG charts in data/plots/
```

---

## System Components

| File | Purpose | Run With |
|------|---------|----------|
| demo.py | Interactive visualization | `python demo.py` |
| main.py | Full system, evaluation mode | `python main.py` |
| main.py | Training mode | `python main.py --train` |
| train_orchestrator.py | Auto 2-phase training | `python train_orchestrator.py` |
| evaluate.py | Compare policies | `python evaluate.py --policy dqn` |
| plot_results.py | Visualize results | `python plot_results.py` |
| test_all_scenarios.py | Validate system | `python test_all_scenarios.py` |

---

## Ground Sensors (No YOLO)
```bash
# Speed up training - skip YOLO, use ground truth
python main.py --train --no-yolo
python train_orchestrator.py  # Already has this optimization
```

---

## What You'll Get

### After demo.py
- Overhead camera view (55m above intersection)
- RED boxes for emergency vehicles
- Real-time dashboard: mode, queue, wait times
- Console output: event log

### After train_orchestrator.py
- Trained DQN agent weights
- Training progress logs
- Ready to run demo with trained model

### After evaluate.py (all 3 policies)
- CSV files comparing DQN vs Fixed vs Random
- Metrics: avg wait, queue, throughput

### After plot_results.py
- avg_wait_comparison.png
- queue_comparison.png
- throughput_comparison.png
- reward_curve.png

---

## Typical Workflow

```
1. python demo.py                          # See it work (5 min)
2. python train_orchestrator.py            # Train agent (3-4 hours)
3. python demo.py                          # See trained agent (5 min)
4. python evaluate.py --policy dqn --episodes 30       # Test DQN
5. python evaluate.py --policy fixed --episodes 30     # Test baseline
6. python plot_results.py                  # Generate charts (1 min)
```

---

## Modes You'll See

| Mode | When | What Happens |
|------|------|--------------|
| RL MODE | Normal operation | DQN picks best phase adaptively |
| FALLBACK | Confidence <0.35 | Fixed 30-30-5 cycle (safe) |
| GRACE | Emergency vehicle detected | 2s: let mid-crossing vehicles clear |
| PRE_CLEAR | After grace | 2s: all signals RED |
| EMERGENCY | Vehicle in zone | Ambulance arm GREEN, others RED |
| RECOVERY | After emergency | 10s: fixed-time cycle to drain backlog |

---

## Scenarios Running Automatically

- **Every 5+ minutes of demo**: Sensor degradation (→FALLBACK mode)
- **Every 40 seconds**: Emergency vehicle spawns
- **Continuous**: DQN adaptive control (unless in other mode)

---

## If Something Goes Wrong

```bash
# Check CARLA is running
python test_system_logic.py

# Validate all scenarios work
python test_all_scenarios.py

# Check weights file exists
ls data/dqn_weights_int1.json
```

---

**Pro Tip**: Run `train_orchestrator.py` once, then you can run `demo.py` unlimited times with the trained agent.
