# Traffic Intelligence System for CARLA

**Hybrid Architecture**: Realistic ground sensors backend + overhead camera visualization

## 🚀 Quick Start

```bash
# Terminal 1: Start CARLA
CarlaUE4.exe /Game/Carla/Maps/Town03

# Terminal 2: Run demo
python demo.py
```

**5 minutes later** → See overhead camera + emergency RED boxes + DQN control

---

## 📁 Project Structure

```
.
├── *.py                          # Python scripts (14 files)
│   ├── demo.py                   # Interactive demo (overhead camera)
│   ├── main.py                   # Full system with ground sensors
│   ├── train_orchestrator.py     # Auto 2-phase training
│   ├── evaluate.py               # Compare policies (DQN vs Fixed vs Random)
│   ├── plot_results.py           # Generate comparison charts
│   ├── dqn_agent.py              # Intelligent control
│   ├── ground_sensors.py         # Realistic per-arm detection
│   ├── fallback.py               # Confidence-based fallback
│   ├── waiting_time.py           # Queue tracking
│   ├── test_all_scenarios.py     # Validate system
│   └── [others]
│
├── data/                         # Training & results (auto-created)
│   ├── dqn_weights_int1.json     # Trained agent weights
│   ├── eval_*.csv                # Evaluation metrics
│   └── plots/                    # Generated charts
│
└── docs/                         # ALL INSTRUCTIONS (organized here!)
    ├── README.md                 # Main documentation
    ├── COMMANDS.md               # Copy-paste commands
    ├── QUICK_START.md            # 5-minute guide
    ├── WORKFLOW_GUIDE.md         # Detailed workflows
    ├── HYBRID_SYSTEM_COMPLETE.md # Architecture overview
    ├── TRAINING_GUIDE.md         # Training instructions
    ├── TROUBLESHOOTING_DEMO.md   # If something breaks
    ├── VIVA_REFERENCE.md         # Presentation notes
    └── [11 other guides]
```

---

## 📚 Documentation Guide

Start with these (in order):

1. **`docs/COMMANDS.md`** ← Quick copy-paste reference
2. **`docs/QUICK_START.md`** ← 5-minute getting started
3. **`docs/WORKFLOW_GUIDE.md`** ← Detailed instructions for each command
4. **`docs/HYBRID_SYSTEM_COMPLETE.md`** ← Architecture explanation

For specific needs:
- **Training?** → `docs/TRAINING_GUIDE.md`
- **Issues?** → `docs/TROUBLESHOOTING_DEMO.md`
- **Presentation?** → `docs/VIVA_REFERENCE.md`
- **Full project story?** → `docs/PROJECT_END_TO_END_NARRATIVE.md`

---

## 🎯 Most Used Commands

```bash
# See it working (5 min)
python demo.py

# Train agent (3-4 hours)
python train_orchestrator.py

# Evaluate & compare (45 min)
python evaluate.py --policy dqn --episodes 30
python evaluate.py --policy fixed --episodes 30
python plot_results.py

# Validate system
python test_all_scenarios.py
```

---

## ✨ Key Features

✅ **Realistic Ground Sensors** - Per-arm vehicle detection (N/S/E/W)
✅ **Overhead Camera** - Clean bird's-eye view visualization
✅ **Emergency Vehicles** - RED boxes on camera for ambulances
✅ **Adaptive Control** - DQN learns optimal phase timing
✅ **Safe Fallback** - Fixed-time mode when confidence drops
✅ **Multiple Policies** - Compare DQN vs Fixed vs Random
✅ **Full Logging** - CSV metrics + console output + visualization

---

## 🔄 Typical Workflow

```
demo.py (5 min)
    ↓
train_orchestrator.py (3-4 hours)
    ↓
evaluate.py --policy dqn (15 min)
evaluate.py --policy fixed (15 min)
    ↓
plot_results.py (1 min)
    ↓
View comparison charts in data/plots/
```

---

## 📊 System Modes

| Mode | Triggered | Behavior |
|------|-----------|----------|
| **RL (DQN)** | Normal | Intelligent adaptive control |
| **FALLBACK** | Conf < 0.35 | Safe fixed 30-30-5 cycling |
| **EMERGENCY** | Ambulance | Priority GREEN for ambulance arm |
| **RECOVERY** | Post-emergency | Fixed-time drainage (10s) |

---

## 🎓 Next Steps

1. **Read**: `docs/QUICK_START.md`
2. **Run**: `python demo.py`
3. **Train**: `python train_orchestrator.py`
4. **Evaluate**: `python evaluate.py --policy dqn --episodes 30`
5. **Plot**: `python plot_results.py`

---

## 📞 Need Help?

- **Getting started?** → `docs/QUICK_START.md`
- **Commands?** → `docs/COMMANDS.md`
- **Workflows?** → `docs/WORKFLOW_GUIDE.md`
- **Something broken?** → `docs/TROUBLESHOOTING_DEMO.md`
- **Signal control issues?** → `docs/SIGNAL_DEBUG_GUIDE.md` + `docs/QUICK_FIX_GUIDE.md`
- **Want details?** → `docs/HYBRID_SYSTEM_COMPLETE.md` + `docs/SIGNAL_CONTROL_DETAILED.md`

All guides are in the `docs/` folder for easy organization!

---

**Status**: ✅ **READY TO USE**

All instructions organized in `docs/` folder. Start with `docs/COMMANDS.md` for copy-paste commands.
