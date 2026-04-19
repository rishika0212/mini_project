# 2-3 Minute Demo Speaking Script

## Goal (10-15 sec)
Good morning. This project is an intelligent traffic signal control system in CARLA using DQN-based adaptive control with emergency vehicle priority and safe fallback logic.

I will quickly show:
1. Normal adaptive signal behavior
2. Emergency handling sequence
3. Evaluation outputs proving improvement

---

## Part 1: What you are seeing on screen (30-40 sec)
This live window is from `demo.py`.

- Left side: intersection overview camera and live signal state
- Right side: dashboard with queue, waiting time, throughput, and mode
- Modes include: RL control, emergency states, and fallback/recovery

In normal traffic, DQN selects which direction should get green based on pressure, which is queue plus waiting trend. So the signal timing is not fixed; it adapts to traffic.

---

## Part 2: Emergency handling (45-60 sec)
When an ambulance is detected, the controller switches to a safe state machine:

1. GRACE: short hold to avoid unsafe sudden switching
2. PRE_CLEAR: all-red interval to clear the intersection
3. EMERGENCY: ambulance approach gets green, others remain red
4. RECOVERY: temporary controlled cycling to drain backlog
5. Return to RL mode

Key safety point: we do not immediately force green. We clear the junction first, then give priority.

---

## Part 3: Why this is better than fixed-time (35-45 sec)
After training, we evaluate with:

- `evaluate.py --policy dqn`
- `evaluate.py --policy fixed`
- `evaluate.py --policy random`

Then we generate comparison plots using `plot_results.py`.

What to highlight in results:
- DQN reduces average waiting time
- DQN reduces queue buildup
- DQN improves throughput
- Emergency flow is handled safely and system returns to normal

So this is both adaptive and safety-aware, unlike pure fixed-time control.

---

## Short technical explanation (20-25 sec)
The DQN receives state features such as per-arm queue/wait context and phase context, then chooses the next phase action.
A reward function encourages lower delay and queue with better flow, while emergency logic is handled by a separate safety state machine.

So learning handles efficiency, and deterministic logic guarantees safety transitions.

---

## Final closing line (10 sec)
In summary, this system demonstrates practical intelligent control: adaptive in normal traffic, safe during emergencies, and supported by measurable evaluation outputs.

Thank you.

---

## Optional Q&A one-liners
- Why not immediate green for ambulance?
Because vehicles may already be crossing; pre-clear avoids collision risk.

- How do you prove improvement?
By policy-wise evaluation CSVs and plots comparing DQN vs fixed-time vs random.

- If detection confidence drops?
System falls back to safe deterministic timing until confidence recovers.
