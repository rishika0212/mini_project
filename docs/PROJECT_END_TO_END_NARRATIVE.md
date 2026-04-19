# Traffic Intelligence System: End-to-End Narrative

## Purpose

This project implements an intelligent traffic signal control system in CARLA with a practical, real-world-inspired design philosophy. The core objective is to reduce waiting time and queue buildup while maintaining strict safety constraints and prioritizing emergency vehicles. The system combines adaptive reinforcement learning with deterministic signal safety logic, fallback behavior, and rich visualization for demonstration and explanation.

The project evolved from a vision-first prototype toward a deployment-style architecture. In the early stage, camera detection using YOLO was a major decision input. In the current direction, control relies on structured per-arm traffic state (N/S/E/W), which is closer to how real intersections are instrumented. The camera and YOLO pipeline remain highly useful for visualization, confidence monitoring, and communication during demos.

## End-to-End System Flow

Each simulation tick follows a full control loop:

1. Traffic state is observed near each intersection.
2. State features are built per arm (queue, wait, phase context, emergency flag).
3. The controller selects operating mode (DQN, fallback, emergency, recovery).
4. If in normal mode, the DQN agent proposes the next phase.
5. Signal changes are applied with safe transitions (green -> yellow -> all-red -> green).
6. Waiting, queue, throughput, and diagnostic metrics are logged for analysis.

This creates a complete perception -> decision -> control -> measurement pipeline that can be trained, evaluated, and demonstrated repeatedly.

## Why the Ground Sensor Direction Matters

A major architectural decision in this project is the shift toward ground-sensor-like real-time inputs for control. This is important because real urban signal systems are usually driven by lane loops, stop-line detectors, or arm-level counts, not only by camera object detection. Arm-wise counts are more stable for control logic, easier to map to signal phases, and less sensitive to visual issues like occlusion and perspective.

In this project, the sensing direction is represented through per-arm traffic estimation and dedicated sensor abstractions. The visualization stack still uses overhead camera views so that system behavior is easy to understand in demos and presentations.

## Signal Control Logic at the Intersection

The signal controller follows a conflict-safe policy. For a 4-way intersection, only one movement corridor receives right-of-way at a time while conflicting movements remain red. Depending on how CARLA traffic-light heads are grouped, this can appear as one light head green and others red, or as grouped heads for one direction green while the cross direction remains red. In both cases, the control meaning is the same: one non-conflicting movement set proceeds, all conflicting sets are blocked.

The DQN agent does not directly toggle lamp colors every tick. Instead, it selects the next target phase. The controller enforces hard safety timing between phases using yellow and all-red clearance intervals. This prevents unsafe instant switching and maintains realistic signal operation.

## DQN-Based Adaptive Control

The DQN agent uses a normalized state vector containing arm-wise queue and waiting features, current phase, elapsed phase time, time context, and emergency flag. Based on this state, it predicts action values for available phases and selects the next phase according to policy mode (exploration during training, greedy during evaluation).

The reward function combines multiple objectives: lower average waiting time, lower queue length, higher throughput, and better flow speed, while discouraging unnecessary switching. This encourages stable adaptive timing that reacts to pressure asymmetry across arms instead of using fixed cyclic timing.

## Emergency Vehicle Handling

Emergency handling is implemented as a dedicated state machine rather than a simple override command. The stages are:

1. **GRACE**: brief hold to let mid-crossing vehicles clear.
2. **PRE_CLEAR**: all-red interval to empty the junction safely.
3. **EMERGENCY**: priority green for the emergency approach arm, others red.
4. **RECOVERY**: temporary fixed-time drainage to clear post-emergency backlog.
5. Return to normal adaptive control.

This sequence is designed to maximize emergency priority without introducing intersection conflicts. It is also presentation-ready because each stage has a clear operational purpose and safety justification.

## Fallback Safety Behavior

The system includes confidence-based fallback logic. If detection reliability remains below threshold for a configured window, control shifts from adaptive mode to deterministic fixed-time operation. This guarantees predictable behavior under uncertain perception. Once confidence recovers consistently, control transitions back to adaptive mode.

This hybrid policy is a practical engineering safeguard: use intelligence when confidence is reliable, use deterministic safety when it is not.

## Evaluation and Evidence Pipeline

The project is designed to produce measurable evidence, not only visual results. It includes:

- Training and inference workflows for DQN agents.
- Policy comparison workflows (DQN vs fixed-time and other baselines).
- CSV logging for waiting time, queue length, throughput, phase behavior, and mode transitions.
- Plot generation for report and viva discussion.

This supports both technical validation and presentation quality. The same architecture can be explained from algorithm level (state, reward, policy), system level (modes and transitions), and operations level (metrics and reproducibility).

## Project Summary

This project demonstrates a complete intelligent traffic system that balances adaptation and safety. It combines:

- Real-time intersection state estimation,
- DQN-based adaptive phase selection,
- Hard signal safety transitions,
- Emergency preemption with staged clearance,
- Confidence-based fallback,
- Visualization and diagnostics for explainability.

The key outcome is a realistic end-to-end traffic control workflow: practical sensing assumptions, safe control guarantees, adaptive intelligence, and measurable performance.
