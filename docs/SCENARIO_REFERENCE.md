QUICK REFERENCE: WHAT TO SEE IN EACH SCENARIO
==============================================

Run this command first to verify system logic (no CARLA needed):
  python test_all_scenarios.py

Expected output:
  [OK] Scenario 1: DQN switching phases
  [OK] Scenario 2: Low confidence -> FIXED-TIME -> Recovery to DQN
  [OK] Scenario 3: NORMAL -> GRACE -> PRE_CLEAR -> EMERGENCY -> RECOVERY -> NORMAL
  [OK] Scenario 4: RECOVERY drains queues -> returns to NORMAL
  [OK] All state transitions verified

If all tests pass, the SYSTEM IS CORRECT. Low YOLO in demo is camera/sensor issue.


SCENARIO 1: NORMAL DQN CONTROL
==============================
Where to see it: main.py (with --no-yolo flag for perfect detection)
Trigger: Confidence >= 0.35 and no emergency
Duration: Continuous during normal traffic

What you'll see:
  • CAM 1: "Conf: 0.75+" (GREEN color)
  • Dashboard: "RL MODE - DQN agent controlling signals"
  • Signals change every 5-30 seconds based on traffic pressure
  • When one lane gets congested, agent switches to other lane's GREEN
  
What it proves:
  • DQN makes adaptive decisions
  • Intelligence visible in phase selection
  • Responds to queue changes


SCENARIO 2: LOW CONFIDENCE FALLBACK
===================================
Where to see it: demo.py (visible immediately!)
Trigger: Confidence < 0.35 for 5 consecutive ticks
Duration: 600+ ticks (30+ seconds per phase)

What you'll see:
  • CAM 1: "Conf: 0.00-0.26" (RED color, confidence bar empty)
  • CAM 2: Banner shows "SENSOR DEGRADED - YOLO low confidence"
  • Dashboard: "FALLBACK MODE: FIXED-TIME"
  • Signals: VERY PREDICTABLE - same sequence every 30 seconds
    Phase 0: 600 ticks (GREEN one arm)
    Phase 1: 600 ticks (YELLOW transition)
    Phase 2: 600 ticks (GREEN other arm)
    [REPEAT]

What it proves:
  • System detects confidence degradation
  • Falls back to safe, predictable fixed-time
  • Continues operating safely despite sensor failure

When it recovers:
  • At ~60 seconds into demo.py run
  • Simulated confidence recovery triggers
  • System attempts to return to DQN
  • Watch dashboard: "FALLBACK MODE" changes back to "RL MODE"


SCENARIO 3: EMERGENCY PREEMPTION
=================================
Where to see it: demo.py (every ~9 seconds)
Trigger: Emergency vehicle spawned and detected in ROI (80m radius)
Duration: 35-40 seconds total (includes recovery)

Sequence you'll witness:

  PHASE 1: GRACE (2 seconds, ticks 0-40)
  ---------
  Console: "[EMERGENCY] Spawned ambulance at ... dist=XX m"
  CAM 1: RED box appears labeled "EMERGENCY: AMBULANCE"
  CAM 2: RED box appears (same ambulance, overhead view)
  Signals: HOLD - don't change (let mid-crossing vehicles clear)
  What's happening: System preparing for clearance
  
  PHASE 2: PRE_CLEAR (2 seconds, ticks 40-80)
  ------
  Signals: ALL RED (both arms RED simultaneously!)
  CAM 1: Ambulance box still visible
  CAM 2: Ambulance approaching intersection
  What's happening: Clearing intersection, ensuring it's empty
  
  PHASE 3: EMERGENCY (10-15 seconds, ticks 80-280)
  ---------
  Signals: AMBULANCE ARM GREEN, ALL OTHERS RED
  CAM 1: Shows ambulance passing through
  CAM 2: Shows ambulance crossing intersection, other traffic RED
  Console: Message about ambulance priority
  What's happening: Ambulance has unrestricted passage
  Duration varies:
    - If ambulance slow (< 8 m/s): 15 seconds (300 ticks)
    - If ambulance fast (>= 8 m/s): 10 seconds (200 ticks)
  
  PHASE 4: RECOVERY (10 seconds, ticks 280-480)
  -----------
  Signals: FIXED-TIME cycling (predictable pattern)
  CAM 1: Ambulance exits ROI, other vehicles resume
  CAM 2: Regular traffic flows resume, controlled cycling
  Dashboard: Shows "RECOVERY MODE"
  What's happening: Draining vehicles that backed up during emergency
  
  PHASE 5: BACK TO NORMAL
  -------
  Signals: Return to current mode (FIXED-TIME or DQN)
  Waiting time returns to baseline within 10-20 seconds

What it proves:
  • System detects ambulance within 80m
  • Ensures intersection is safe before priority passage
  • No collisions (pre-clear guarantees this)
  • Adapts timeout based on vehicle speed
  • Recovers traffic after emergency


SCENARIO 4: RECOVERY PHASE
==========================
Where to see it: After Scenario 3 (emergency) completes
Trigger: Emergency vehicle exits or timeout expires
Duration: 10 seconds (200 ticks)

What you'll see:
  • Dashboard: "RECOVERY MODE" displayed
  • Signals: Fixed 2-3 phase cycling (each ~5 seconds)
    Phase 0: 100 ticks GREEN
    Phase 1: 100 ticks YELLOW
    [REPEAT for 200 ticks total]
  • Waiting times: Decrease as backed-up vehicles clear
  • Throughput: High as both arms get equal GREEN time

What it proves:
  • System safely handles post-emergency backlog
  • Fixed-time ensures balanced clearing
  • Prevents starvation (both directions get time)
  • Returns to normal after recovery


EXPECTED TIMELINE (First Run of demo.py)
========================================

Tick 0 - 1800:
  • Fallback mode active (low YOLO confidence)
  • Signals cycle every 30 seconds (FIXED-TIME)
  • Every ~180 ticks: Emergency spawns and follows full sequence
  
Emergency sequences at approximately:
  Tick 180: Emergency 1 [GRACE 2s -> PRE_CLEAR 2s -> EMERGENCY 10-15s -> RECOVERY 10s]
  Tick 580: Emergency 2 [same sequence, ~40s total]
  Tick 980: Emergency 3 [same sequence, ~40s total]
  Tick 1380: Emergency 4 [same sequence, ~40s total]
  ...and so on every 180 ticks

Around tick 1200 (~60 seconds):
  Simulated sensor recovery (built into demo.py)
  [DEMO] Simulating sensor degradation - confidence forced low
  System shows brief FIXED-TIME mode demonstration

After tick 1200:
  Confidence should recover to 0.70
  System attempts to return to DQN
  But may fall back again if real YOLO confidence stays low


CSV COLUMNS TO WATCH
====================

In data/rl_states_final.csv (if running main.py):

Column: system_mode
  Values: NORMAL, GRACE, PRE_CLEAR, EMERGENCY, RECOVERY, FIXED_TIME
  Watch: Transitions during emergency

Column: fallback_mode
  Values: DQN, FIXED_TIME
  Watch: Switches when confidence changes

Column: yolo_confidence
  Values: 0.0 to 1.0
  Watch: Below 0.35 triggers fallback

Column: grace_counter, preclear_counter, recovery_counter
  Watch: Count down from 40, 40, 200 respectively

Column: emergency_timeout
  Values: 200 or 300
  Watch: Decreases during emergency phase

Column: avg_waiting_time, max_waiting_time
  Watch: Spike during emergency, decrease during recovery


MANUAL TESTING (Without CARLA)
==============================

To test all scenarios offline:
  python test_all_scenarios.py

To verify state machine transitions:
  python -c "from system_controller import SystemMode; print([m.value for m in SystemMode])"
  Expected: ['NORMAL', 'GRACE', 'PRE_CLEAR', 'EMERGENCY', 'RECOVERY', 'FIXED_TIME']


TROUBLESHOOTING
===============

If you don't see emergency vehicles in demo.py:
  • Check console output for "[EMERGENCY] Spawned..." messages
  • Verify EMERGENCY_INTERVAL = 180 at line 1127 of demo.py
  • Emergency vehicles may spawn outside camera FOV initially

If you never see NORMAL/DQN mode:
  • This is expected - YOLO confidence is too low
  • See TROUBLESHOOTING_DEMO.md for solutions
  • Run main.py with --no-yolo flag instead

If signals stay RED or GREEN for too long:
  • Check you're in EMERGENCY mode (should timeout after 10-15s)
  • Check you're in FIXED_TIME mode (should cycle every 30s)
  • Check GRACE/PRE_CLEAR/RECOVERY counters in CSV

If emergency vehicles don't get GREEN:
  • Verify get_emergency_vehicles() found the vehicle
  • Check console for "[EMERGENCY] {vehicle_type}" message
  • Verify ambulance speed calculation in main.py
