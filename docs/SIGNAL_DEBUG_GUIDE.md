# Signal Control Debugging Guide

## Problem Statement
Multiple traffic lights turning GREEN simultaneously when only ONE direction should be green.

## Debugging Enhancements Added

### 1. **Initialization Diagnostics** (Lines 302-430)

When `demo.py` starts, it now prints:

```
=== TRAFFIC LIGHT SETUP ===
Found N traffic lights at main intersection (within 100m)

Light cluster bounds: X=[x1..x2] (Xm wide)
                      Y=[y1..y2] (Ym tall)

[SORT ORDER] After sorting by diff:
  Position 0: Light 0, diff=-5.2
  Position 1: Light 1, diff=-2.8
  ...

✅ BALANCED ASSIGNMENT:
  Group A (NS arm): N lights
    A[0] @ (x, y)
    A[1] @ (x, y)
  Group B (EW arm): N lights
    B[0] @ (x, y)
    B[1] @ (x, y)

[CONTROL] Frozen/Manual control: N lights
✅ All N lights accounted for and frozen.
```

**What to look for:**
- ✅ Balanced groups (e.g., 2-2, 3-3, 4-4, NOT 3-1)
- ✅ All lights accounted for (no "Unaccounted" warnings)
- ✅ All lights frozen
- ✅ Cluster bounds < 80m wide/tall (single intersection)

### 2. **Phase Control Logging** (Lines 634-680)

When phases switch, `set_phase()` now logs:

```
[PHASE DEBUG] Phase 0: Set 2 A-lights GREEN, 2 B-lights RED
[SIGNAL VERIFY] Total: 4 | GREEN: 1 | YELLOW: 0 | RED: 3 ✅
```

**What to look for:**
- Set correct counts of lights per group
- Verification shows exactly 1 GREEN (or 1-2 YELLOW during transition)
- If ❌ VIOLATION appears → multiple GREENs detected

### 3. **Continuous Monitoring** (Every 500 ticks)

```
[TICK 501] Diagnostic Check:
  [SIGNAL VERIFY] Total: 4 | GREEN: 1 | YELLOW: 0 | RED: 3 ✅
```

**What to look for:**
- Should see ✅ consistently
- If ❌ VIOLATION appears → captures the exact moment/lights involved

### 4. **Emergency State Transitions**

Signal verification added at all critical points:
- GRACE → PRE_YELLOW (turn yellow)
- PRE_YELLOW → PRE_CLEAR (turn red)
- PRE_CLEAR → EMERGENCY (set ambulance arm green)
- EMERGENCY → RECOVERY (reset to phase 2)
- RECOVERY → DQN (back to normal)

Each transition logs:
```
[PRE_YELLOW] Grace expired — all YELLOW for X ticks.
  [SIGNAL VERIFY] Total: 4 | GREEN: 0 | YELLOW: 4 | RED: 0 ✅
```

---

## Running the Debugging Session

1. **Start the demo:**
   ```bash
   python demo.py
   ```

2. **Let it run for 1-2 minutes** to see:
   - Initialization diagnostics
   - First tick-1 diagnostic check
   - Multiple phase transitions
   - Emergency event (if triggered)

3. **Look at console output for:**
   - ❌ VIOLATION messages (exact light states)
   - ⚠️  WARNING messages (unaccounted lights, cluster spread)
   - Mismatch between set_phase() calls and verify_signal_states() results

---

## Possible Issues & Fixes

### Issue: Unbalanced grouping (e.g., 3-1 split)

**Symptom:**
```
Group A (NS arm): 3 lights
Group B (EW arm): 1 light
```

**Solution:** The diff sorting might not be working as expected. Could happen if:
- Lights are at non-standard angles (not aligned 0°/90°)
- Lights very close together (< 5m apart)

**Fix:** Change line ~331 to use a different split criteria (e.g., angle-based):
```python
# Sort by angle instead of diff
light_data.sort(key=lambda x: math.atan2(x[3], x[2]))
```

---

### Issue: Unaccounted lights detected

**Symptom:**
```
⚠️  WARNING: 2 traffic lights at intersection NOT in our control!
   Unaccounted[0] @ (x, y)
   is_frozen=False
```

**Solution:** These lights are not being frozen, so CARLA controls them automatically.

**Fix:** Freeze ALL lights in the scene, not just those in our groups:
```python
for light in all_lights_at_intersection:
    if light not in lights_accounted:
        light.freeze(True)
        group_A.append(light)  # Add to control, split later
```

---

### Issue: Lights spread over wide area

**Symptom:**
```
Light cluster bounds: X=[100.0..250.0] (150.0m wide)
                      Y=[200.0..250.0] (50.0m tall)
⚠️  WARNING: Lights spread over 150.0m × 50.0m
    This might include multiple intersections!
```

**Solution:** Reduce the 100m radius to capture only main intersection.

**Fix:** Change line ~307 to:
```python
if dist < 60:  # Reduced from 100m to 60m
```

---

### Issue: Multiple GREENs detected

**Symptom:**
```
[SIGNAL VERIFY] Total: 4 | GREEN: 3 | YELLOW: 0 | RED: 1 ❌ VIOLATION: Multiple GREEN detected!
    A[0] @ (x, y): GREEN
    A[1] @ (x, y): GREEN
    B[0] @ (x, y): GREEN
    B[1] @ (x, y): RED
```

**Root causes:**

1. **set_phase() not being called** → lights stuck in previous state
   - Check console: Is `[PHASE DEBUG]` output appearing?
   - If no: verify phases[0] is changing (print it)

2. **set_phase() called but ignored** → lights frozen in wrong state
   - After `set_phase(0)`, wait one world.tick() before verifying
   - Add: `world.tick(); verify_signal_states()`

3. **Lights not responding to set_state()** → frozen but not controlled
   - Unfreeze, set state, re-freeze:
     ```python
     for l in group_A:
         l.freeze(False)
         l.set_state(carla.TrafficLightState.Green)
         l.freeze(True)
     ```

---

## Quick Diagnostic Checklist

Run through this to isolate the issue:

- [ ] Do all lights in group_A have the same state?
- [ ] Do all lights in group_B have the same state?
- [ ] Are group_A and group_B mutually exclusive (one GREEN, other RED)?
- [ ] Do unbalanced groups (3-1) show the problem?
- [ ] Does the problem occur during normal DQN control or only emergency?
- [ ] Does the problem occur at specific ticks or continuously?
- [ ] Are unaccounted lights detected (not in our groups)?

---

## Next Steps

After running demo.py with these diagnostics:

1. **Capture the console output** showing the violation
2. **Note the tick number** when violation occurred
3. **Check which lights are in which group** (positions will help)
4. **Verify group sizes** are balanced
5. **Report findings** with exact output

This will pinpoint the root cause and guide the fix.
