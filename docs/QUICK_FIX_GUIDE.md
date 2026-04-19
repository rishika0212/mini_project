# Quick Fix Steps for Multiple GREEN Signals Issue

## Problem
Multiple traffic lights turn green simultaneously (e.g., 3-1 split instead of 2-2).

## Root Cause Analysis

Based on the code, the issue could be one of these (in order of likelihood):

1. **Unbalanced light grouping** (3-1 instead of 2-2)
   - Lights not being split evenly

2. **Unaccounted lights** (>4 lights at intersection)
   - Extra lights controlled by CARLA, not our system

3. **Light grouping includes multiple intersections**
   - 100m radius too large, capturing lights from adjacent intersections

4. **set_state() not taking effect**
   - Lights frozen in wrong state
   - CARLA overriding our commands

## What to Do Now

### Step 1: Run Demo with Debugging (5 min)

```bash
python demo.py 2>&1 | tee debug_output.txt
```

This captures everything to `debug_output.txt`.

### Step 2: Look for These Outputs

**GOOD - Startup shows:**
```
Found 4 traffic lights at main intersection
[SORT ORDER] After sorting by diff:
  Position 0: Light 0, diff=-5.2
  Position 1: Light 1, diff=-2.8
  Position 2: Light 2, diff=3.1
  Position 3: Light 3, diff=5.8

✅ BALANCED ASSIGNMENT:
  Group A (NS arm): 2 lights
    A[0] @ (100.1, 200.2)
    A[1] @ (100.5, 199.8)
  Group B (EW arm): 2 lights
    B[0] @ (99.1, 201.0)
    B[1] @ (98.5, 199.5)
✅ All 4 lights accounted for and frozen.
```

**GOOD - Startup test shows:**
```
[STARTUP TEST] Verifying signal control...
[TEST] Forced Phase 0: A=GREEN (2), B=RED (2)
  [SIGNAL VERIFY] Total: 4 | GREEN: 2 | YELLOW: 0 | RED: 2 ✅
```

### Step 3: If There's a Problem

#### Problem A: Unbalanced Group (3-1)
```
Group A (NS arm): 3 lights
Group B (EW arm): 1 light
```

**Fix:** In demo.py, change line 331:
```python
# OLD (line 331):
light_data.sort(key=lambda x: x[4])  # Sort by diff

# NEW - use angle-based sorting instead:
import math
light_data.sort(key=lambda x: math.atan2(x[3], x[2]))
```

---

#### Problem B: Unaccounted Lights
```
⚠️  WARNING: 2 traffic lights at intersection NOT in our control!
   Unaccounted[0] @ (101.5, 202.0)
   is_frozen=False
```

**Fix:** In demo.py, after line 388, add:
```python
# Freeze and add unaccounted lights
for light in unaccounted:
    light.freeze(True)
    group_A.append(light)  # Add to control
    print(f"  Added unaccounted light to group_A")

# Rebalance groups
if len(unaccounted) > 0:
    print(f"  Re-balancing with {len(group_A)} + {len(group_B)} total lights")
    all_lights = group_A + group_B
    mid = len(all_lights) // 2
    group_A = all_lights[:mid]
    group_B = all_lights[mid:]
```

---

#### Problem C: Cluster Spread Wide
```
Light cluster bounds: X=[100.0..250.0] (150.0m wide)
                      Y=[200.0..250.0] (50.0m tall)
⚠️  WARNING: Lights spread over 150.0m × 50.0m
    This might include multiple intersections!
```

**Fix:** In demo.py, change line 307:
```python
# OLD:
if dist < 100:  # Within 100m of intersection center

# NEW - reduce radius:
if dist < 60:  # Within 60m of intersection center
```

---

#### Problem D: set_state() Not Working
```
[STARTUP TEST] Verifying signal control...
[TEST] Forced Phase 0: A=GREEN (2), B=RED (2)
  [SIGNAL VERIFY] Total: 4 | GREEN: 0 | YELLOW: 0 | RED: 4 ❌ VIOLATION
```

The lights aren't changing state even though we're calling set_state().

**Fix:** Try unfreezing before setting state:
```python
def set_phase(phase_idx, debug=False):
    # Try setting state with unfreeze/refreeze
    for l in group_A:
        l.freeze(False)  # Unfreeze
        world.tick()     # Wait one tick
        l.set_state(carla.TrafficLightState.Green)
        world.tick()     # Wait one tick
        l.freeze(True)   # Refreeze
```

---

## Expected Output During Demo Run

Every 500 ticks you should see:
```
[TICK 501] Diagnostic Check:
  [SIGNAL VERIFY] Total: 4 | GREEN: 1 | YELLOW: 0 | RED: 3 ✅
[TICK 1001] Diagnostic Check:
  [SIGNAL VERIFY] Total: 4 | GREEN: 2 | YELLOW: 0 | RED: 2 ✅
```

The GREEN count should vary (1, 2, or 0) depending on the phase, but should NEVER be 3 or 4 at the same time (unless you're seeing the tick right after a phase change).

---

## Quick Checklist

- [ ] Run `python demo.py 2>&1 | tee debug_output.txt`
- [ ] Look for startup diagnostics output
- [ ] Look for "✅ All lights accounted for" or "⚠️ WARNING" messages
- [ ] Look for "[STARTUP TEST]" section
- [ ] Identify which problem (A, B, C, or D) applies
- [ ] Apply the corresponding fix
- [ ] Run again to verify

---

## If Still Stuck

Share the console output from:
```
[STARTUP TEST] ... [SIGNAL VERIFY] ...
```

And also:
```
Found X traffic lights
Group A: N lights
Group B: N lights
⚠️ WARNING ...
✅ All ... lights accounted for
```

This tells us exactly what's being controlled and why the problem is occurring.
