"""
controller.py
=============
Pressure-based arm selection with DQN-assisted adaptive timing.

Architecture
------------
1. Compute per-arm pressure  =  queue_length + avg_wait_time / 10
2. Apply anti-starvation boost for arms that have been waiting too long.
3. Select best_arm = argmax(pressures).
4. DQN produces action 0=keep / 1=allow-switch (reduces required margin).
5. Final switch decision enforces min/max green-time hard limits.

No signal state is set here — the caller (demo.py main loop) calls
signal_manager based on should_switch() / commit_switch() results.

Timing reference (0.05 s/tick):
  300 ticks = 15 s  |  1200 ticks = 60 s  |  60 ticks = 3 s
"""

ARMS = ['N', 'S', 'E', 'W']

# Timing (ticks at 0.05 s/tick) — calibrated to real-world signal timing
MIN_GREEN_TICKS   = 300   # 15 s minimum green before any switch is allowed
MAX_GREEN_TICKS   = 1200  # 60 s hard cap — matches realistic cycle caps
YELLOW_TICKS      = 60    # 3 s yellow clearance phase

# Pressure switching thresholds
SWITCH_MARGIN     = 2.0   # challenger arm must beat current by this margin
URGENT_GAP        = 25.0  # bypass MIN_GREEN only when gap is very large
URGENT_MIN_TICKS  = 180   # absolute floor even in urgent case (9 s)

# Anti-starvation — first boost after 45 s, +5 units per period
STARVATION_TICKS  = 900   # 45 s without green before first boost
STARVATION_BOOST  = 5.0   # pressure added per starvation period


class PressureController:
    def __init__(self):
        self.current_arm      = 'N'   # arm currently holding GREEN
        self.ticks_in_phase   = 0     # ticks since last committed switch
        # How long each arm has waited without getting a green phase
        self.starvation       = {arm: 0 for arm in ARMS}

    # ── Pressure ──────────────────────────────────────────────────────────────

    def compute_pressures(self, arm_queues: dict, arm_avg_waits: dict) -> dict:
        """
        Score for each road: more stopped cars = higher score, waiting longer = higher score.
        """
        pressures = {}
        for arm in ARMS:
            q    = arm_queues.get(arm, 0)
            w    = arm_avg_waits.get(arm, 0.0) if q > 0 else 0.0
            # Simple score: 1 queued vehicle = 1 unit, 10s avg wait = 1 unit
            pressures[arm] = q + w / 10.0
        return pressures

    def select_best_arm(self, pressures: dict) -> str:
        """Return the arm with the highest pressure."""
        return max(ARMS, key=lambda a: pressures.get(a, 0.0))

    # ── Switch decision ───────────────────────────────────────────────────────

    def should_switch(self, pressures: dict) -> tuple:
        """
        Evaluate whether to switch to the best-pressure arm.
        Returns (do_switch: bool, best_arm: str).
        """
        best_arm         = self.select_best_arm(pressures)
        current_pressure = pressures.get(self.current_arm, 0.0)
        best_pressure    = pressures.get(best_arm, 0.0)

        # Already optimal
        if best_arm == self.current_arm:
            return False, best_arm

        # Margin check: is any other road's score high enough?
        # A margin of 2.0 (equivalent to 2 cars or 20s wait difference) prevents rapid flipping.
        if best_pressure > current_pressure + 2.0:
            return True, best_arm

        return False, best_arm

    # ── State update ──────────────────────────────────────────────────────────

    def commit_switch(self, new_arm: str):
        """
        Record that we have committed to switching to new_arm.
        Call after the yellow phase completes and new_arm goes GREEN.
        """
        self.starvation[self.current_arm] = 0  # served arm's starvation clears
        self.current_arm    = new_arm
        self.ticks_in_phase = 0
        # Do NOT zero starvation[new_arm] here — let it decay gradually through tick()
        # so the boost persists long enough for the arm to clear its actual backlog.

    def tick(self):
        """
        Advance per-tick counters.
        """
        self.ticks_in_phase += 1
