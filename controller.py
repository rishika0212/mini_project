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

# Timing (ticks at 0.1 s/tick) — faster simulation speed
MIN_GREEN_TICKS   = 150   # 15 s minimum green before any switch is allowed
MAX_GREEN_TICKS   = 250   # 25 s hard cap
YELLOW_TICKS      = 30    # 3 s yellow clearance phase

# Pressure switching thresholds
SWITCH_MARGIN     = 2.0   # challenger arm must beat current by this margin
URGENT_GAP        = 25.0  # bypass MIN_GREEN only when gap is very large
URGENT_MIN_TICKS  = 90    # absolute floor even in urgent case (9 s)

# Anti-starvation — first boost after 45 s, +5 units per period
STARVATION_TICKS  = 450   # 45 s without green before first boost
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
        Includes anti-starvation boost.
        """
        pressures = {}
        for arm in ARMS:
            q    = arm_queues.get(arm, 0)
            w    = arm_avg_waits.get(arm, 0.0) if q > 0 else 0.0
            
            # Starvation boost: add units for every period of waiting without green
            boost = (self.starvation[arm] // STARVATION_TICKS) * STARVATION_BOOST
            
            # Simple score: 1 queued vehicle = 1 unit, 10s avg wait = 1 unit
            pressures[arm] = q + w / 10.0 + boost
        return pressures

    def select_best_arm(self, pressures: dict) -> str:
        """Return the arm with the highest pressure."""
        return max(ARMS, key=lambda a: pressures.get(a, 0.0))

    # ── Switch decision ───────────────────────────────────────────────────────

    def should_switch(self, pressures: dict, dqn_action: int = 1) -> tuple:
        """
        Evaluate whether to switch to the best-pressure arm.
        Returns (do_switch: bool, best_arm: str).
        """
        best_arm         = self.select_best_arm(pressures)
        current_pressure = pressures.get(self.current_arm, 0.0)
        best_pressure    = pressures.get(best_arm, 0.0)

        # 1. Already serving the highest-pressure arm
        if best_arm == self.current_arm:
            return False, best_arm

        # 2. Hard constraint: MAX_GREEN_TICKS reached — must switch
        if self.ticks_in_phase >= MAX_GREEN_TICKS:
            return True, best_arm

        # 3. DQN Veto: action 0 means 'keep current'
        if dqn_action == 0:
            return False, best_arm

        # 4. Safety: MIN_GREEN_TICKS floor (bypass only for huge URGENT_GAP)
        gap = best_pressure - current_pressure
        if self.ticks_in_phase < MIN_GREEN_TICKS:
            if gap > URGENT_GAP and self.ticks_in_phase >= URGENT_MIN_TICKS:
                return True, best_arm
            return False, best_arm

        # 5. Margin check: prevent rapid flipping
        if best_pressure > current_pressure + SWITCH_MARGIN:
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

    def tick(self, arm_queues: dict = None):
        """
        Advance per-tick counters. 
        Only increment starvation for arms that actually have vehicles waiting.
        """
        self.ticks_in_phase += 1
        for arm in ARMS:
            if arm == self.current_arm:
                self.starvation[arm] = 0
            elif arm_queues and arm_queues.get(arm, 0) > 0:
                self.starvation[arm] += 1
            else:
                # If arm is empty, starvation shouldn't build up
                self.starvation[arm] = 0
