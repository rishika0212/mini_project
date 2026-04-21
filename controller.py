"""
controller.py
=============
Pressure-based arm selection with DQN-assisted adaptive timing.

Architecture
------------
1. Compute per-arm pressure  =  queue_length + 1.5 * avg_wait_time
2. Apply anti-starvation boost for arms that have been waiting too long.
3. Select best_arm = argmax(pressures).
4. DQN produces action 0=keep / 1=allow-switch (reduces required margin).
5. Final switch decision enforces min/max green-time hard limits.

No signal state is set here — the caller (demo.py main loop) calls
signal_manager based on should_switch() / commit_switch() results.
"""

ARMS = ['N', 'S', 'E', 'W']

# Timing (ticks at 0.05 s/tick)
MIN_GREEN_TICKS   = 80    # 4 s minimum green before any switch allowed
MAX_GREEN_TICKS   = 320   # 16 s hard cap — force switch if exceeded
YELLOW_TICKS      = 40    # 2 s yellow transition

# Pressure switching thresholds
SWITCH_MARGIN     = 2.0   # new arm must exceed current by at least this
RL_MARGIN_FACTOR  = 0.70  # DQN action=1 relaxes margin to 70 %
URGENT_GAP        = 12.0  # bypass MIN_GREEN (down to URGENT_MIN) when gap this large
URGENT_MIN_TICKS  = 30    # absolute minimum green even in urgent case (~1.5 s)

# Anti-starvation
STARVATION_TICKS  = 150   # ticks without green before boost applies
STARVATION_BOOST  = 8.0   # pressure added per starvation period


class PressureController:
    def __init__(self):
        self.current_arm      = 'N'   # arm currently holding GREEN
        self.ticks_in_phase   = 0     # ticks since last committed switch
        # How long each arm has waited without getting a green phase
        self.starvation       = {arm: 0 for arm in ARMS}

    # ── Pressure ──────────────────────────────────────────────────────────────

    def compute_pressures(self, arm_queues: dict, arm_avg_waits: dict) -> dict:
        """
        pressure[arm] = queue_length + 1.5 * avg_wait_time + starvation_boost

        avg_wait_time is only included when there are queued vehicles so stale
        wait values from cleared arms don't produce phantom pressure.
        """
        pressures = {}
        for arm in ARMS:
            q    = arm_queues.get(arm, 0)
            w    = arm_avg_waits.get(arm, 0.0) if q > 0 else 0.0
            base = q + 1.5 * w
            periods = self.starvation[arm] // STARVATION_TICKS
            boost   = periods * STARVATION_BOOST if periods > 0 else 0.0
            pressures[arm] = min(base + boost, 60.0)
        return pressures

    def select_best_arm(self, pressures: dict) -> str:
        """Return the arm with the highest pressure."""
        return max(ARMS, key=lambda a: pressures.get(a, 0.0))

    # ── Switch decision ───────────────────────────────────────────────────────

    def should_switch(self, pressures: dict, rl_action: int) -> tuple:
        """
        Evaluate whether to switch to the best-pressure arm.

        Returns (do_switch: bool, best_arm: str).

        Rules (in order):
          1. Already on the best arm → no switch.
          2. Below MIN_GREEN_TICKS → no switch (hard minimum).
          3. Above MAX_GREEN_TICKS and another arm has demand → force switch.
          4. best_arm pressure exceeds current by margin
             (margin lowered by RL if rl_action == 1) → switch.
        """
        best_arm         = self.select_best_arm(pressures)
        current_pressure = pressures.get(self.current_arm, 0.0)
        best_pressure    = pressures.get(best_arm, 0.0)

        # Already optimal
        if best_arm == self.current_arm:
            return False, best_arm

        # Hard minimum green time — bypass down to URGENT_MIN when gap is very large
        if self.ticks_in_phase < MIN_GREEN_TICKS:
            gap = best_pressure - current_pressure
            if gap < URGENT_GAP or self.ticks_in_phase < URGENT_MIN_TICKS:
                return False, best_arm

        # Hard maximum green time — force switch if other arm has any demand
        if self.ticks_in_phase >= MAX_GREEN_TICKS and best_pressure > 0.5:
            return True, best_arm

        # RL-assisted threshold: action=1 allows switching with a smaller gap
        margin = SWITCH_MARGIN * (RL_MARGIN_FACTOR if rl_action == 1 else 1.0)
        if best_pressure > current_pressure + margin:
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
        Call once per simulation tick when in DQN or FALLBACK mode.
        """
        self.ticks_in_phase += 1
        for arm in ARMS:
            if arm != self.current_arm:
                self.starvation[arm] += 1
            else:
                # Decay starvation at 3× the accumulation rate so the boost earned
                # during a long wait persists ~1/3 as long while serving, giving the
                # arm time to drain its backlog before the boost vanishes.
                self.starvation[arm] = max(0, self.starvation[arm] - 3)

    # ── DQN state vector ─────────────────────────────────────────────────────

    def get_state_vector(self, pressures: dict, emergency_flag: int) -> list:
        """
        Build the 7-element normalised state vector for the DQN.

        Features:
          [0-3] pN, pS, pE, pW  — relative pressures (max-normalised, 0-1)
          [4]   current_arm      — index 0-3 normalised by 3
          [5]   time_in_phase    — normalised by MAX_GREEN_TICKS
          [6]   emergency_flag   — 0 or 1
        """
        pN = pressures.get('N', 0.0)
        pS = pressures.get('S', 0.0)
        pE = pressures.get('E', 0.0)
        pW = pressures.get('W', 0.0)
        max_p = max(pN, pS, pE, pW, 1.0)   # avoid divide-by-zero

        arm_idx = ARMS.index(self.current_arm) if self.current_arm in ARMS else 0
        return [
            pN / max_p,
            pS / max_p,
            pE / max_p,
            pW / max_p,
            arm_idx / 3.0,
            min(self.ticks_in_phase, MAX_GREEN_TICKS) / MAX_GREEN_TICKS,
            float(emergency_flag),
        ]
