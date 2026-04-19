"""
fallback.py
===========
Safety fallback + emergency priority mechanism.

Control mode state machine:

  DQN  ──low-conf──►  FIXED_TIME  ──conf-recovered──►  DQN
   ▲                                                     ▲
   │                                                     │
   └──────────── emergency cleared + recovery done ──────┘
                          │
  DQN / FIXED_TIME  ──emergency-flag──►  EMERGENCY
                          │
                  emergency clears
                          │
                          ▼
                    RECOVERY (fixed-time, 1 full cycle)
                          │
                  recovery ticks expire
                          │
                          ▼
                        DQN

Rules:
  1. Emergency vehicle detected -> EMERGENCY mode immediately (RL paused).
  2. Emergency vehicle clears ROI -> RECOVERY mode (1 fixed cycle to
     rebalance traffic that piled up during green-override).
  3. Recovery ticks expire -> resume DQN.
  4. YOLO confidence drops -> FIXED_TIME (safety fallback).
  5. YOLO confidence recovers -> DQN.

Import this into main.py and demo.py.
"""

from enum import Enum

class ControlMode(Enum):
    DQN        = "DQN"        # normal intelligent control
    FIXED_TIME = "FIXED_TIME" # safety fallback - fixed 30-s phases
    EMERGENCY  = "EMERGENCY"  # emergency vehicle override
    RECOVERY   = "RECOVERY"   # post-emergency rebalancing (fixed-time)


class FallbackController:
    """
    Monitors detection quality and emergency state; switches control modes.

    Usage:
        fb = FallbackController(intersection_id=1)

        mode = fb.update(
            yolo_confidence = 0.42,
            yolo_count      = 5,
            emergency_flag  = 0,
        )
        if   mode == ControlMode.DQN:       action = agent.act(state)
        elif mode == ControlMode.EMERGENCY:  # force GREEN handled in main
        elif mode in (ControlMode.FIXED_TIME,
                      ControlMode.RECOVERY): # fixed timing handles switching
    """

    # ── Confidence thresholds ─────────────────────────────────────────────
    CONF_THRESHOLD    = 0.35   # below this = unreliable detection
    LOW_CONF_WINDOW   = 5      # consecutive low-conf ticks before fallback
    RECOVERY_WINDOW   = 10     # consecutive good-conf ticks before DQN resumes

    # ── Timing ───────────────────────────────────────────────────────────
    FIXED_PHASE_TICKS = 600    # 30 s per phase in ticks (30 / 0.05 s)

    # ── Post-emergency recovery ───────────────────────────────────────────
    # Run exactly 1 full fixed cycle (GREEN->YELLOW->RED) after an emergency
    # clears so that traffic that backed-up during the override can drain
    # before RL takes over again.
    POST_EMERGENCY_RECOVERY_TICKS = 1800   # 3 full fixed phases × 600 ticks

    def __init__(self, intersection_id):
        self.intersection_id  = intersection_id
        self.mode             = ControlMode.DQN

        # Confidence tracking
        self.low_conf_count   = 0
        self.recovery_count   = 0

        # Fixed-time phase tracking (shared by FIXED_TIME and RECOVERY)
        self.phase_tick       = 0
        self.fixed_phase      = 0

        # Post-emergency recovery counter
        self.recovery_ticks   = 0

        # Diagnostics
        self.mode_history     = []
        self.fallback_count   = 0   # FIXED_TIME activations
        self.emergency_count  = 0   # EMERGENCY activations

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, yolo_confidence, yolo_count, emergency_flag):
        """
        Call every simulation tick.

        yolo_confidence : float 0-1  - avg confidence of all detections
        yolo_count      : int        - number of vehicles YOLO detected
        emergency_flag  : int 0|1    - 1 if emergency vehicle in ROI

        Returns: ControlMode
        """

        # ── STEP 1: Emergency takes absolute priority ─────────────────────
        if emergency_flag == 1:
            if self.mode != ControlMode.EMERGENCY:
                self.emergency_count += 1
                print(f"  [Emergency] Int {self.intersection_id}: "
                      f"OVERRIDE - forcing GREEN, RL paused")
            self._set_mode(ControlMode.EMERGENCY)
            return self.mode

        # ── STEP 2: Emergency just cleared -> start RECOVERY phase ─────────
        if self.mode == ControlMode.EMERGENCY:
            self.recovery_ticks = self.POST_EMERGENCY_RECOVERY_TICKS
            self.phase_tick     = 0
            self.fixed_phase    = 0
            self._set_mode(ControlMode.RECOVERY)
            print(f"  [Emergency] Int {self.intersection_id}: "
                  f"Cleared - entering RECOVERY ({self.recovery_ticks} ticks)")
            return self.mode

        # ── STEP 3: Count down RECOVERY ───────────────────────────────────
        if self.mode == ControlMode.RECOVERY:
            self.recovery_ticks -= 1
            if self.recovery_ticks <= 0:
                self._set_mode(ControlMode.DQN)
                print(f"  [Emergency] Int {self.intersection_id}: "
                      f"Recovery complete - resuming DQN")
            return self.mode

        # ── STEP 4: Confidence-based DQN ↔ FIXED_TIME switching ──────────
        effective_conf = yolo_confidence if yolo_count > 0 else 0.0

        if self.mode == ControlMode.DQN:
            if effective_conf < self.CONF_THRESHOLD:
                self.low_conf_count += 1
                self.recovery_count  = 0
                if self.low_conf_count >= self.LOW_CONF_WINDOW:
                    self._set_mode(ControlMode.FIXED_TIME)
                    self.fallback_count += 1
                    self.phase_tick      = 0
                    print(f"  [Fallback] Int {self.intersection_id}: "
                          f"-> FIXED_TIME "
                          f"(conf={effective_conf:.2f} for {self.low_conf_count} ticks)")
            else:
                self.low_conf_count = 0

        elif self.mode == ControlMode.FIXED_TIME:
            if effective_conf >= self.CONF_THRESHOLD:
                self.recovery_count += 1
                if self.recovery_count >= self.RECOVERY_WINDOW:
                    self._set_mode(ControlMode.DQN)
                    print(f"  [Fallback] Int {self.intersection_id}: "
                          f"-> DQN resumed "
                          f"(conf={effective_conf:.2f} recovered)")
            else:
                self.recovery_count = 0

        return self.mode

    def _set_mode(self, new_mode):
        if new_mode != self.mode:
            self.mode_history.append({
                'from': self.mode.value,
                'to':   new_mode.value,
            })
        self.mode = new_mode

    def get_fixed_time_action(self):
        """
        Returns (should_switch_phase, next_phase_idx).
        Call every tick when in FIXED_TIME or RECOVERY mode.
        Cycles through all 4 phases.
        """
        self.phase_tick += 1
        if self.phase_tick >= self.FIXED_PHASE_TICKS:
            self.phase_tick  = 0
            self.fixed_phase = (self.fixed_phase + 1) % 4  # Fixed: was 3, now 4
            return True, self.fixed_phase
        return False, self.fixed_phase

    def get_status(self):
        return {
            'intersection_id' : self.intersection_id,
            'mode'            : self.mode.value,
            'low_conf_count'  : self.low_conf_count,
            'recovery_count'  : self.recovery_count,
            'fallback_count'  : self.fallback_count,
            'emergency_count' : self.emergency_count,
            'recovery_ticks'  : self.recovery_ticks,
        }


# ══════════════════════════════════════════════════════════════
# CONFIDENCE ESTIMATOR
# Extracts average confidence from YOLO results
# ══════════════════════════════════════════════════════════════

VEHICLE_CLASSES = {2, 3, 5, 7}

def get_yolo_confidence(results):
    """
    Extracts average detection confidence from YOLO results.
    Returns (avg_confidence, vehicle_count).
    """
    confs = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                confs.append(float(box.conf[0]))

    if not confs:
        return 0.0, 0

    return sum(confs) / len(confs), len(confs)
