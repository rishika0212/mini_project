"""
fallback.py
===========
Safety fallback mechanism.

Monitors YOLO detection confidence and switches to
fixed-time control when detection is unreliable.

Rules:
  - If YOLO confidence < CONF_THRESHOLD for WINDOW consecutive frames
    → switch to FIXED_TIME mode
  - If confidence recovers for RECOVERY_WINDOW frames
    → switch back to DQN mode
  - Emergency vehicles always get priority regardless of mode

Import this into main.py and demo.py.
"""

from enum import Enum

class ControlMode(Enum):
    DQN        = "DQN"         # normal intelligent control
    FIXED_TIME = "FIXED_TIME"  # fallback — fixed 30s phases
    EMERGENCY  = "EMERGENCY"   # emergency override

class FallbackController:
    """
    Monitors detection quality and switches control modes.

    Usage:
        fb = FallbackController(intersection_id=1)

        # Every tick, call update with latest YOLO confidence
        mode = fb.update(
            yolo_confidence = 0.3,   # avg confidence of detections
            yolo_count      = 2,     # number of vehicles detected
            emergency_flag  = 0
        )

        if mode == ControlMode.DQN:
            action = agent.act(state)
        elif mode == ControlMode.FIXED_TIME:
            action = 0   # fixed timing handles switching
        elif mode == ControlMode.EMERGENCY:
            # force green
    """

    # Thresholds
    CONF_THRESHOLD    = 0.35   # below this = unreliable detection
    LOW_CONF_WINDOW   = 5      # consecutive low-conf frames before fallback
    RECOVERY_WINDOW   = 10     # consecutive good frames before DQN resumes
    FIXED_PHASE_TICKS = 600    # 30 seconds in ticks (30 / 0.05)

    def __init__(self, intersection_id):
        self.intersection_id  = intersection_id
        self.mode             = ControlMode.DQN
        self.low_conf_count   = 0
        self.recovery_count   = 0
        self.phase_tick       = 0
        self.fixed_phase      = 0   # current phase in fixed-time mode
        self.mode_history     = []  # track mode changes for logging
        self.fallback_count   = 0   # total number of fallback activations

    def update(self, yolo_confidence, yolo_count, emergency_flag):
        """
        Call every tick.
        yolo_confidence: float 0-1 (avg confidence of all detections)
        yolo_count:      int (number of vehicles YOLO detected)
        emergency_flag:  int 0 or 1

        Returns: ControlMode
        """
        # Emergency always takes priority
        if emergency_flag == 1:
            self._set_mode(ControlMode.EMERGENCY)
            return self.mode

        # If no detections at all, treat as low confidence
        effective_conf = yolo_confidence if yolo_count > 0 else 0.0

        if self.mode == ControlMode.DQN:
            if effective_conf < self.CONF_THRESHOLD:
                self.low_conf_count += 1
                self.recovery_count  = 0
                if self.low_conf_count >= self.LOW_CONF_WINDOW:
                    self._set_mode(ControlMode.FIXED_TIME)
                    self.fallback_count += 1
                    print(f"  [Fallback] Int {self.intersection_id}: "
                          f"Switching to FIXED_TIME "
                          f"(conf={effective_conf:.2f} for {self.low_conf_count} frames)")
            else:
                self.low_conf_count = 0

        elif self.mode == ControlMode.FIXED_TIME:
            if effective_conf >= self.CONF_THRESHOLD:
                self.recovery_count += 1
                if self.recovery_count >= self.RECOVERY_WINDOW:
                    self._set_mode(ControlMode.DQN)
                    print(f"  [Fallback] Int {self.intersection_id}: "
                          f"Resuming DQN control "
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
        Call every tick when in FIXED_TIME mode.
        """
        self.phase_tick += 1
        if self.phase_tick >= self.FIXED_PHASE_TICKS:
            self.phase_tick  = 0
            self.fixed_phase = (self.fixed_phase + 1) % 3
            return True, self.fixed_phase
        return False, self.fixed_phase

    def get_status(self):
        return {
            'intersection_id' : self.intersection_id,
            'mode'            : self.mode.value,
            'low_conf_count'  : self.low_conf_count,
            'recovery_count'  : self.recovery_count,
            'fallback_count'  : self.fallback_count,
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
