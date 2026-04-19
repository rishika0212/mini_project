"""
system_controller.py
====================
Centralized Traffic Intelligence System Controller

Manages all 4 layers:
  1. Perception: YOLO detection + confidence
  2. State: Vehicle queue, waiting time, speed
  3. Decision: DQN agent + FallbackController (confidence-based fallback)
  4. Control: 4-phase signal control with emergency preemption

State Machine (per-intersection):
  NORMAL ──emergency──> GRACE ──grace_expire──> PRE_CLEAR ──clear_expire──> EMERGENCY
    ▲                                                                            │
    │                                                                            ▼
    └─────────────────────────────────── RECOVERY (10s) <──────────────────────┘

    │
    └──low_confidence──> FIXED_TIME ──confidence_recover──> NORMAL
       (5 ticks at <0.35)                      (10 ticks at ≥0.35)

Transitions are atomic and mutually exclusive. Emergency takes priority over all others.
"""

import numpy as np
from enum import Enum
from fallback import FallbackController, ControlMode


class SystemMode(Enum):
    """System control modes (per-intersection)."""
    NORMAL = "NORMAL"           # DQN control, normal traffic
    GRACE = "GRACE"             # Emergency detected, hold signals (2s)
    PRE_CLEAR = "PRE_CLEAR"     # All RED to clear intersection (2s)
    EMERGENCY = "EMERGENCY"     # Emergency vehicle GREEN override (10-15s)
    RECOVERY = "RECOVERY"       # Fixed-time cycling post-emergency (10s)
    FIXED_TIME = "FIXED_TIME"   # Fallback: low confidence YOLO (30s phases)


class TrafficSystemController:
    """
    Unified system controller for one intersection.

    Encapsulates:
    - Emergency state machine (NORMAL/GRACE/PRE_CLEAR/EMERGENCY/RECOVERY)
    - Fallback confidence switching (DQN ↔ FIXED_TIME)
    - Signal control logic
    - Diagnostic tracking
    """

    def __init__(self, intersection_id, dqn_agent, num_phases=4):
        """
        Parameters
        ----------
        intersection_id : int
            Unique identifier (1, 2, 3, ...)
        dqn_agent : DQNAgent
            Pre-trained or initialized DQN agent
        num_phases : int
            Number of signal phases (4 for main.py, 3 for demo.py)
        """
        self.intersection_id = intersection_id
        self.dqn_agent = dqn_agent
        self.num_phases = num_phases

        # ── Main state machine ─────────────────────────────────────────
        self.system_mode = SystemMode.NORMAL
        self.pending_mode = SystemMode.NORMAL

        # ── Fallback controller (confidence-based switching) ──────────
        self.fallback = FallbackController(intersection_id)

        # ── Emergency state variables ──────────────────────────────────
        self.grace_counter = 0
        self.preclear_counter = 0
        self.recovery_counter = 0
        self.emergency_timeout = 0

        # ── Timing constants (hardcoded, non-tunable) ──────────────────
        self.GRACE_TICKS = 40         # 2 seconds
        self.PRECLEAR_TICKS = 40      # 2 seconds
        self.RECOVERY_TICKS = 200     # 10 seconds
        self.MIN_EMERGENCY_GREEN = 200  # 10 seconds minimum
        self.MAX_EMERGENCY_GREEN = 300  # 15 seconds maximum (slow vehicles)

        # ── Signal state (for hard constraints) ───────────────────────
        self.trans_state = 'GREEN'    # GREEN | YELLOW | ALL_RED
        self.trans_counter = 0
        self.pending_phase = 0
        self.active_phase = 0
        self.phase_counter = 0
        self.phase_counter_internal = 0   # ticks in current fixed-time/recovery phase

        # ── Diagnostic tracking ────────────────────────────────────────
        self.mode_history = []
        self.transition_count = 0
        self.yolo_confidence = 0.0
        self.yolo_count = 0
        self.last_mode = None

    # ────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────

    def update(self, state, yolo_confidence, yolo_count, emergency_flag, emg_vehicle=None):
        """
        Update system state and compute next action.

        Called every simulation tick.

        Parameters
        ----------
        state : np.ndarray
            12-element state vector from build_state()
        yolo_confidence : float
            Average YOLO detection confidence [0, 1]
        yolo_count : int
            Number of vehicles detected by YOLO
        emergency_flag : int
            1 if emergency vehicle in ROI, else 0
        emg_vehicle : carla.Vehicle or None
            Emergency vehicle object (for signal control)

        Returns
        -------
        action : int
            Next phase (0-3 for 4-phase, 0-2 for 3-phase)
        mode : SystemMode
            Current control mode
        signal_override : dict or None
            Signal control override (for EMERGENCY/PRE_CLEAR states)
        """
        # Store for diagnostics
        self.yolo_confidence = yolo_confidence
        self.yolo_count = yolo_count

        # ── STEP 1: Update fallback controller (confidence checking) ───
        fallback_mode = self.fallback.update(yolo_confidence, yolo_count, emergency_flag)

        # ── STEP 2: Route to appropriate handler ────────────────────────
        if self.system_mode == SystemMode.NORMAL:
            action, override = self._handle_normal(state, emergency_flag, fallback_mode)
        elif self.system_mode == SystemMode.GRACE:
            action, override = self._handle_grace(emergency_flag, fallback_mode)
        elif self.system_mode == SystemMode.PRE_CLEAR:
            action, override = self._handle_preclear(emergency_flag, fallback_mode)
        elif self.system_mode == SystemMode.EMERGENCY:
            action, override = self._handle_emergency(emergency_flag, emg_vehicle)
        elif self.system_mode == SystemMode.RECOVERY:
            action, override = self._handle_recovery(fallback_mode)
        elif self.system_mode == SystemMode.FIXED_TIME:
            action, override = self._handle_fixed_time(fallback_mode)
        else:
            # Fallback to NORMAL
            action, override = 0, None
            self.system_mode = SystemMode.NORMAL

        # ── STEP 3: Track mode transitions for diagnostics ──────────────
        if self.system_mode != self.last_mode:
            self.mode_history.append({
                'from': self.last_mode.value if self.last_mode else 'START',
                'to': self.system_mode.value,
                'tick': self.transition_count
            })
            self.last_mode = self.system_mode

        self.transition_count += 1

        return action, self.system_mode, override

    def get_status(self):
        """Return diagnostic status dictionary."""
        return {
            'intersection_id': self.intersection_id,
            'mode': self.system_mode.value,
            'fallback_mode': self.fallback.mode.value,
            'yolo_confidence': self.yolo_confidence,
            'yolo_count': self.yolo_count,
            'grace_counter': self.grace_counter,
            'preclear_counter': self.preclear_counter,
            'recovery_counter': self.recovery_counter,
            'emergency_timeout': self.emergency_timeout,
            'mode_history': self.mode_history[-10:],  # Last 10 transitions
            'transition_count': self.transition_count,
        }

    # ────────────────────────────────────────────────────────────────
    # PRIVATE: STATE HANDLERS
    # ────────────────────────────────────────────────────────────────

    def _handle_normal(self, state, emergency_flag, fallback_mode):
        """
        Normal operation: DQN or FIXED_TIME control.

        Returns
        -------
        action : int
            Next phase from DQN or fixed-time
        override : dict or None
            None (no signal override needed)
        """
        if emergency_flag == 1:
            # Emergency detected: transition to GRACE
            self.system_mode = SystemMode.GRACE
            self.grace_counter = self.GRACE_TICKS
            return self.active_phase, None

        # Check fallback mode (confidence-based)
        if fallback_mode == ControlMode.FIXED_TIME:
            # Low confidence: switch to FIXED_TIME fallback
            self.system_mode = SystemMode.FIXED_TIME
            self.fallback.phase_tick = 0
            self.fallback.fixed_phase = 0
            return 0, None

        # Normal DQN control
        action = self.dqn_agent.act(state)
        return action, None

    def _handle_grace(self, emergency_flag, fallback_mode):
        """
        GRACE period: Hold signals, let mid-crossing vehicles clear.

        Returns
        -------
        action : int
            Current phase (unchanged)
        override : dict or None
            None (no signal change)
        """
        self.grace_counter -= 1

        if self.grace_counter <= 0:
            if emergency_flag == 1:
                # Emergency still present: transition to PRE_CLEAR
                self.system_mode = SystemMode.PRE_CLEAR
                self.preclear_counter = self.PRECLEAR_TICKS
                return self.active_phase, {'set_all_red': True}
            else:
                # Vehicle cleared: back to NORMAL
                self.system_mode = SystemMode.NORMAL
                return self.active_phase, None

        # Hold current phase
        return self.active_phase, None

    def _handle_preclear(self, emergency_flag, fallback_mode):
        """
        PRE_CLEAR period: All RED to ensure intersection empty.

        Returns
        -------
        action : int
            Current phase (all-red state)
        override : dict or None
            {'set_all_red': True}
        """
        self.preclear_counter -= 1

        if self.preclear_counter <= 0:
            if emergency_flag == 1:
                # Pre-clearance complete, emergency still present
                self.system_mode = SystemMode.EMERGENCY

                # Adaptive timeout based on speed
                # Default: MIN_EMERGENCY_GREEN (200 ticks = 10s)
                # Will be set based on vehicle speed in main.py if needed
                self.emergency_timeout = self.MIN_EMERGENCY_GREEN

                return self.active_phase, {'set_emergency_phase': True}
            else:
                # Vehicle cleared during pre-clearance: back to NORMAL
                self.system_mode = SystemMode.NORMAL
                return self.active_phase, None

        # Keep all RED
        return self.active_phase, {'set_all_red': True}

    def _handle_emergency(self, emergency_flag, emg_vehicle):
        """
        EMERGENCY: Give priority to emergency vehicle.

        Returns
        -------
        action : int
            Emergency phase (0 or 2, depending on vehicle location)
        override : dict or None
            {'set_emergency_phase': True, 'vehicle': emg_vehicle}
        """
        self.emergency_timeout -= 1

        if emergency_flag == 0 or self.emergency_timeout <= 0:
            # Emergency cleared or timeout: transition to RECOVERY
            self.system_mode = SystemMode.RECOVERY
            self.recovery_counter = self.RECOVERY_TICKS
            self.fallback.phase_tick = 0
            self.fallback.fixed_phase = 0
            return self.active_phase, None

        # Keep emergency priority
        return self.active_phase, {'set_emergency_phase': True, 'vehicle': emg_vehicle}

    def _handle_recovery(self, fallback_mode):
        """
        RECOVERY: Fixed-time cycling to drain backed-up traffic.

        Cycles through all num_phases phases at ~5 s each (100 ticks),
        then returns to NORMAL DQN control.
        """
        self.recovery_counter -= 1

        if self.recovery_counter <= 0:
            self.system_mode = SystemMode.NORMAL
            self.phase_counter_internal = 0
            return self.active_phase, None

        # Independent fixed-time cycling — rotates through all phases evenly
        self.phase_counter_internal += 1
        if self.phase_counter_internal >= 100:   # 5 s per phase at 0.05 s/tick
            self.phase_counter_internal = 0
            self.active_phase = (self.active_phase + 1) % self.num_phases

        return self.active_phase, None

    def _handle_fixed_time(self, fallback_mode):
        """
        FIXED_TIME: Safety fallback for low YOLO confidence.

        Cycles through all num_phases phases at 30 s each (600 ticks).
        Returns to NORMAL when FallbackController reports confidence recovered.
        """
        if fallback_mode == ControlMode.DQN:
            self.system_mode = SystemMode.NORMAL
            self.phase_counter_internal = 0
            return self.active_phase, None

        # Independent fixed-time cycling — 30 s per phase
        self.phase_counter_internal += 1
        if self.phase_counter_internal >= 600:
            self.phase_counter_internal = 0
            self.active_phase = (self.active_phase + 1) % self.num_phases

        return self.active_phase, None

    def set_active_phase(self, phase):
        """Update current active phase (called from main.py)."""
        self.active_phase = phase

    def set_emergency_timeout(self, timeout):
        """
        Set emergency timeout based on vehicle speed.
        Called from main.py after detecting ambulance speed.
        """
        self.emergency_timeout = timeout

    def sync_mode_from_main(self, emergency_state_str):
        """
        Synchronise system_mode to match main.py's ground-truth emergency_state.

        main.py runs its own emergency state machine (NORMAL/GRACE/PRE_CLEAR/
        EMERGENCY/RECOVERY). Call this every tick so controllers[idx].system_mode
        stays consistent for logging and diagnostics.

        Parameters
        ----------
        emergency_state_str : str
            Value from main.py's emergency_state[idx] list.
        """
        mapping = {
            'NORMAL':    SystemMode.NORMAL,
            'GRACE':     SystemMode.GRACE,
            'PRE_CLEAR': SystemMode.PRE_CLEAR,
            'EMERGENCY': SystemMode.EMERGENCY,
            'RECOVERY':  SystemMode.RECOVERY,
        }
        new_mode = mapping.get(emergency_state_str, SystemMode.NORMAL)
        if new_mode != self.system_mode:
            self.mode_history.append({
                'from': self.system_mode.value,
                'to':   new_mode.value,
                'tick': self.transition_count
            })
            self.last_mode   = self.system_mode
            self.system_mode = new_mode
