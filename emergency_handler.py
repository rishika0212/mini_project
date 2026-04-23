"""
emergency_handler.py
====================
Robust, deterministic emergency vehicle handling for CARLA intersections.

State machine (strictly ordered — no skipping):
  NORMAL ──emergency detected──► PRE_CLEAR  (~2 s, all-RED)
  PRE_CLEAR ──timer expires──► EMERGENCY_ACTIVE  (priority arm GREEN)
  EMERGENCY_ACTIVE ──vehicle exits or timeout──► RECOVERY  (~20 s)
  RECOVERY ──timer expires──► NORMAL

Public API (four required methods):
  detect_emergency(all_vehicles)      → list of qualifying actors
  get_emergency_lane(vehicle)         → 'N'|'S'|'E'|'W' or None
  update_state_machine(all_vehicles)  → current EmergencyState string
  apply_emergency_control(sig_mgr)    → set signals; handles all states

Additional helpers:
  is_active()          → True when handler owns signal control
  recovery_fraction()  → progress through RECOVERY (0.0 → 1.0)
  last_detected        → list from most recent update_state_machine()
  current_arm          → arm currently green (or None)
  in_yellow            → True when a yellow transition is in progress

Backward-compatible:
  update(emergency_vehicles)  → legacy thin wrapper (prefer update_state_machine)
  EmergencyState.DQN          → alias for NORMAL
  EmergencyState.EMERGENCY    → alias for EMERGENCY_ACTIVE
"""

import math

# ── Timing constants (ticks at 0.05 s/tick) ──────────────────────────────────
PRE_CLEAR_TICKS     = 40    # ~2 s   — all-RED safety clearance
EMERGENCY_TIMEOUT   = 500   # ~25 s  — max GREEN before forced recovery
RECOVERY_TICKS      = 900   # ~45 s  — ensured full cycle (4 x 220 ticks)
RECOVERY_ARM_TICKS  = 180   # ~9 s   — time on each arm during cycling
YELLOW_TICKS        = 40    # ~2 s   — yellow transition

# ── Detection geometry ────────────────────────────────────────────────────────
DETECTION_RANGE     = 40.0  # metres — maximum distance to detect emergency vehicle
INTERSECTION_ZONE   = 8.0   # metres — vehicles inside this radius are already through

# ── Vehicle identification ────────────────────────────────────────────────────
EMERGENCY_TYPE_KW   = ('ambulance', 'firetruck', 'police')
FIXED_ARM_ORDER     = ['N', 'E', 'S', 'W']   # recovery cycling sequence


# ── State names ───────────────────────────────────────────────────────────────

class EmergencyState:
    """Symbolic state constants for the FSM."""
    NORMAL           = 'NORMAL'
    PRE_CLEAR        = 'PRE_CLEAR'
    EMERGENCY_ACTIVE = 'EMERGENCY_ACTIVE'
    RECOVERY         = 'RECOVERY'

    # Backward-compatibility aliases so existing callers need not change
    DQN      = NORMAL
    EMERGENCY = EMERGENCY_ACTIVE


# ── Module-level helper (kept for backward compatibility with demo.py) ────────

def arm_from_location(loc, center) -> str:
    """
    Return which arm ('N','S','E','W') a vehicle is on, purely from geometry.
    Town03: East=+X, South=+Y, West=-X, North=-Y
    """
    dx = loc.x - center.x
    dy = loc.y - center.y
    if abs(dy) >= abs(dx):
        return 'S' if dy > 0 else 'N'
    return 'E' if dx > 0 else 'W'


# ── Internal utility ──────────────────────────────────────────────────────────

def _safe_distance(vehicle, center) -> float:
    """Distance from vehicle to center; returns ∞ on actor error."""
    try:
        return vehicle.get_location().distance(center)
    except RuntimeError:
        return float('inf')


# ── Main class ────────────────────────────────────────────────────────────────

class EmergencyHandler:
    """
    Self-contained emergency vehicle detector and intersection signal manager.

    The handler owns the full emergency lifecycle:
      detect → PRE_CLEAR → priority GREEN → RECOVERY → back to normal.

    Callers only need two calls per tick:
        state = handler.update_state_machine(all_actors)
        if handler.is_active():
            handler.apply_emergency_control(signal_manager)
    """

    def __init__(self, center, roi_radius: float = DETECTION_RANGE):
        """
        center     : carla.Location of intersection centre
        roi_radius : detection range for emergency vehicles (metres)
        """
        self.center     = center
        self.roi_radius = roi_radius

        # ── FSM state ─────────────────────────────────────────────────────────
        self.state         = EmergencyState.NORMAL
        self.emergency_arm = None   # arm locked on entry to EMERGENCY_ACTIVE

        # ── Tick counters ─────────────────────────────────────────────────────
        self._pre_ticks  = 0
        self._emg_ticks  = 0
        self._rec_ticks  = 0

        # ── Recovery cycling state ────────────────────────────────────────────
        self._rec_arm_idx         = 0
        self._rec_arm_phase_ticks = 0
        self._rec_in_yellow       = False
        self._rec_yellow_ticks    = 0
        self._rec_current_arm     = FIXED_ARM_ORDER[0]

        # ── Cache from last update ────────────────────────────────────────────
        self._last_detected: list = []

    # ── Required public API ───────────────────────────────────────────────────

    def detect_emergency(self, all_vehicles: list, known: list = None) -> list:
        """
        Scan all_vehicles for emergency vehicles approaching this intersection.

        known: optional list of vehicles already identified as emergency by the caller
               (e.g. a spawned emergency actor tracked in demo.py). These bypass the
               type-ID check and the direction filter — only range and zone are checked.
               This handles generic/promoted vehicles whose type_id and role_name don't
               match any emergency keyword.

        Admission criteria for type-discovered vehicles (all must pass):
          1. Vehicle type_id or role_name matches an emergency keyword.
          2. Distance ≤ roi_radius from intersection centre.
          3. Distance ≥ INTERSECTION_ZONE (not already inside intersection).
          4. Moving vehicles (speed > 0.5 m/s) must be heading toward centre.
             Stopped vehicles are always admitted — they may be queued at light.

        Handles destroyed or otherwise inaccessible actors gracefully.
        Returns a list of qualifying CARLA vehicle actors (may be empty).
        """
        detected     = []
        detected_ids = set()

        # ── Known/forced vehicles: bypass type and direction checks ───────────
        for v in (known or []):
            try:
                if not v.is_alive:
                    continue
                dist = _safe_distance(v, self.center)
                if self.roi_radius >= dist >= INTERSECTION_ZONE:
                    detected.append(v)
                    detected_ids.add(v.id)
            except RuntimeError:
                continue

        # ── Standard type-based detection ─────────────────────────────────────
        for v in all_vehicles:
            try:
                if v.id in detected_ids:
                    continue            # already added via known path
                if not v.is_alive:
                    continue
                if not self._is_emergency_type(v):
                    continue

                loc  = v.get_location()
                dx   = loc.x - self.center.x
                dy   = loc.y - self.center.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > self.roi_radius:
                    continue        # too far away

                # Admission criteria:
                # 1. Moving vehicles (speed > 0.5 m/s) must be heading toward centre or be inside.
                # 2. Driving away (dot < 0) means it has crossed.
                vel   = v.get_velocity()
                speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                if speed > 0.5:
                    dir_x = -dx / dist
                    dir_y = -dy / dist
                    dot   = (vel.x * dir_x + vel.y * dir_y) / speed
                    if dot < -0.2:  # Tolerance for "heading away"
                        continue    # driving away — already crossed

                detected.append(v)

            except RuntimeError:
                continue            # actor destroyed between checks — skip safely

        return detected

    def get_emergency_lane(self, vehicle) -> 'str | None':
        """
        Return the arm ('N','S','E','W') the vehicle is approaching from.

        Returns None if:
          - Vehicle is inside the intersection zone (arm is ambiguous).
          - Vehicle actor is inaccessible (destroyed).

        The arm is determined purely from geometry — no turn prediction needed.
        CARLA axes: +X = East, +Y = South.
        """
        try:
            loc  = vehicle.get_location()
            dx   = loc.x - self.center.x
            dy   = loc.y - self.center.y

            if abs(dx) < INTERSECTION_ZONE and abs(dy) < INTERSECTION_ZONE:
                return None     # inside intersection — arm undefined

            if abs(dy) >= abs(dx):
                return 'S' if dy > 0 else 'N'
            return 'E' if dx > 0 else 'W'

        except RuntimeError:
            return None

    def update_state_machine(self, all_vehicles: list, known: list = None) -> str:
        """
        Advance the FSM one tick using freshly detected emergency vehicles.

        known: optional list of pre-identified emergency vehicles passed directly
               by the caller (see detect_emergency for details).
        Detected vehicles are cached in self.last_detected for event logging.
        Returns the current EmergencyState string.
        """
        emg_vehicles        = self.detect_emergency(all_vehicles, known=known)
        self._last_detected = emg_vehicles
        emg_present         = bool(emg_vehicles)

        if self.state == EmergencyState.NORMAL:
            if emg_present:
                self._enter_pre_clear(emg_vehicles)

        elif self.state == EmergencyState.PRE_CLEAR:
            self._pre_ticks -= 1
            if self._pre_ticks <= 0:
                if emg_present:
                    self._enter_emergency_active(emg_vehicles)
                else:
                    # Vehicle cleared before pre-clear finished — skip to recovery
                    self._enter_recovery("vehicle left during PRE_CLEAR")

        elif self.state == EmergencyState.EMERGENCY_ACTIVE:
            # Arm is LOCKED at entry — never refreshed here to prevent flipping
            # when vehicle enters intersection zone (where arm_from_location() is unreliable).
            self._emg_ticks -= 1
            if not emg_present:
                self._enter_recovery("vehicle exited ROI")
            elif self._emg_ticks <= 0:
                self._enter_recovery("timeout")

        elif self.state == EmergencyState.RECOVERY:
            self._rec_ticks -= 1
            if self._rec_ticks <= 0:
                self.state         = EmergencyState.NORMAL
                self.emergency_arm = None
                print("[EMERGENCY] Recovery complete — resuming normal control.")

        return self.state

    def apply_emergency_control(self, signal_manager) -> None:
        """
        Set signal states for the current emergency state.

        PRE_CLEAR        → all lights RED (clear intersection)
        EMERGENCY_ACTIVE → emergency arm GREEN, all others RED (idempotent)
        RECOVERY         → fixed-time arm cycling with yellow transitions
        NORMAL           → no-op (caller controls signals)

        Safe to call every tick. Wraps all signal calls in try/except so a
        CARLA RPC error never propagates to the main loop.
        """
        try:
            if self.state == EmergencyState.PRE_CLEAR:
                signal_manager.set_all_red()

            elif self.state == EmergencyState.EMERGENCY_ACTIVE:
                if self.emergency_arm:
                    signal_manager.set_arm_green(self.emergency_arm)
                else:
                    signal_manager.set_all_red()    # safety fallback

            elif self.state == EmergencyState.RECOVERY:
                self._tick_recovery_signal(signal_manager)

            # NORMAL → no-op

        except (RuntimeError, AttributeError) as exc:
            print(f"[EMERGENCY] Signal control error (ignored): {exc}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def last_detected(self) -> list:
        """List of emergency vehicles found during the last update_state_machine call."""
        return self._last_detected

    @property
    def current_arm(self) -> 'str | None':
        """
        Arm that is currently (or about to be) GREEN.
        EMERGENCY_ACTIVE  → locked emergency arm
        RECOVERY          → current recovery cycling arm
        NORMAL/PRE_CLEAR  → None
        """
        if self.state == EmergencyState.EMERGENCY_ACTIVE:
            return self.emergency_arm
        if self.state == EmergencyState.RECOVERY:
            return self._rec_current_arm
        return None

    @property
    def in_yellow(self) -> bool:
        """True when a yellow transition is in progress during RECOVERY."""
        return self.state == EmergencyState.RECOVERY and self._rec_in_yellow

    def is_active(self) -> bool:
        """True when the handler owns signal control (any state except NORMAL)."""
        return self.state != EmergencyState.NORMAL

    def recovery_fraction(self) -> float:
        """0.0 = just entered RECOVERY, 1.0 = recovery complete."""
        if self.state != EmergencyState.RECOVERY or RECOVERY_TICKS == 0:
            return 1.0
        return 1.0 - (self._rec_ticks / RECOVERY_TICKS)

    # ── Backward-compatible legacy API ────────────────────────────────────────

    def update(self, emergency_vehicles: list) -> str:
        """
        Legacy API: accepts a pre-filtered list instead of all_vehicles.
        Prefer update_state_machine(all_vehicles) for new code.
        """
        self._last_detected = emergency_vehicles
        emg_present         = bool(emergency_vehicles)

        if self.state == EmergencyState.NORMAL:
            if emg_present:
                self._enter_pre_clear(emergency_vehicles)

        elif self.state == EmergencyState.PRE_CLEAR:
            self._pre_ticks -= 1
            if self._pre_ticks <= 0:
                if emg_present:
                    self._enter_emergency_active(emergency_vehicles)
                else:
                    self._enter_recovery("vehicle left during PRE_CLEAR")

        elif self.state == EmergencyState.EMERGENCY_ACTIVE:
            self._emg_ticks -= 1
            if not emg_present:
                self._enter_recovery("vehicle exited ROI")
            elif self._emg_ticks <= 0:
                self._enter_recovery("timeout")

        elif self.state == EmergencyState.RECOVERY:
            self._rec_ticks -= 1
            if self._rec_ticks <= 0:
                self.state         = EmergencyState.NORMAL
                self.emergency_arm = None
                print("[EMERGENCY] Recovery complete — resuming normal control.")

        return self.state

    # ── Internal FSM transitions ──────────────────────────────────────────────

    def _enter_pre_clear(self, emg_vehicles: list):
        self.state      = EmergencyState.PRE_CLEAR
        self._pre_ticks = PRE_CLEAR_TICKS
        names = self._vehicle_names(emg_vehicles)
        print(f"[EMERGENCY] Detected ({names}) — "
              f"PRE_CLEAR: all-RED for {PRE_CLEAR_TICKS * 0.05:.1f}s.")

    def _enter_emergency_active(self, emg_vehicles: list):
        self.state      = EmergencyState.EMERGENCY_ACTIVE
        self._emg_ticks = EMERGENCY_TIMEOUT

        # Pick the closest vehicle and LOCK the arm.
        # The arm is NOT updated again during EMERGENCY_ACTIVE because once the
        # vehicle enters the intersection zone (< INTERSECTION_ZONE m) the
        # geometry-based arm becomes unreliable.
        closest = min(emg_vehicles, key=lambda v: _safe_distance(v, self.center))
        arm = self.get_emergency_lane(closest)

        if arm is None:
            # Vehicle is very close to centre already; fall back to last known arm
            arm = self.emergency_arm or 'N'

        self.emergency_arm = arm
        print(f"[EMERGENCY] ACTIVE — arm {self.emergency_arm} GREEN "
              f"(max {EMERGENCY_TIMEOUT * 0.05:.0f}s).")

    def _enter_recovery(self, reason: str = ""):
        self.state                = EmergencyState.RECOVERY
        self._rec_ticks           = RECOVERY_TICKS
        self._rec_arm_idx         = 0
        self._rec_arm_phase_ticks = 0
        self._rec_in_yellow       = False
        self._rec_yellow_ticks    = 0
        self._rec_current_arm     = FIXED_ARM_ORDER[0]
        self.emergency_arm        = None
        msg = f" ({reason})" if reason else ""
        print(f"[EMERGENCY] RECOVERY{msg} — "
              f"cycling arms for {RECOVERY_TICKS * 0.05:.0f}s.")

    def _tick_recovery_signal(self, signal_manager):
        """
        Drive the fixed-time arm-cycling during RECOVERY with yellow transitions.
        Called every tick from apply_emergency_control().
        """
        if self._rec_in_yellow:
            # Yellow phase: count down then switch to next arm
            self._rec_yellow_ticks -= 1
            signal_manager.set_arm_yellow(self._rec_current_arm)
            if self._rec_yellow_ticks <= 0:
                self._rec_arm_idx         = (self._rec_arm_idx + 1) % len(FIXED_ARM_ORDER)
                self._rec_current_arm     = FIXED_ARM_ORDER[self._rec_arm_idx]
                self._rec_arm_phase_ticks = 0
                self._rec_in_yellow       = False
                signal_manager.set_arm_green(self._rec_current_arm)
        else:
            # Green phase: hold arm until arm duration expires
            signal_manager.set_arm_green(self._rec_current_arm)
            self._rec_arm_phase_ticks += 1
            if self._rec_arm_phase_ticks >= RECOVERY_ARM_TICKS:
                self._rec_in_yellow    = True
                self._rec_yellow_ticks = YELLOW_TICKS
                # signal_manager.set_arm_yellow called on next tick

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_emergency_type(vehicle) -> bool:
        """True if vehicle type_id or role_name matches an emergency keyword."""
        try:
            role = str(vehicle.attributes.get('role_name', '')).lower()
            if role == 'emergency':
                return True
            return any(kw in vehicle.type_id.lower() for kw in EMERGENCY_TYPE_KW)
        except RuntimeError:
            return False

    @staticmethod
    def _vehicle_names(vehicles: list) -> str:
        """Comma-separated short type names for log messages."""
        names = []
        for v in vehicles:
            try:
                names.append(v.type_id.split('.')[-1])
            except RuntimeError:
                names.append('unknown')
        return ', '.join(names) if names else 'none'
