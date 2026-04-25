"""
signal_manager.py
=================
Strict one-arm-green-at-a-time signal controller.

Supports multiple traffic lights per arm so that multi-lane approaches
all receive the correct state simultaneously.

Safety invariants enforced at every call:
  - Only ONE arm can be GREEN at a time.
  - All other arms are always RED.
  - Yellow applies ONLY to a named arm (all others are RED).
  - No state can produce conflicting or simultaneous greens.
"""

import carla

ARMS = ('N', 'S', 'E', 'W')


class SignalManager:
    def __init__(self, lane_lights: dict):
        """
        lane_lights: dict mapping arm ('N','S','E','W') to either:
          - a single carla.TrafficLight actor, OR
          - a list of carla.TrafficLight actors (multi-lane support).
        """
        # Normalise: every value becomes a list
        self._groups: dict[str, list] = {}
        for arm, val in lane_lights.items():
            self._groups[arm] = val if isinstance(val, list) else [val]

        self._state      = 'all_red'
        self._active_arm: str | None = None
        
        # Performance/Flicker prevention: track actual state of every light
        self._light_states = {} # actor_id -> TrafficLightState
        self._set_all_state(carla.TrafficLightState.Red, force=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_arm_state(self, arm: str, state, force=False) -> None:
        for light in self._groups.get(arm, []):
            try:
                if not force and self._light_states.get(light.id) == state:
                    continue
                
                # Directly set state while frozen to avoid jump/flicker
                light.set_state(state)
                self._light_states[light.id] = state
            except RuntimeError:
                pass

    def _set_all_state(self, state, force=False) -> None:
        for arm in ARMS:
            self._set_arm_state(arm, state, force=force)

    # ── Core primitives ───────────────────────────────────────────────────────

    def set_all_red(self):
        """Set every mapped arm to RED."""
        if self._state == 'all_red':
            return
        self._set_all_state(carla.TrafficLightState.Red)
        self._state      = 'all_red'
        self._active_arm = None

    def set_arm_green(self, arm: str):
        """
        Set *arm* to GREEN and all others to RED.
        """
        if self._state == 'green' and self._active_arm == arm:
            return
        self._set_all_state(carla.TrafficLightState.Red)
        self._set_arm_state(arm, carla.TrafficLightState.Green)
        self._state      = 'green'
        self._active_arm = arm

    def set_arm_yellow(self, arm: str):
        """
        Set *arm* to YELLOW, all others to RED.
        """
        if self._state == 'yellow' and self._active_arm == arm:
            return
        self._set_all_state(carla.TrafficLightState.Red)
        self._set_arm_state(arm, carla.TrafficLightState.Yellow)
        self._state      = 'yellow'
        self._active_arm = arm

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def current_green_arm(self) -> str | None:
        return self._active_arm if self._state == 'green' else None

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def verify(self) -> dict:
        """
        Read actual CARLA light states and check the single-green invariant.
        Returns diagnostic dict with per-arm states and green arm count.
        """
        states      = {}
        green_count = 0
        for arm in ARMS:
            lights = self._groups.get(arm, [])
            if not lights:
                continue
            # Use representative (first) light for the arm's state
            try:
                state = lights[0].get_state()
            except RuntimeError:
                continue
            if state == carla.TrafficLightState.Green:
                states[arm] = 'GREEN'
                green_count += 1
            elif state == carla.TrafficLightState.Yellow:
                states[arm] = 'YELLOW'
            else:
                states[arm] = 'RED'

        if green_count > 1:
            print(f"[SIGNAL ALERT] {green_count} GREEN arms simultaneously — "
                  f"safety violation! States: {states}")
        return {'states': states, 'green_count': green_count}
