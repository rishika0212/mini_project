"""
signal_manager.py
=================
Strict one-arm-green-at-a-time signal controller.

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
        lane_lights: dict mapping arm ('N','S','E','W') to carla.TrafficLight actor.
        """
        self.lane_lights = lane_lights
        # Internal state tracking so we only issue CARLA RPC calls when the
        # desired state differs from what is already set.  Calling set_state()
        # every tick causes CARLA to flicker through the intermediate all-RED
        # step even in synchronous mode.
        self._state      = 'all_red'   # 'all_red' | 'green' | 'yellow'
        self._active_arm: str | None = None   # arm that is GREEN or YELLOW

    # ── Core primitives ───────────────────────────────────────────────────────

    def set_all_red(self):
        """Set every mapped arm to RED. Skipped if already all-red."""
        if self._state == 'all_red':
            return
        for arm in ARMS:
            light = self.lane_lights.get(arm)
            if light is not None:
                light.set_state(carla.TrafficLightState.Red)
        self._state      = 'all_red'
        self._active_arm = None

    def set_arm_green(self, arm: str):
        """
        Set *arm* to GREEN and all others to RED.
        No-op if *arm* is already the active green arm (prevents flickering).
        """
        if self._state == 'green' and self._active_arm == arm:
            return  # already correct — skip RPC calls
        for a in ARMS:
            light = self.lane_lights.get(a)
            if light is not None:
                light.set_state(carla.TrafficLightState.Red)
        light = self.lane_lights.get(arm)
        if light is not None:
            light.set_state(carla.TrafficLightState.Green)
        self._state      = 'green'
        self._active_arm = arm

    def set_arm_yellow(self, arm: str):
        """
        Set *arm* to YELLOW, all others to RED.
        No-op if *arm* is already the active yellow arm (prevents flickering).
        """
        if self._state == 'yellow' and self._active_arm == arm:
            return  # already correct — skip RPC calls
        for a in ARMS:
            light = self.lane_lights.get(a)
            if light is not None:
                light.set_state(carla.TrafficLightState.Red)
        light = self.lane_lights.get(arm)
        if light is not None:
            light.set_state(carla.TrafficLightState.Yellow)
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
        Returns diagnostic dict with per-arm states and green count.
        Prints an alert if more than one arm is GREEN.
        """
        states = {}
        green_count = 0
        for arm in ARMS:
            light = self.lane_lights.get(arm)
            if light is None:
                continue
            state = light.get_state()
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
