"""
ground_sensors.py
=================
Realistic inductive loop sensors for traffic detection.

Each intersection has 4 ground sensors (N/S/E/W arms).
Each sensor detects vehicles in its waiting zone (before stop line).
Emergency vehicles detected and flagged separately.
"""

import carla
import math

class GroundSensor:
    """Simulates an inductive loop sensor on one approach arm."""

    def __init__(self, arm_name, intersection_center, arm_direction, detection_range=50):
        """
        arm_name: 'N', 'S', 'E', 'W'
        intersection_center: carla.Location of intersection
        arm_direction: yaw angle of the road
        detection_range: how far back to detect vehicles (50m default)
        """
        self.arm_name = arm_name
        self.center = intersection_center
        self.direction = arm_direction
        self.range = detection_range

        # Sensor position: 30m back from intersection on this arm
        self.sensor_pos = self._calculate_sensor_position()

        self.vehicle_count = 0
        self.emergency_count = 0
        self.emergency_vehicle = None

    def _calculate_sensor_position(self):
        """Calculate sensor position 30m back on the approach arm."""
        rad = math.radians(self.direction)
        back_distance = 30
        x = self.center.x + back_distance * math.cos(rad)
        y = self.center.y + back_distance * math.sin(rad)
        z = self.center.z
        return carla.Location(x=x, y=y, z=z)

    def update(self, all_vehicles, emergency_vehicles):
        """
        Count vehicles in waiting zone on this arm.
        Returns: (vehicle_count, emergency_detected, emergency_vehicle)
        """
        self.vehicle_count = 0
        self.emergency_vehicle = None
        self.emergency_count = 0

        # Detect all regular vehicles in zone
        for v in all_vehicles:
            if not v.is_alive:
                continue

            loc = v.get_location()
            dist_to_sensor = loc.distance(self.sensor_pos)

            # Vehicle within detection range?
            if dist_to_sensor <= self.range:
                # Check if vehicle is on this arm (angular check)
                dx = loc.x - self.center.x
                dy = loc.y - self.center.y
                angle = math.degrees(math.atan2(dy, dx))

                # Angular tolerance for arm detection
                diff = abs((angle - self.direction + 180) % 360 - 180)
                if diff < 45:  # Within ±45° of arm direction
                    self.vehicle_count += 1

        # Detect emergency vehicles in zone
        for emg_v in emergency_vehicles:
            if not emg_v.is_alive:
                continue

            loc = emg_v.get_location()
            dist_to_sensor = loc.distance(self.sensor_pos)

            if dist_to_sensor <= self.range:
                dx = loc.x - self.center.x
                dy = loc.y - self.center.y
                angle = math.degrees(math.atan2(dy, dx))

                diff = abs((angle - self.direction + 180) % 360 - 180)
                if diff < 45:
                    self.emergency_count += 1
                    self.emergency_vehicle = emg_v

        return self.vehicle_count, self.emergency_count > 0, self.emergency_vehicle


class IntersectionGroundSensors:
    """Manages 4 ground sensors for one intersection."""

    def __init__(self, intersection_id, intersection_center, arm_directions):
        """
        arm_directions: dict with keys 'N','S','E','W' and yaw angles as values
        """
        self.intersection_id = intersection_id
        self.center = intersection_center

        self.sensors = {
            'N': GroundSensor('N', intersection_center, arm_directions['N']),
            'S': GroundSensor('S', intersection_center, arm_directions['S']),
            'E': GroundSensor('E', intersection_center, arm_directions['E']),
            'W': GroundSensor('W', intersection_center, arm_directions['W']),
        }

        self.arm_counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.emergency_flag = 0
        self.emergency_vehicle = None

    def update(self, all_vehicles, emergency_vehicles):
        """
        Update all 4 sensors.
        Returns: {
            'arm_counts': {'N': count, 'S': count, 'E': count, 'W': count},
            'emergency_flag': 0 or 1,
            'emergency_vehicle': vehicle or None
        }
        """
        self.emergency_flag = 0
        self.emergency_vehicle = None

        for arm, sensor in self.sensors.items():
            count, has_emg, emg_v = sensor.update(all_vehicles, emergency_vehicles)
            self.arm_counts[arm] = count

            if has_emg:
                self.emergency_flag = 1
                self.emergency_vehicle = emg_v

        return {
            'arm_counts': self.arm_counts,
            'emergency_flag': self.emergency_flag,
            'emergency_vehicle': self.emergency_vehicle
        }

    def get_status(self):
        """Return sensor status for diagnostics."""
        return {
            'intersection_id': self.intersection_id,
            'N_vehicles': self.arm_counts['N'],
            'S_vehicles': self.arm_counts['S'],
            'E_vehicles': self.arm_counts['E'],
            'W_vehicles': self.arm_counts['W'],
            'emergency_flag': self.emergency_flag,
            'total_vehicles': sum(self.arm_counts.values()),
        }
