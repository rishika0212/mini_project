# ==============================
# WAITING TIME TRACKER
# Tracks per-vehicle waiting time accurately, with per-arm breakdowns.
#
# A vehicle is "waiting" if its speed < 0.5 m/s
# Arms: N / S / E / W — determined by vehicle position relative to center.
#   CARLA coords: +X ≈ East, +Y ≈ South (UE4 left-hand system)
#
# Metrics computed:
#   - avg_waiting_time   : average seconds waited per vehicle per episode
#   - max_waiting_time   : worst case vehicle wait
#   - throughput         : vehicles that passed through intersection per minute
#   - queue_length       : how many vehicles waiting at any moment
#   - arm_queues         : {N,S,E,W} count of waiting vehicles per arm
#   - arm_avg_waits      : {N,S,E,W} avg waiting time per arm (seconds)
# ==============================

import math

WAITING_SPEED_THRESHOLD = 0.5   # m/s — below this = waiting
TICK_DURATION           = 0.05  # seconds per tick (fixed_delta_seconds)


INTERSECTION_ZONE_M = 8.0    # vehicles closer than this to center are inside intersection

def _get_arm(loc, center):
    """
    Classify a vehicle into N/S/E/W approach arm.
    Returns None if the vehicle is inside the intersection zone.
    Town03: North=+X, South=-X, East=+Y, West=-Y
    """
    dx = loc.x - center.x
    dy = loc.y - center.y
    if abs(dx) < INTERSECTION_ZONE_M and abs(dy) < INTERSECTION_ZONE_M:
        return None
    if abs(dx) >= abs(dy):
        return 'N' if dx >= 0 else 'S'
    return 'E' if dy >= 0 else 'W'


class VehicleWaitTracker:
    """Tracks waiting time for a single vehicle."""

    def __init__(self, vehicle_id, start_tick, arm='?'):
        self.vehicle_id          = vehicle_id
        self.start_tick          = start_tick
        self.waiting_ticks       = 0   # current continuous stop (resets on movement)
        self.total_waiting_ticks = 0   # cumulative stopped ticks across all stops
        self.total_ticks         = 0
        self.is_waiting          = False
        self.passed              = False
        self.arm                 = arm   # 'N', 'S', 'E', or 'W'

    def update(self, speed, current_tick):
        self.total_ticks += 1
        if speed < WAITING_SPEED_THRESHOLD:
            self.waiting_ticks       += 1
            self.total_waiting_ticks += 1
            self.is_waiting           = True
        else:
            self.is_waiting    = False
            self.waiting_ticks = 0

    @property
    def current_wait_seconds(self):
        """Current continuous stop duration — used for per-arm pressure."""
        return self.waiting_ticks * TICK_DURATION

    @property
    def waiting_time_seconds(self):
        """Total accumulated wait across all stops — used for episode/dashboard stats."""
        return self.total_waiting_ticks * TICK_DURATION

    @property
    def total_time_seconds(self):
        return self.total_ticks * TICK_DURATION


class IntersectionWaitingTimeTracker:
    """
    Tracks waiting time for all vehicles near one intersection.
    Call update() every tick.
    Call get_stats() to get current episode metrics.
    Call reset_episode() at episode boundaries.
    """

    def __init__(self, intersection_id, roi_radius=35, center=None):
        self.intersection_id = intersection_id
        self.roi_radius      = roi_radius
        self.center          = center   # carla.Location (needs .x, .y attributes)

        # Active trackers: vehicle_id -> VehicleWaitTracker
        self.active   = {}

        # Completed trackers (vehicle left ROI)
        self.completed = []

        # Episode stats
        self.episode_waiting_times = []
        self.tick_count            = 0
        self.throughput_count      = 0   # vehicles that entered AND left ROI

    def update(self, world_actors, center, current_tick):
        """
        Call every tick.
        world_actors = world.get_actors().filter('vehicle.*')
        center       = carla.Location of intersection center

        Admission rules for new vehicles entering the tracker:
          - Must be within roi_radius and outside INTERSECTION_ZONE_M.
          - Moving vehicles (speed > 0.5 m/s) must be approaching the center
            (positive dot product of velocity toward center). This filters out
            vehicles that have already crossed and are driving away.
          - Stopped vehicles are always admitted — they represent the queue.
        Already-tracked vehicles are updated regardless of direction.
        """
        self.tick_count += 1

        # Get vehicles currently in ROI
        in_roi = {}
        for v in world_actors:
            loc  = v.get_location()
            dx   = loc.x - center.x
            dy   = loc.y - center.y
            dist = math.sqrt(dx * dx + dy * dy)

            # Skip intersection zone and vehicles beyond ROI
            if dist < INTERSECTION_ZONE_M or dist >= self.roi_radius:
                continue

            vel   = v.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            arm   = _get_arm(loc, center)

            # Skip if arm is None (redundant safety check for intersection zone)
            if arm is None:
                continue

            # For NEW vehicles: apply approaching filter to moving vehicles.
            # Vehicles already tracked are kept regardless of direction.
            if v.id not in self.active and speed > 0.5:
                dir_x = -dx / dist
                dir_y = -dy / dist
                dot   = (vel.x * dir_x + vel.y * dir_y) / speed
                if dot < 0:
                    continue  # moving away — already passed the intersection

            in_roi[v.id] = (speed, arm)

        # Update active trackers
        for vid, (speed, arm) in in_roi.items():
            if vid not in self.active:
                self.active[vid] = VehicleWaitTracker(vid, current_tick, arm)
            self.active[vid].update(speed, current_tick)

        # Detect vehicles that left ROI → mark as completed
        left_ids = [vid for vid in self.active if vid not in in_roi]
        for vid in left_ids:
            tracker = self.active.pop(vid)
            if tracker.total_ticks > 10:   # ignore very brief entries
                self.completed.append(tracker)
                self.episode_waiting_times.append(tracker.waiting_time_seconds)
                self.throughput_count += 1

    def get_stats(self):
        """Returns current episode statistics including per-arm breakdowns."""
        # Include active vehicles still in ROI
        all_waits = self.episode_waiting_times.copy()
        for tracker in self.active.values():
            if tracker.total_ticks > 10:
                all_waits.append(tracker.waiting_time_seconds)

        # Overall queue length
        queue_length = sum(1 for t in self.active.values() if t.is_waiting)

        # Per-arm stats (from currently active vehicles)
        arm_queues    = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        arm_wait_secs = {'N': [], 'S': [], 'E': [], 'W': []}
        for t in self.active.values():
            arm = t.arm
            if arm and arm in arm_queues:
                if t.is_waiting:
                    arm_queues[arm] += 1
                    arm_wait_secs[arm].append(t.current_wait_seconds)
        arm_avg_waits = {
            arm: (sum(ws) / len(ws)) if ws else 0.0
            for arm, ws in arm_wait_secs.items()
        }

        if all_waits:
            avg_wait = sum(all_waits) / len(all_waits)
            max_wait = max(all_waits)
        else:
            avg_wait = 0.0
            max_wait = 0.0

        # Throughput: vehicles/minute
        elapsed_minutes = (self.tick_count * TICK_DURATION) / 60.0
        throughput = self.throughput_count / elapsed_minutes \
                     if elapsed_minutes > 0 else 0.0

        return {
            'intersection_id'  : self.intersection_id,
            'avg_waiting_time' : round(avg_wait, 3),
            'max_waiting_time' : round(max_wait, 3),
            'queue_length'     : queue_length,
            'throughput_vpm'   : round(throughput, 3),
            'vehicles_tracked' : len(all_waits),
            'vehicles_in_roi'  : len(self.active),
            'arm_queues'       : arm_queues,
            'arm_avg_waits'    : arm_avg_waits,
        }

    def reset_episode(self):
        """Call at episode boundary to reset per-episode stats."""
        for tracker in self.active.values():
            if tracker.total_ticks > 10:
                self.completed.append(tracker)

        self.active                = {}
        self.completed             = []
        self.episode_waiting_times = []
        self.tick_count            = 0
        self.throughput_count      = 0

    def get_all_time_stats(self):
        """Overall stats across all episodes."""
        all_waits = self.episode_waiting_times.copy()
        for t in self.active.values():
            if t.total_ticks > 10:
                all_waits.append(t.waiting_time_seconds)
        for t in self.completed:
            all_waits.append(t.waiting_time_seconds)

        if not all_waits:
            return {'avg': 0, 'max': 0, 'count': 0}

        return {
            'avg'  : round(sum(all_waits)/len(all_waits), 3),
            'max'  : round(max(all_waits), 3),
            'count': len(all_waits)
        }
