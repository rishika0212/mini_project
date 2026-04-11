# ==============================
# WAITING TIME TRACKER
# Tracks per-vehicle waiting time accurately
#
# A vehicle is "waiting" if its speed < 0.5 m/s
# Waiting time = total seconds spent waiting near an intersection
#
# Metrics computed:
#   - avg_waiting_time   : average seconds waited per vehicle per episode
#   - max_waiting_time   : worst case vehicle wait
#   - throughput         : vehicles that passed through intersection per minute
#   - queue_length       : how many vehicles waiting at any moment
# ==============================

import math
import time

WAITING_SPEED_THRESHOLD = 0.5   # m/s — below this = waiting
TICK_DURATION           = 0.05  # seconds per tick (fixed_delta_seconds)


class VehicleWaitTracker:
    """Tracks waiting time for a single vehicle."""

    def __init__(self, vehicle_id, start_tick):
        self.vehicle_id    = vehicle_id
        self.start_tick    = start_tick
        self.waiting_ticks = 0
        self.total_ticks   = 0
        self.is_waiting    = False
        self.wait_start    = None
        self.passed        = False   # has this vehicle cleared the intersection

    def update(self, speed, current_tick):
        self.total_ticks += 1
        if speed < WAITING_SPEED_THRESHOLD:
            self.waiting_ticks += 1
            self.is_waiting     = True
        else:
            self.is_waiting = False

    @property
    def waiting_time_seconds(self):
        return self.waiting_ticks * TICK_DURATION

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

    def __init__(self, intersection_id, roi_radius=50):
        self.intersection_id = intersection_id
        self.roi_radius      = roi_radius

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
        """
        self.tick_count += 1

        # Get vehicles currently in ROI
        in_roi = {}
        for v in world_actors:
            dist = v.get_location().distance(center)
            if dist < self.roi_radius:
                vel   = v.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                in_roi[v.id] = speed

        # Update active trackers
        for vid, speed in in_roi.items():
            if vid not in self.active:
                self.active[vid] = VehicleWaitTracker(vid, current_tick)
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
        """Returns current episode statistics."""
        # Include active vehicles still in ROI
        all_waits = self.episode_waiting_times.copy()
        for tracker in self.active.values():
            if tracker.total_ticks > 10:
                all_waits.append(tracker.waiting_time_seconds)

        # Current queue length
        queue_length = sum(1 for t in self.active.values()
                           if t.is_waiting)

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
        }

    def reset_episode(self):
        """Call at episode boundary to reset per-episode stats."""
        # Save completed waiting times
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
