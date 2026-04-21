"""
main.py
=======
Training script for the demo.py system.

Architecture matches demo.py EXACTLY:
  - Single intersection, single-arm control (N/S/E/W)
  - PressureController selects best arm
  - DQN action: 0=keep, 1=allow-switch
  - 7-feature state vector: [pN, pS, pE, pW, arm_idx, time_in_phase, emg_flag]
  - EmergencyHandler FSM: NORMAL→PRE_CLEAR→EMERGENCY_ACTIVE→RECOVERY→NORMAL
  - IntersectionGroundSensors for perception
  - IntersectionWaitingTimeTracker for queue/wait data
  - Weights saved to data/dqn_weights_int1.json (same file demo.py loads)

Usage:
    python main.py --train           # train with emergencies enabled
    python main.py --train --no-emg  # train without emergencies (Phase 1)
    python main.py                   # evaluate loaded weights (no training)
"""

import carla
import random
import os
import csv
import math
import argparse
import numpy as np

from signal_manager    import SignalManager
from controller        import PressureController, YELLOW_TICKS, MIN_GREEN_TICKS, MAX_GREEN_TICKS
from emergency_handler import EmergencyHandler, EmergencyState
from dqn_agent         import DQNAgent, EpisodeTracker, build_state, compute_reward
from waiting_time      import IntersectionWaitingTimeTracker
from ground_sensors    import IntersectionGroundSensors
import itertools

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--train',  action='store_true', help='Train DQN weights')
parser.add_argument('--no-emg', action='store_true', help='Disable emergency vehicles (Phase 1 training)')
parser.add_argument('--episodes', type=int, default=500, help='Episodes to run (default 500)')
parser.add_argument('--vehicles', type=int, default=150, help='Number of vehicles to spawn (default 150)')
args = parser.parse_args()

MODE = 'TRAINING' if args.train else 'EVALUATION'
print(f"\n{'='*55}")
print(f"  Traffic Intelligence System — Single-Arm DQN")
print(f"  Mode      : {MODE}")
print(f"  Emergency : {'DISABLED' if args.no_emg else 'ENABLED'}")
print(f"  Episodes  : {args.episodes}")
print(f"  Vehicles  : {args.vehicles}")
print(f"{'='*55}\n")

# ── Connect ────────────────────────────────────────────────────────────────
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world  = client.get_world()

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode   = True   # headless for training speed
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.global_percentage_speed_difference(-15)
traffic_manager.set_global_distance_to_leading_vehicle(3.0)
print("Connected to CARLA.")

# ── DQN agent (same architecture as demo.py) ───────────────────────────────
agent   = DQNAgent()
tracker = EpisodeTracker(episode_length=500)
tracker.set_int_id(1)

WEIGHT_PATH = "data/dqn_weights_int1.json"
agent.load(WEIGHT_PATH)

if not args.train:
    agent.epsilon = 0.0   # greedy during evaluation
    print("Evaluation mode — epsilon forced to 0.0")
else:
    print(f"Training mode — epsilon={agent.epsilon:.3f}")

# ── Intersection detection (mirrors demo.py logic) ─────────────────────────
def group_lights(lights, threshold=45):
    groups = []
    for light in lights:
        loc   = light.get_location()
        added = False
        for group in groups:
            if loc.distance(group[0].get_location()) < threshold:
                group.append(light); added = True; break
        if not added:
            groups.append([light])
    return groups

traffic_lights = world.get_actors().filter("traffic.traffic_light")
groups         = group_lights(list(traffic_lights), threshold=45)
quad_groups    = [g for g in groups if len(g) >= 4]
valid_groups   = quad_groups if quad_groups else [g for g in groups if len(g) >= 3]

PREFERRED_CENTRE = carla.Location(x=0, y=0, z=0)
valid_groups.sort(key=lambda g: sum(l.get_location().distance(PREFERRED_CENTRE)
                                    for l in g) / len(g))

if not valid_groups:
    raise RuntimeError("No valid intersection found in this map.")

group  = valid_groups[0]
cx     = sum(l.get_location().x for l in group) / len(group)
cy     = sum(l.get_location().y for l in group) / len(group)
cz     = sum(l.get_location().z for l in group) / len(group)
center = carla.Location(x=cx, y=cy, z=cz)
print(f"Intersection center: ({cx:.1f}, {cy:.1f})")

ROI_RADIUS = 35

# ── Assign lights to arms (mirrors demo.py _assign_lights_to_arms) ─────────
def _light_angle_from_center(light):
    loc = light.get_location()
    return math.degrees(math.atan2(loc.y - center.y, loc.x - center.x))

def assign_lights_to_arms(lights):
    targets    = {'N': -90.0, 'S': 90.0, 'E': 0.0, 'W': 180.0}
    arms       = ['N', 'S', 'E', 'W']
    candidates = sorted(lights, key=lambda l: l.get_location().distance(center))[:4]
    if len(candidates) < 4:
        candidates = list(lights)
    if len(candidates) >= 4:
        best_cost, best_map = float('inf'), None
        for perm in itertools.permutations(candidates[:4], 4):
            cost = sum(abs((_light_angle_from_center(l) - targets[a] + 180) % 360 - 180)
                       for a, l in zip(arms, perm))
            if cost < best_cost:
                best_cost = cost
                best_map  = {a: l for a, l in zip(arms, perm)}
        if best_map:
            return best_map
    # greedy fallback
    remaining, assigned = list(candidates), {}
    for arm in arms:
        if not remaining: break
        best = min(remaining,
                   key=lambda l: abs((_light_angle_from_center(l) - targets[arm] + 180) % 360 - 180))
        assigned[arm] = best
        remaining.remove(best)
    return assigned

lane_lights = assign_lights_to_arms(list(group))
for arm in lane_lights:
    lane_lights[arm].freeze(True)
print("Lane-light mapping:", {arm: lane_lights[arm].id for arm in lane_lights})

# ── Core system components (identical to demo.py) ──────────────────────────
signal_manager   = SignalManager(lane_lights)
pressure_ctrl    = PressureController()
emergency_handler = EmergencyHandler(center, roi_radius=ROI_RADIUS + 20)

arm_directions = {'N': 270, 'S': 90, 'E': 0, 'W': 180}
ground_sensor  = IntersectionGroundSensors(
    intersection_id=1, intersection_center=center, arm_directions=arm_directions)
wait_tracker   = IntersectionWaitingTimeTracker(1, ROI_RADIUS, center)

# ── Vehicle spawning ───────────────────────────────────────────────────────
EMERGENCY_KW = ['ambulance', 'firetruck', 'police']
blueprints   = world.get_blueprint_library().filter("vehicle.*")
car_bps      = [bp for bp in blueprints
                if int(bp.get_attribute('number_of_wheels').as_int()) >= 4
                and not any(kw in bp.id.lower() for kw in EMERGENCY_KW)]
spawn_points = world.get_map().get_spawn_points()
vehicles     = []

# Global spawns
for sp in spawn_points[:args.vehicles]:
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)
        vehicles.append(v)

# Nearby spawns around intersection
nearby_sps = [sp for sp in spawn_points if sp.location.distance(center) < 80]
n = 0
for sp in nearby_sps:
    if n >= 40: break
    bp = random.choice(car_bps)
    v  = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)
        vehicles.append(v)
        n += 1

print(f"Spawned {len(vehicles)} vehicles.")
for _ in range(20): world.tick()

# ── Emergency spawn (mirrors demo.py spawn_emergency) ─────────────────────
_approach_spawn_candidates = []
for sp in world.get_map().get_spawn_points():
    if 30 < sp.location.distance(center) < 65:
        _approach_spawn_candidates.append(sp)
_approach_spawn_candidates.sort(key=lambda sp: sp.location.distance(center), reverse=True)
print(f"Emergency spawn candidates: {len(_approach_spawn_candidates)}")

def spawn_emergency():
    if args.no_emg:
        return None
    amb_bps = list(world.get_blueprint_library().filter("vehicle.*ambulance*"))
    if not amb_bps:
        amb_bps = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
                   if any(kw in bp.id.lower() for kw in EMERGENCY_KW)]
    if not amb_bps:
        return None

    for sp in _approach_spawn_candidates:
        bp = random.choice(amb_bps)
        v  = world.try_spawn_actor(bp, sp)
        if v:
            world.tick()
            actual_dist = v.get_location().distance(center)
            if actual_dist > 75:
                v.destroy()
                continue
            v.set_autopilot(True)
            traffic_manager.ignore_lights_percentage(v, 100)
            traffic_manager.ignore_signs_percentage(v, 100)
            traffic_manager.ignore_vehicles_percentage(v, 100)
            traffic_manager.vehicle_percentage_speed_difference(v, -80)
            traffic_manager.distance_to_leading_vehicle(v, 0.5)
            print(f"[EMERGENCY] Spawned {v.type_id} dist={actual_dist:.0f}m")
            return v
    return None

def _safe_destroy(v):
    try:
        if v and v.is_alive:
            v.destroy()
    except RuntimeError:
        pass

def is_emergency_vehicle(v):
    try:
        role = str(v.attributes.get('role_name', '')).lower()
        return role == 'emergency' or any(kw in v.type_id.lower() for kw in EMERGENCY_KW)
    except RuntimeError:
        return False

# ── Initial green arm ──────────────────────────────────────────────────────
_startup_actors  = list(world.get_actors().filter("vehicle.*"))
_startup_result  = ground_sensor.update(_startup_actors, [])
_startup_counts  = _startup_result['arm_counts']
initial_arm      = max(['N', 'S', 'E', 'W'], key=lambda a: _startup_counts.get(a, 0))
pressure_ctrl.current_arm = initial_arm
signal_manager.set_arm_green(initial_arm)
print(f"Initial green arm: {initial_arm}")

# ── Yellow-transition state (identical to demo.py) ─────────────────────────
yellow_active  = False
yellow_arm     = None
yellow_counter = 0
pending_arm    = None

# ── CSV logging ────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
csv_file   = open("data/rl_states_final.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "tick", "episode", "epsilon",
    "current_arm", "ticks_in_phase",
    "pN", "pS", "pE", "pW",
    "queue_N", "queue_S", "queue_E", "queue_W",
    "avg_wait", "queue_length", "throughput_vpm",
    "rl_action", "do_switch", "reward",
    "emergency_state", "emergency_arm",
    "wrong_lane", "unnecessary_switch",
])

# ── Emergency lifecycle counters ───────────────────────────────────────────
EMERGENCY_INTERVAL = 1800   # ~90 s between emergency events during training
EMERGENCY_LIFETIME = 500    # ~25 s max lifetime
emergency_vehicle  = None
emg_counter        = EMERGENCY_INTERVAL   # start immediately after first interval
emg_age            = 0
emg_post_gap       = 0

# ── Main training loop ─────────────────────────────────────────────────────
FIXED_ARM_ORDER  = ['N', 'S', 'E', 'W']
fallback_arm_ticks = 0
FIXED_GREEN_TICKS  = 600

print(f"\nTraining loop starting — {args.episodes} episodes.\n")

tick_count    = 0
prev_state    = None
prev_action   = 0
pressures     = {arm: 0.0 for arm in ['N', 'S', 'E', 'W']}
system_mode   = 'DQN'

try:
    while tracker.episode < args.episodes:
        try:
            world.tick()
        except RuntimeError as e:
            print(f"[WARN] Tick failed: {e}")
            break
        tick_count += 1

        # ── Emergency vehicle lifecycle ────────────────────────────────────
        if emergency_vehicle and emergency_vehicle.is_alive:
            emg_age += 1
            if emg_age >= EMERGENCY_LIFETIME:
                _safe_destroy(emergency_vehicle)
                emergency_vehicle = None
                emg_age = 0
                emg_post_gap = 100
        else:
            if emg_post_gap > 0:
                emg_post_gap -= 1
            else:
                emg_counter += 1
                if emg_counter >= EMERGENCY_INTERVAL and not args.no_emg:
                    emergency_vehicle = spawn_emergency()
                    emg_age     = 0
                    emg_counter = 0 if emergency_vehicle else EMERGENCY_INTERVAL - 60

        all_actors = list(world.get_actors().filter("vehicle.*"))

        # -- Vehicle replenishment (every 300 ticks) -------------------------
        # Vehicles leave the map over time. Without replenishment the
        # intersection empties and the DQN trains on zero-pressure states,
        # which is exactly what caused the useless -1500 reward flat-line.
        if tick_count % 300 == 0:
            alive_nearby = sum(
                1 for v in all_actors
                if v.is_alive
                and v.get_location().distance(center) < 150
                and not is_emergency_vehicle(v)
            )
            if alive_nearby < 40:
                need = 80 - alive_nearby
                random.shuffle(spawn_points)
                added = 0
                for sp in spawn_points:
                    if added >= need:
                        break
                    if sp.location.distance(center) < 130:
                        bp = random.choice(car_bps)
                        nv = world.try_spawn_actor(bp, sp)
                        if nv:
                            nv.set_autopilot(True)
                            vehicles.append(nv)
                            added += 1
                if added:
                    print(f"  [REPLENISH] tick={tick_count} "
                          f"nearby={alive_nearby} added={added}")


        # ── Sensors ────────────────────────────────────────────────────────
        emergency_inputs = ([emergency_vehicle]
                            if emergency_vehicle is not None and emergency_vehicle.is_alive
                            else [])
        sensor_result = ground_sensor.update(all_actors, emergency_inputs)
        wait_tracker.update(all_actors, center, tick_count)
        wt_stats      = wait_tracker.get_stats()

        # Pressure inputs from wait tracker (same as demo.py lines 893-899)
        arm_queues    = dict(wt_stats['arm_queues'])
        arm_avg_waits = dict(wt_stats['arm_avg_waits'])

        # ── Emergency FSM (delegates to EmergencyHandler — same as demo.py) ──
        _was_active = emergency_handler.is_active()
        em_state    = emergency_handler.update_state_machine(all_actors)

        if emergency_handler.is_active():
            system_mode = em_state
            emergency_handler.apply_emergency_control(signal_manager)

            if em_state == EmergencyState.PRE_CLEAR:
                # Cancel any yellow transition mid-flight when emergency triggers.
                # Without this, yellow_active stays True and fires against a stale
                # pending_arm on the first DQN tick after recovery ends.
                yellow_active = False

            elif em_state == EmergencyState.EMERGENCY_ACTIVE:
                emg_arm = emergency_handler.current_arm
                if emg_arm and emg_arm != pressure_ctrl.current_arm:
                    pressure_ctrl.current_arm    = emg_arm
                    pressure_ctrl.ticks_in_phase = 0

            elif em_state == EmergencyState.RECOVERY:
                rec_arm = emergency_handler.current_arm
                if rec_arm and rec_arm != pressure_ctrl.current_arm:
                    # Clear old arm starvation before switching, matching
                    # commit_switch() behaviour where both sides reset on arm change.
                    pressure_ctrl.starvation[pressure_ctrl.current_arm] = 0
                    pressure_ctrl.current_arm    = rec_arm
                    pressure_ctrl.ticks_in_phase = 0
                    pressure_ctrl.starvation[rec_arm] = 0

        elif yellow_active:
            # Yellow transition in progress
            yellow_counter += 1
            signal_manager.set_arm_yellow(yellow_arm)
            if yellow_counter >= YELLOW_TICKS:
                yellow_active = False
                pressure_ctrl.commit_switch(pending_arm)
                signal_manager.set_arm_green(pending_arm)

        else:
            # ── Normal DQN + pressure control (identical to demo.py) ───────
            system_mode = 'DQN'
            pressure_ctrl.tick()
            pressures = pressure_ctrl.compute_pressures(arm_queues, arm_avg_waits)

            rl_state  = build_state(pressures, pressure_ctrl.current_arm,
                                    pressure_ctrl.ticks_in_phase,
                                    1 if emergency_handler.is_active() else 0)
            rl_action = agent.act(rl_state)

            do_switch, best_arm = pressure_ctrl.should_switch(pressures, rl_action)

            best_pressure    = pressures.get(best_arm, 0.0)
            current_pressure = pressures.get(pressure_ctrl.current_arm, 0.0)
            wrong_lane  = 1 if (best_pressure > current_pressure * 1.3
                                and best_arm != pressure_ctrl.current_arm) else 0
            unnecessary = 1 if (do_switch and best_pressure <= current_pressure * 1.05) else 0

            if do_switch:
                pending_arm    = best_arm
                yellow_arm     = pressure_ctrl.current_arm
                yellow_active  = True
                yellow_counter = 0
                signal_manager.set_arm_yellow(yellow_arm)
            else:
                signal_manager.set_arm_green(pressure_ctrl.current_arm)

            # ── Reward + training ──────────────────────────────────────────
            reward = compute_reward(
                avg_waiting_time     = wt_stats['avg_waiting_time'],
                queue_length         = wt_stats['queue_length'],
                vehicles_cleared     = wt_stats.get('throughput_vpm', 0.0) * 0.05,
                wrong_lane_selection = wrong_lane,
                unnecessary_switch   = unnecessary,
            )

            if args.train and prev_state is not None:
                done = tracker.is_done()
                agent.remember(prev_state, prev_action, reward, rl_state, done)
                loss = agent.replay()
                tracker.update(reward, loss)

                if done:
                    tracker.next_episode(agent)
                    # Reset environment state for next episode
                    yellow_active = False
                    pressure_ctrl.ticks_in_phase = 0

            prev_state  = rl_state
            prev_action = rl_action

            # ── CSV row ────────────────────────────────────────────────────
            csv_writer.writerow([
                tick_count, tracker.episode, round(agent.epsilon, 4),
                pressure_ctrl.current_arm, pressure_ctrl.ticks_in_phase,
                round(pressures.get('N', 0), 2), round(pressures.get('S', 0), 2),
                round(pressures.get('E', 0), 2), round(pressures.get('W', 0), 2),
                arm_queues.get('N', 0), arm_queues.get('S', 0),
                arm_queues.get('E', 0), arm_queues.get('W', 0),
                round(wt_stats['avg_waiting_time'], 2),
                wt_stats['queue_length'],
                round(wt_stats.get('throughput_vpm', 0), 2),
                rl_action, int(do_switch), round(reward, 3),
                em_state,
                emergency_handler.current_arm or '',
                wrong_lane, unnecessary,
            ])

        # ── Status print every 200 ticks ──────────────────────────────────
        if tick_count % 200 == 0:
            wt = wt_stats['avg_waiting_time']
            ql = wt_stats['queue_length']
            print(f"  tick={tick_count:6d}  ep={tracker.episode:3d}  "
                  f"eps={agent.epsilon:.3f}  "
                  f"arm={pressure_ctrl.current_arm}  "
                  f"wait={wt:.1f}s  queue={ql}  "
                  f"mode={system_mode}  "
                  f"p=N{pressures.get('N',0):.1f}/"
                  f"S{pressures.get('S',0):.1f}/"
                  f"E{pressures.get('E',0):.1f}/"
                  f"W{pressures.get('W',0):.1f}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    print("\nCleaning up...")
    csv_file.close()

    if args.train:
        agent.save(WEIGHT_PATH)
        print(f"Weights saved → {WEIGHT_PATH}")

    # Restore async mode
    settings.synchronous_mode = False
    try:
        world.apply_settings(settings)
    except RuntimeError:
        pass

    for v in vehicles:
        _safe_destroy(v)
    _safe_destroy(emergency_vehicle)

    print(f"\nDone. Episodes completed: {tracker.episode}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"CSV log: data/rl_states_final.csv")