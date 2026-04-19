import numpy as np
import random
from collections import deque
import json
import os

# ==============================
# DEEP Q-NETWORK AGENT
# ==============================

# STATE (12 features):
#   [0]  arm_N_queue  — vehicles waiting on North arm  (normalised)
#   [1]  arm_S_queue  — vehicles waiting on South arm  (normalised)
#   [2]  arm_E_queue  — vehicles waiting on East arm   (normalised)
#   [3]  arm_W_queue  — vehicles waiting on West arm   (normalised)
#   [4]  arm_N_wait   — avg wait time, North arm       (normalised)
#   [5]  arm_S_wait   — avg wait time, South arm       (normalised)
#   [6]  arm_E_wait   — avg wait time, East arm        (normalised)
#   [7]  arm_W_wait   — avg wait time, West arm        (normalised)
#   [8]  active_phase — current green phase 0-3        (normalised /3)
#   [9]  phase_time   — ticks spent in current phase   (normalised)
#   [10] time_of_day  — elapsed time modulo 10 min     (normalised)
#   [11] emergency    — 1 if emergency vehicle in ROI

# ACTION (4):
#   0 = Phase 0 — NS Straight + Right  (N & S green, E & W red)
#   1 = Phase 1 — NS Left turns        (N & S green protected left)
#   2 = Phase 2 — EW Straight + Right  (E & W green, N & S red)
#   3 = Phase 3 — EW Left turns        (E & W green protected left)
#
# DQN chooses the NEXT desired phase. Transition (yellow → all-red) is
# hardcoded in main.py and not controlled by the agent.

STATE_SIZE  = 12
ACTION_SIZE = 4   # choose next phase (0-3)

MAX_VEHICLES = 30.0
MAX_WAIT     = 120.0   # seconds — used to normalise waiting time
MAX_DURATION = 600.0   # 30 s max green in ticks (600 ticks × 0.05 s)


class DQNAgent:
    def __init__(self,
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=64):

        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.lr            = learning_rate

        self.memory = deque(maxlen=memory_size)

        # Network weights: 12 -> 64 -> 64 -> 4
        self.w1 = np.random.randn(state_size, 64) * 0.1
        self.b1 = np.zeros((1, 64))
        self.w2 = np.random.randn(64, 64) * 0.1
        self.b2 = np.zeros((1, 64))
        self.w3 = np.random.randn(64, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))

        # Target network
        self.tw1 = self.w1.copy()
        self.tb1 = self.b1.copy()
        self.tw2 = self.w2.copy()
        self.tb2 = self.b2.copy()
        self.tw3 = self.w3.copy()
        self.tb3 = self.b3.copy()

        self.update_target_every = 100
        self.step_count          = 0
        self.total_reward        = 0.0
        self.episode_rewards     = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, x, use_target=False):
        if use_target:
            w1,b1,w2,b2,w3,b3 = self.tw1,self.tb1,self.tw2,self.tb2,self.tw3,self.tb3
        else:
            w1,b1,w2,b2,w3,b3 = self.w1,self.b1,self.w2,self.b2,self.w3,self.b3
        h1 = self.relu(x @ w1 + b1)
        h2 = self.relu(h1 @ w2 + b2)
        q  = h2 @ w3 + b3
        return q, h1, h2

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_arr    = np.array(state).reshape(1, -1)
        q_values,_,_ = self.forward(state_arr)
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.total_reward += reward

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch       = random.sample(self.memory, self.batch_size)
        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch])

        q_current,_,_ = self.forward(states)
        q_next,_,_       = self.forward(next_states, use_target=True)
        q_target         = q_current.copy()

        for i in range(self.batch_size):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                q_target[i, actions[i]] = rewards[i] + \
                    self.gamma * np.max(q_next[i])

        loss = self._backprop(states, q_target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target()

        return loss

    def _backprop(self, x, q_target):
        batch        = x.shape[0]
        q_pred,h1_c,h2_c = self.forward(x)

        d_out = (q_pred - q_target) / batch

        dw3 = h2_c.T @ d_out
        db3 = d_out.sum(axis=0, keepdims=True)

        d_h2 = d_out @ self.w3.T * self.relu_grad(h2_c)
        dw2  = h1_c.T @ d_h2
        db2  = d_h2.sum(axis=0, keepdims=True)

        d_h1 = d_h2 @ self.w2.T * self.relu_grad(h1_c)
        dw1  = x.T @ d_h1
        db1  = d_h1.sum(axis=0, keepdims=True)

        clip = 1.0
        for d in [dw1,db1,dw2,db2,dw3,db3]:
            np.clip(d, -clip, clip, out=d)

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3

        return float(np.mean((q_pred - q_target) ** 2))

    def _update_target(self):
        self.tw1=self.w1.copy(); self.tb1=self.b1.copy()
        self.tw2=self.w2.copy(); self.tb2=self.b2.copy()
        self.tw3=self.w3.copy(); self.tb3=self.b3.copy()

    def save(self, path="data/dqn_weights.json"):
        data = {
            'w1': self.w1.tolist(), 'b1': self.b1.tolist(),
            'w2': self.w2.tolist(), 'b2': self.b2.tolist(),
            'w3': self.w3.tolist(), 'b3': self.b3.tolist(),
            'epsilon':        self.epsilon,
            'step_count':     self.step_count,
            'episode_rewards':self.episode_rewards
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved weights → {path}")

    def load(self, path="data/dqn_weights.json"):
        if not os.path.exists(path):
            print(f"  No saved weights at {path} — starting fresh.")
            return
        with open(path) as f:
            data = json.load(f)
        # Guard against stale weights from an older state-size
        saved_input_size = np.array(data['w1']).shape[0]
        if saved_input_size != self.state_size:
            print(f"  Weight size mismatch (saved input={saved_input_size}, "
                  f"current={self.state_size}) — starting fresh.")
            return
        self.w1 = np.array(data['w1']); self.b1 = np.array(data['b1'])
        self.w2 = np.array(data['w2']); self.b2 = np.array(data['b2'])
        self.w3 = np.array(data['w3']); self.b3 = np.array(data['b3'])
        self.epsilon         = data['epsilon']
        self.step_count      = data['step_count']
        self.episode_rewards = data.get('episode_rewards', [])
        self._update_target()
        print(f"  Loaded weights. Epsilon={self.epsilon:.3f}  Steps={self.step_count}  Episodes={len(self.episode_rewards)}")


# ==============================
# STATE BUILDER
# ==============================

def build_state(arm_n_queue, arm_s_queue, arm_e_queue, arm_w_queue,
                arm_n_wait,  arm_s_wait,  arm_e_wait,  arm_w_wait,
                current_phase, phase_counter,
                emergency_flag, elapsed_seconds):
    """
    Build the 12-element normalised state vector fed to the DQN.

    Parameters
    ----------
    arm_*_queue  : int   – vehicles currently waiting (speed < 0.5 m/s) per arm
    arm_*_wait   : float – avg waiting time (seconds) for vehicles on that arm
    current_phase: int   – active green phase (0-3); during transitions this is
                           the phase just ended (agent sees pending decision context)
    phase_counter: int   – ticks spent in the current green phase
    emergency_flag: int  – 1 if emergency vehicle detected in ROI
    elapsed_seconds: float – simulation time used to derive time-of-day
    """
    time_norm = (elapsed_seconds % 600) / 600.0
    return np.array([
        min(arm_n_queue, MAX_VEHICLES) / MAX_VEHICLES,   # North arm queue
        min(arm_s_queue, MAX_VEHICLES) / MAX_VEHICLES,   # South arm queue
        min(arm_e_queue, MAX_VEHICLES) / MAX_VEHICLES,   # East  arm queue
        min(arm_w_queue, MAX_VEHICLES) / MAX_VEHICLES,   # West  arm queue
        min(arm_n_wait,  MAX_WAIT)     / MAX_WAIT,       # North avg wait
        min(arm_s_wait,  MAX_WAIT)     / MAX_WAIT,       # South avg wait
        min(arm_e_wait,  MAX_WAIT)     / MAX_WAIT,       # East  avg wait
        min(arm_w_wait,  MAX_WAIT)     / MAX_WAIT,       # West  avg wait
        current_phase  / 3.0,                             # active phase
        min(phase_counter, MAX_DURATION) / MAX_DURATION, # time in phase
        time_norm,                                        # time of day
        float(emergency_flag),                            # emergency
    ], dtype=np.float32)


# ==============================
# REWARD FUNCTION
#
# Formula:
#   R = - α * waiting_time_penalty
#       - β * queue_length_penalty
#       + γ * vehicles_cleared_bonus
#       + δ * speed_bonus
#       ± emergency_handling
#       - switching_penalty   (penalise unnecessary phase changes)
#
# "Pressure" concept:
#   The agent learns to give GREEN to the arm with highest pressure
#   (long queue + long wait). Per-arm state features enable this.
# ==============================

def compute_reward(avg_speed,
                   emergency_flag,
                   switching,
                   avg_waiting_time=0.0,
                   queue_length=0,
                   vehicles_cleared=0):
    """
    Reward formula (spec-aligned):
        R = - α * waiting_time_penalty
            - β * queue_length_penalty
            + γ * vehicles_cleared_bonus
            + δ * avg_speed_bonus
            + emergency_handling_bonus
            - switching_penalty

    All terms normalised so individual contributions are on similar scales,
    preventing any single factor from dominating Q-value learning.
    """
    MAX_SPEED = 14.0    # m/s — typical urban speed limit

    # α — penalise long average waiting time (primary metric)
    wait_norm  = min(avg_waiting_time, MAX_WAIT) / MAX_WAIT
    reward     = -wait_norm * 6.0

    # β — penalise queue build-up
    queue_norm = min(queue_length, MAX_VEHICLES) / MAX_VEHICLES
    reward    -= queue_norm * 3.0

    # γ — reward vehicles cleared through the intersection
    reward    += min(vehicles_cleared, 15) * 0.4

    # δ — reward free-flowing traffic (speed bonus)
    speed_norm = min(avg_speed, MAX_SPEED) / MAX_SPEED
    reward    += speed_norm * 2.5

    # Emergency handling: bonus when emergency vehicle is in ROI
    # (the agent should not try to override emergency — it's handled externally,
    # but a positive signal confirms the priority state is recognised)
    if emergency_flag == 1:
        reward += 3.0

    # Penalise unnecessary phase switching — stable phases reduce delay spikes
    if switching:
        reward -= 0.5

    return float(reward)


# ==============================
# EPISODE TRACKER
# ==============================

class EpisodeTracker:
    def __init__(self, episode_length=500):
        self.episode_length  = episode_length
        self.episode         = 0
        self.step            = 0
        self.episode_reward  = 0.0
        self.episode_rewards = []
        self.losses          = []

    def update(self, reward, loss=None):
        self.step           += 1
        self.episode_reward += reward
        if loss is not None:
            self.losses.append(loss)

    def is_done(self):
        return self.step >= self.episode_length

    def next_episode(self, agent):
        self.episode        += 1
        self.episode_rewards.append(self.episode_reward)
        agent.episode_rewards.append(self.episode_reward)

        avg10    = np.mean(self.episode_rewards[-10:])
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0

        print(f"  Episode {self.episode:4d} | "
              f"Reward: {self.episode_reward:8.2f} | "
              f"Avg10: {avg10:8.2f} | "
              f"Eps: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")

        self.step           = 0
        self.episode_reward = 0.0
        self.losses         = []

        if self.episode % 10 == 0:
            agent.save(f"data/dqn_weights_int{self.int_id}.json")

        return self.episode

    def set_int_id(self, int_id):
        self.int_id = int_id
