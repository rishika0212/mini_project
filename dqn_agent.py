import numpy as np
import random
from collections import deque
import json
import os

# ==============================
# DEEP Q-NETWORK AGENT
# ==============================

# STATE (7 features):
#   [0-3] pN, pS, pE, pW  — per-arm pressures, max-normalised to [0,1]
#   [4]   current_arm      — index of green arm (0-3), normalised /3
#   [5]   time_in_phase    — ticks in current green phase, normalised
#   [6]   emergency_flag   — 1 if emergency vehicle in ROI

# ACTION (2):
#   0 = keep current phase (do not switch)
#   1 = allow switch (rule-based controller may switch to best-pressure arm)
#
# DQN assists the pressure-based controller by deciding WHEN to switch,
# not WHICH arm to switch to. The rule-based controller enforces all
# safety invariants (min/max green time, pressure margin).

STATE_SIZE  = 7
ACTION_SIZE = 2   # 0=keep, 1=allow-switch

MAX_PRESSURE = 60.0    # normalisation ceiling for pressure values
MAX_DURATION = 1200.0  # MAX_GREEN_TICKS from controller.py


class DQNAgent:
    def __init__(self,
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.9995,
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

        # Network weights: 7 -> 64 -> 64 -> 2
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
        print(f"  Saved weights -> {path}")

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

_ARMS = ['N', 'S', 'E', 'W']

def build_state(pressures: dict, current_arm: str, ticks_in_phase: int,
                emergency_flag: int) -> np.ndarray:
    """
    Build the 7-element normalised state vector for the DQN.

    Parameters
    ----------
    pressures      : dict  – {'N': float, 'S': float, 'E': float, 'W': float}
                             pressure = queue + 1.5 * avg_wait (from controller.py)
    current_arm    : str   – arm currently holding GREEN ('N','S','E','W')
    ticks_in_phase : int   – ticks spent in current green phase
    emergency_flag : int   – 1 if emergency vehicle detected in ROI
    """
    pN = pressures.get('N', 0.0)
    pS = pressures.get('S', 0.0)
    pE = pressures.get('E', 0.0)
    pW = pressures.get('W', 0.0)
    max_p = max(pN, pS, pE, pW, 1.0)   # relative normalisation
    arm_idx = _ARMS.index(current_arm) if current_arm in _ARMS else 0
    return np.array([
        pN / max_p,
        pS / max_p,
        pE / max_p,
        pW / max_p,
        arm_idx / 3.0,
        min(ticks_in_phase, MAX_DURATION) / MAX_DURATION,
        float(emergency_flag),
    ], dtype=np.float32)


# ==============================
# REWARD FUNCTION
#
# Formula (all terms are costs, except vehicles_cleared which is a benefit):
#   R = - 5 * norm(avg_waiting_time)
#       - 3 * norm(queue_length)
#       + 2 * norm(vehicles_cleared)
#       - 3 * wrong_lane_selection   (1 if serving non-max-pressure arm)
#       - 1 * unnecessary_switch     (1 if switched without pressure advantage)
#
# This strongly penalises green for empty lanes and rewards clearing congestion.
# ==============================

def compute_reward(avg_waiting_time: float = 0.0,
                   queue_length: int = 0,
                   vehicles_cleared: float = 0.0,
                   wrong_lane_selection: int = 0,
                   unnecessary_switch: int = 0) -> float:
    """
    Reward shaped to penalise wrong arm selection and reward clearing congestion.

    wrong_lane_selection: 1 if the currently-serving arm pressure is below
                          the max-pressure arm (i.e. wrong arm is green).
    unnecessary_switch:   1 if a switch occurred but the new arm had no
                          meaningful pressure advantage.
    """
    MAX_WAIT_S  = 120.0
    MAX_QUEUE   = 30.0
    MAX_CLEARED = 10.0

    wait_norm    = min(avg_waiting_time, MAX_WAIT_S) / MAX_WAIT_S
    queue_norm   = min(queue_length,     MAX_QUEUE)  / MAX_QUEUE
    cleared_norm = min(vehicles_cleared, MAX_CLEARED) / MAX_CLEARED

    reward  = -5.0 * wait_norm
    reward -= 3.0 * queue_norm
    reward += 2.0 * cleared_norm
    reward -= 3.0 * float(wrong_lane_selection)
    reward -= 1.0 * float(unnecessary_switch)
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
