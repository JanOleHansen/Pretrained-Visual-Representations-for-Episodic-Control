""" Model Free Episodic Control (MFEC) algorithm implementation.



Algorithm: Model-Free Episodic Control (MFEC)

Input:
    encoder φ           # fixed embedding function (random projection or VAE)
    k                   # number of nearest neighbours
    γ                   # discount factor
    ε                   # exploration probability
    capacity            # max entries per action buffer

Initialise:
    for each action a ∈ A:
        Q_EC[a] ← empty buffer of (key, value) pairs

# ─── Main training loop ─────────────────────────────────────────
repeat for each episode:
    observe initial observation o_0
    trajectory ← empty list

    # ─── Episode rollout ────────────────────────────────────────
    for t = 0, 1, 2, ... until episode ends:
        s_t ← φ(o_t)                          # encode observation

        # ε-greedy action selection
        if random() < ε:
            a_t ← random action from A
        else:
            for each action a ∈ A:
                Q̂(s_t, a) ← EstimateQ(s_t, a)
            a_t ← argmax_a Q̂(s_t, a)

        execute a_t, observe r_{t+1} and next observation o_{t+1}
        append (s_t, a_t, r_{t+1}) to trajectory

    # ─── Backward Monte Carlo return computation ───────────────
    G ← 0
    for t = T-1 down to 0:                    # walk episode in reverse
        G ← r_{t+1} + γ · G                   # discounted return from step t
        (s_t, a_t, _) ← trajectory[t]
        UpdateMemory(s_t, a_t, G)


# ─── Helper: Q-value estimation via kNN ────────────────────────
function EstimateQ(s, a):
    if (s, a) is exactly in Q_EC[a]:
        return Q_EC[a][s]                     # exact match: return stored value
    else:
        neighbours ← k nearest keys to s in Q_EC[a]    # kNN search
        return (1/k) · Σ Q_EC[a][s_i] for s_i in neighbours


# ─── Helper: Memory update with max aggregation ────────────────
function UpdateMemory(s, a, G):
    if (s, a) is in Q_EC[a]:
        Q_EC[a][s] ← max(Q_EC[a][s], G)       # keep best return seen
    else:
        if |Q_EC[a]| ≥ capacity:
            evict least-recently-used entry from Q_EC[a]
        insert (s, G) into Q_EC[a]


"""

from __future__ import annotations
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor

from sklearn.neighbors import KDTree



from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState

#
#   Main algorithm class
#

class MFECAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            device: torch.device | None = None,
            *,
            obs_key: str = "pixels",
            #   Episodic memory parameters
            buffer_size: int = 1_000_000,
            k: int = 11,
            state_dim: int = 64,
            #   Discount
            gamma: float = 0.99,
            #   Exploration
            esp_start: float = 1.0,
            esp_end: float = 0.05,
            annealing_frames: int = 1_000_000,
            #   Data collection
            frames_per_batch: int = 1_000,
            max_frames_per_traj: int = -1,
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.buffer_size = buffer_size
        self.k = k
        self.state_dim = state_dim
        self.gamma = gamma
        self.esp_start = esp_start
        self.esp_end = esp_end
        self.annealing_frames = annealing_frames
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self._collected_frames = 0

        #
        #   Setup
        #

        def setup(self, make_env: Callable[[], EnvBase]) -> None:
            proof_env = make_env()
            obs_shape = tuple(proof_env(proof_env.observation_spec[self.obs_key]).shape)

            action_spec = proof_env.action_spec
            num_actions = int(action_spec.space.n)
            proof_env.close()

            # Fixed random projection: [H*W, state_dim], never updated.
            obs_flat_dim = int(np.prod(obs_shape[1:]))  # H * W after channel-averaging
            self.projection = np.random.randn(obs_flat_dim, self.state_dim)
            self.projection /= np.linalg.norm(self.projection, axis=0)

            self.qec = QEC(range(num_actions), self.buffer_size, self.k)
            self._num_actions = num_actions

            self.qec_policy = QECPolicy(self.qec, self.projection, num_actions)
            self.q_actor = QValueActor(
                module=self.qec_policy,
                spec=action_spec,
                in_keys=[self.obs_key],
            )

            self.greedy_module = EGreedyModule(
                spec=action_spec,
                eps_init=self.eps_start,
                eps_end=self.eps_end,
                annealing_num_steps=self.annealing_frames,
            )
            self._explore_policy = TensorDictSequential(self.q_actor, self.greedy_module)

        #
        #   Training step: no gradient updates, just memory updates.
        #

        def step(self, batch: TensorDict) -> dict[str, float]:
            batch = batch.reshape(-1)  # flatten batch and time dimensions
            n = batch.numel()
            self.greedy_module.step(n)  # update epsilon schedule
            self._collected_frames += n

            obs = batch[self.obs_key]
            actions_np = batch["action"].cpu().numpy().flatten().astype(int)
            rewards_np = batch["next","reward"].cpu().numpy().flatten().astype(np.float64)
            dones_np = batch["next","done"].cpu().numpy().flatten().astype(bool)

            states = self.qec_policy.embed(obs)

            # Split into episodes on done boundaries; compute MC returns; update QEC.
            episode_start = 0
            for t in range(n):
                if dones_np[t] or t == n - 1:
                    ep = slice(episode_start, t + 1)
                    ep_states = states[ep]
                    ep_actions = actions_np[ep]
                    ep_rewards = rewards_np[ep]
                    ep_len = len(ep_rewards)

                    G = np.zeros(ep_len)
                    G[-1] = ep_rewards[-1]
                    for i in range(ep_len - 2, -1, -1):
                        G[i] = ep_rewards[i] + self.gamma * G[i + 1]

                    frame_time = self._collected_frames - n + episode_start
                    for i in range(ep_len):
                        self.qec.update(ep_states[i], ep_actions[i], G[i], frame_time + i)

                    episode_start = t + 1

            return {
                "train/epsilon": float(self.greedy_module.eps),
                "train/qec_size": float(np.mean([len(b) for b in self.qec.buffers])),
            }
        
        #
        #   Policy access
        #

        def get_policy(self) -> TensorDictModule:
            return self.q_actor
        
        def get_explore_policy(self) -> TensorDictModule:
            return self._explore_policy
        
        def get_collector_config(self) -> CollectorConfig:
            return CollectorConfig(
                frames_per_batch=self.frames_per_batch,
                init_random_frames=0,                                   # no random actions; exploration via ε-greedy only
                max_frames_per_traj=self.max_frames_per_traj,
            )
        

        # 
        # Checkpointing
        # 

        def _get_training_state(self) -> TrainingState:
            return TrainingState(
                step=0,
                policy_state_dict={},
                optimizer_state_dict={},
                extra={
                    "projection": self.projection,
                    "qec_state": [
                        {"states": b.states, "values": b.values, "times": b.times}
                        for b in self.qec.buffers
                    ],
                    "collected_frames": self._collected_frames,
                },
            )

        def _load_training_state(self, state: TrainingState) -> None:
            self.projection = state.extra["projection"]
            self.qec_policy.projection = self.projection
            self._collected_frames = int(state.extra["collected_frames"])
            for buf, saved in zip(self.qec.buffers, state.extra["qec_state"]):
                buf.states = saved["states"]
                buf.values = saved["values"]
                buf.times = saved["times"]
                if buf.states:
                    buf._tree = KDTree(buf.states)


#
#   Policy module
#

class QECPolicy(nn.Module):
    def __init__(self, qec: QEC, projection: np.ndarray, num_actions: int) -> None:
        super().__init__()
        self.qec = qec
        self.projection = projection
        self.num_actions = num_actions

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] float tensor -> [B, state_dim] numpy array."""
        obs_np = obs.cpu().numpy()
        flat = obs_np.mean(axis=1).reshape(len(obs_np) -1) # average channels, flatten
        
        return flat @ self.projection
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        states = self.embed(obs)
        q_values = np.array(
            [[self.qec.estimate(state, action) for action in range(self.num_actions)] for state in states],
            dtype=np.float32,
        )
        q = np.where(np.isinf(q), 1e9, q) # inf -> large finite for argmax stability
        return torch.tensor(q, dtype=torch.float32, device=obs.device)

#
#   Episodic memory
#

class QEC:
    def __init__ (self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index is not None:
            return buffer.values[state_index]
        if len(buffer) <= self.k:
            return float('inf')
        
        value = 0.0
        neighbours = buffer.find_k_nearest(state, self.k)
        for neighbour in neighbours:
            value += buffer.values[neighbour]
        return value / self.k
    
    def update(self, state, action, return_):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index is not None:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)

        else:
            buffer.add(state, value, time)


class ActionBuffer:
    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state):
        if self._tree is not None:
            neighbour_index = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbour_index, state]):
                return neighbour_index
        return None
    
    def find_k_nearest(self, state, k):
        return self._tree.query([state])[1][0] if self._tree is not None else []
    
    def add(self, state, value, time):
        if len(self) < self.capacity:
            sekf.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_index = int(np.argmin(self.times))
            if time > self.times[min_time_index]:
                self.replace(state, value, time, min_time_index)
        self._tree = KDTree(self.states)


    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time
    def __len__(self):
        return len(self.states)