"""Model-Free Episodic Control (MFEC) algorithm implementation.

MFEC is a memory-based reinforcement learning algorithm introduced by
Blundell et al. (2016).  Unlike DQN, which trains a neural network to
*approximate* Q-values, MFEC stores actual experiences directly in memory
and retrieves them at decision time using a k-nearest-neighbour (kNN) lookup.

Key ideas
---------
* **Episodic memory (QEC)**: one table per discrete action, each mapping a
  compact state embedding to the best discounted return ever seen from that
  (state, action) pair.
* **Fixed random encoder**: raw pixel frames are reduced to a low-dimensional
  vector by a fixed, random projection matrix — no gradient updates needed.
* **Optimistic initialisation**: if a (state, action) pair has never been
  seen, the Q-estimate is +∞ so the agent is nudged to explore it first.
* **Max-aggregation**: if a state is revisited, we keep the *highest* return
  ever observed, which prevents a bad episode from erasing earlier knowledge.

Algorithm pseudocode (matches the paper)
-----------------------------------------
Input:
    encoder φ           # fixed embedding function (random projection)
    k                   # number of nearest neighbours
    γ                   # discount factor
    ε                   # exploration probability
    capacity            # max entries per action buffer

Initialise:
    for each action a ∈ A:
        Q_EC[a] ← empty buffer of (key, value) pairs

repeat for each episode:
    observe initial observation o_0
    trajectory ← empty list

    for t = 0, 1, 2, … until episode ends:
        s_t ← φ(o_t)                          # encode observation

        if random() < ε:
            a_t ← random action from A
        else:
            for each action a ∈ A:
                Q̂(s_t, a) ← EstimateQ(s_t, a)
            a_t ← argmax_a Q̂(s_t, a)

        execute a_t, observe r_{t+1} and next observation o_{t+1}
        append (s_t, a_t, r_{t+1}) to trajectory

    # Backward Monte Carlo return computation
    G ← 0
    for t = T-1 down to 0:
        G ← r_{t+1} + γ · G                   # discounted return from step t
        (s_t, a_t, _) ← trajectory[t]
        UpdateMemory(s_t, a_t, G)

function EstimateQ(s, a):
    if s is exactly in Q_EC[a]:
        return Q_EC[a][s]                      # exact hit: return stored value
    elif |Q_EC[a]| < k:
        return +∞                              # too few entries → optimistic
    else:
        neighbours ← k nearest keys to s in Q_EC[a]
        return (1/k) · Σ Q_EC[a][s_i] for s_i in neighbours

function UpdateMemory(s, a, G, time):
    if s is in Q_EC[a]:
        Q_EC[a][s] ← max(Q_EC[a][s], G)       # keep best return seen
    else:
        if |Q_EC[a]| ≥ capacity:
            evict least-recently-used entry from Q_EC[a]
        insert (s, G) into Q_EC[a]
"""

from __future__ import annotations
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


#
# Main algorithm class
#

class MFECAlgorithm(BaseAlgorithm):
    """Model-Free Episodic Control.

    Implements the BaseAlgorithm interface expected by StepTrainer.
    The trainer calls setup() once, then step() repeatedly with collected
    batches.  This algorithm performs no gradient updates; the only
    "learning" is inserting experiences into the QEC memory tables.
    """

    def __init__(
            self,
            device: torch.device | None = None,
            *,
            # Key used to retrieve the pixel observation from a TensorDict.
            obs_key: str = "pixels",

            #   Episodic memory
            # Maximum number of (state, return) entries stored per action.
            # Older entries are evicted when this is reached.
            buffer_size: int = 1_000_000,
            # Number of nearest neighbours to average when estimating a Q-value for an unseen state.
            k: int = 11,
            # Dimensionality of the projected state embedding.
            state_dim: int = 64,

            #   Discount γ
            gamma: float = 0.99,

            #   eps-greedy exploration schedule
            eps_start: float = 1.0,
            eps_end: float = 0.05,
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
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.annealing_frames = annealing_frames
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj

        self._collected_frames = 0

    #
    # Setup — called once before training starts
    #

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)

        sample_shape = obs_shape[-3:]
        obs_flat_dim = int(np.prod(sample_shape))
        self.projection = np.random.randn(obs_flat_dim, self.state_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

        # All buffer tensors live on the training device (GPU if available).
        self._buffer_device = (
            self.device if self.device is not None else torch.device("cpu")
        )

        self.qec = QEC(range(num_actions), self.buffer_size, self.k, self._buffer_device)
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

        self._explore_policy = _SharedPolicy(self.q_actor, self.greedy_module)

    #
    #   Training step — no gradient updates, memory-only update
    #

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one batch of collected transitions and update QEC memory.

        MFEC has no neural-network parameters to optimise.  The update is:
          1. Compute discounted Monte Carlo returns for each episode in the batch.
          2. Insert (state_embedding, return) pairs into the per-action QEC buffers.

        Parameters
        ----------
        batch : TensorDict
            Transitions from SyncDataCollector.  May span multiple episodes.

        Returns
        -------
        dict
            Scalar metrics logged by the trainer (epsilon, average buffer fill).
        """

        batch = batch.reshape(-1)
        n = batch.numel()

        self.greedy_module.step(n)
        self._collected_frames += n

        obs        = batch[self.obs_key]
        actions_np = batch["action"].cpu().numpy().flatten().astype(int)
        rewards_np = batch["next", "reward"].cpu().numpy().flatten().astype(np.float64)
        dones_np   = batch["next", "done"].cpu().numpy().flatten().astype(bool)

        # Embed all observations in one vectorised GPU call → stays on device.
        states_gpu = self.qec_policy.embed(obs)   # (n, state_dim)

        # Discounted Monte Carlo returns — scipy IIR is fast on CPU.
        G_all = np.empty(n, dtype=np.float64)
        ends = np.flatnonzero(dones_np)
        if len(ends) == 0 or ends[-1] != n - 1:
            ends = np.append(ends, n - 1)
        ep_start = 0
        for ep_end in ends:
            r = rewards_np[ep_start : ep_end + 1]
            G_all[ep_start : ep_end + 1] = lfilter([1.0], [1.0, -self.gamma], r[::-1])[::-1]
            ep_start = ep_end + 1

        dev = self._buffer_device
        G_gpu = torch.tensor(G_all, dtype=torch.float64, device=dev)

        if states_gpu.device != dev:
            states_gpu = states_gpu.to(dev)

        for action in range(self._num_actions):
            mask_np = actions_np == action
            if not mask_np.any():
                continue

            mask_t     = torch.from_numpy(mask_np).to(dev)
            act_states = states_gpu[mask_t]   # (m, state_dim)
            act_values = G_gpu[mask_t]        # (m,)

            buf = self.qec.buffers[action]

            if buf._size > 0:
                # One batched torch.cdist call for all m states of this action.
                dists, nn_idx = buf.knn_batch(act_states, 1)   # (m, 1) each
                exact = dists[:, 0] < 1e-5                      # (m,) bool

                if exact.any():
                    ex_slots = nn_idx[:, 0][exact]
                    # Max-aggregation in-place; duplicate slots (rare) overwrite, acceptable.
                    buf.values[ex_slots] = torch.maximum(buf.values[ex_slots], act_values[exact])

                new_mask = ~exact
                if new_mask.any():
                    buf.add_batch(act_states[new_mask], act_values[new_mask])
            else:
                buf.add_batch(act_states, act_values)

        return {
            "train/epsilon": float(self.greedy_module.eps),
            "train/qec_size": float(np.mean([len(b) for b in self.qec.buffers])),
        }

    #
    #   Policy access — called by StepTrainer
    #

    def get_policy(self) -> TensorDictModule:
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    #
    #   Checkpointing
    #

    def _get_training_state(self) -> TrainingState:
        """Serialise projection matrix and QEC buffer contents as CPU numpy arrays."""
        return TrainingState(
            step=0,
            policy_state_dict={},
            optimizer_state_dict={},
            extra={
                "projection": self.projection,
                "qec_state": [b.__getstate__() for b in self.qec.buffers],
                "collected_frames": self._collected_frames,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        """Restore algorithm state from a checkpoint."""
        self.projection = state.extra["projection"]
        self.qec_policy.projection = self.projection
        self.qec_policy._proj_tensor = None
        self._collected_frames = int(state.extra["collected_frames"])

        for buf, saved in zip(self.qec.buffers, state.extra["qec_state"]):
            buf.__setstate__(saved)


# ---------------------------------------------------------------------------
# Policy module — converts observations to Q-value vectors
# ---------------------------------------------------------------------------

class QECPolicy(nn.Module):
    """Non-parametric nn.Module that estimates Q-values via kNN memory lookup.

    embed() and forward() are split so that MFECAlgorithm.step() can call
    embed() on its own (to get embeddings for the whole batch efficiently)
    without going through the full forward pass.
    """

    def __init__(self, qec: "QEC", projection: np.ndarray, num_actions: int) -> None:
        super().__init__()
        self.qec = qec
        self.projection = projection
        self.num_actions = num_actions
        self._proj_tensor: torch.Tensor | None = None

    def __deepcopy__(self, memo):
        # TorchRL deepcopies the policy when setting up the collector.  A full
        # deepcopy would give the collector its own isolated, empty QEC that
        # never receives updates from step().  Share self.qec by reference so
        # the collector always reads the latest memory.
        cls = self.__class__
        copy = cls.__new__(cls)
        memo[id(self)] = copy
        super(QECPolicy, copy).__init__()
        copy.qec = self.qec
        copy.projection = self.projection
        copy.num_actions = self.num_actions
        copy._proj_tensor = None
        return copy

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Project observation to a (batch, state_dim) tensor; result stays on obs.device."""
        if self._proj_tensor is None or self._proj_tensor.device != obs.device:
            self._proj_tensor = torch.tensor(
                self.projection, dtype=torch.float32, device=obs.device
            )
        flat = obs.float().reshape(-1, self._proj_tensor.shape[0])
        return flat @ self._proj_tensor   # (batch, state_dim) — no CPU transfer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        states = self.embed(obs)   # (batch, state_dim) on obs.device

        results = list(self.qec.executor.map(
            lambda a: self.qec.estimate_batch(states, a),
            range(self.num_actions),
        ))
        # results: list of (batch,) tensors → (num_actions, batch) → T → (batch, num_actions)
        q_values = torch.stack(results, dim=0).T
        q_values = torch.where(torch.isinf(q_values), torch.full_like(q_values, 1e9), q_values)

        leading = obs.shape[:-3]
        if leading:
            return q_values.reshape(*leading, self.num_actions)
        return q_values.squeeze(0) if q_values.shape[0] == 1 else q_values


class _SharedPolicy(TensorDictSequential):
    """TensorDictSequential that returns itself on deepcopy.

    TorchRL deepcopies the explore policy when setting up the collector.
    A full deepcopy would give the collector a fresh EGreedyModule with
    eps=eps_init that never advances — so the collector always acts randomly.
    Returning self ensures the single EGreedyModule is shared and advanced
    correctly by algorithm.step().
    """

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

class QEC:
    """Q-value Episodic Controller.

    Holds one ActionBuffer per discrete action and exposes the two core
    operations from the MFEC paper: estimate_batch() and the buffer mutation
    primitives (called directly from MFECAlgorithm.step()).

    Parameters
    ----------
    actions     : iterable of action indices (e.g. range(num_actions))
    buffer_size : maximum entries per ActionBuffer before LRU eviction
    k           : number of nearest neighbours for Q-value estimation
    device      : torch.device for all buffer tensors
    """

    def __init__(
        self, actions, buffer_size: int, k: int, device: torch.device
    ) -> None:
        self.buffers = tuple(ActionBuffer(buffer_size, device) for _ in actions)
        self.k = k
        self.executor = ThreadPoolExecutor(max_workers=len(self.buffers))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["executor"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.executor = ThreadPoolExecutor(max_workers=len(self.buffers))

    @torch.no_grad()
    def estimate_batch(self, states: torch.Tensor, action: int) -> torch.Tensor:
        """Estimate Q-values for a batch of states via kNN on GPU tensors.

        Parameters
        ----------
        states : (batch, state_dim) float32 tensor — may be on any device
        action : int

        Returns
        -------
        (batch,) float32 tensor on states.device — inf for states with too
        few stored neighbours (optimistic initialisation)
        """
        buffer = self.buffers[action]
        n   = len(states)
        dev = states.device

        if buffer._size <= self.k:
            return torch.full((n,), float("inf"), dtype=torch.float32, device=dev)

        k_fetch = min(self.k + 1, buffer._size)
        dists, indices = buffer.knn_batch(states, k_fetch)   # (n, k_fetch)

        # Exact match: nearest stored vector is within floating-point tolerance.
        exact = dists[:, 0] < 1e-5   # (n,)

        # kNN average for all rows (exact rows overridden below).
        k_use    = min(self.k, k_fetch)
        knn_idx  = indices[:, :k_use].to(buffer.device)               # (n, k_use)
        knn_vals = buffer.values[knn_idx].to(torch.float32).to(dev)   # (n, k_use)
        knn_avg  = knn_vals.mean(dim=1)                                # (n,)

        exact_vals = buffer.values[indices[:, 0].to(buffer.device)].to(torch.float32).to(dev)
        return torch.where(exact, exact_vals, knn_avg)


class ActionBuffer:
    """Fixed-capacity nearest-neighbour memory for a single action.

    All data lives in two persistent GPU tensors (states, values).
    kNN queries use batched torch.cdist (exact L2, fully GPU-parallelised).
    Eviction is FIFO via a ring buffer — a single modular write pointer,
    no LRU heap or times tensor needed.

    Parameters
    ----------
    capacity : int — maximum number of (state, return) entries to store
    device   : torch.device — device for all tensors (CPU or CUDA)
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity   = capacity
        self.device     = device
        self._size:      int = 0   # grows to capacity, then stays there
        self._write_ptr: int = 0   # next slot to write (ring buffer)
        # states is lazily allocated on the first add_batch() call when
        # state_dim becomes known.
        self.states: torch.Tensor | None = None          # (capacity, state_dim) float32
        self.values: torch.Tensor = torch.empty(capacity, dtype=torch.float64, device=device)

    # ------------------------------------------------------------------
    # kNN lookup
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_batch(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched exact kNN via torch.cdist.

        Parameters
        ----------
        queries : (batch, state_dim) — any device (transferred internally if needed)
        k       : number of neighbours to return

        Returns
        -------
        dists   : (batch, k_eff) L2 distances — on self.device
        indices : (batch, k_eff) int64 indices into self.states — on self.device
        """
        if queries.device != self.device:
            queries = queries.to(self.device)
        size  = self._size
        k_eff = min(k, size)
        # torch.cdist with GEMM-based Euclidean distance: (batch, size)
        dists = torch.cdist(queries, self.states[:size], p=2)
        top_dists, top_idx = dists.topk(k_eff, dim=1, largest=False)
        return top_dists, top_idx

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_batch(
        self,
        states: torch.Tensor,   # (n, state_dim) float32
        values: torch.Tensor,   # (n,) float64
    ) -> None:
        """Insert a batch of (state, value) entries via ring-buffer scatter.

        When n ≥ capacity, only the last `capacity` entries are kept (earlier
        ones would be immediately overwritten anyway).  The write pointer
        advances modulo capacity so no eviction decision is ever needed.
        """
        n = len(states)
        if n == 0:
            return

        if states.device != self.device:
            states = states.to(self.device)
            values = values.to(self.device)

        if self.states is None:
            self.states = torch.empty(
                self.capacity, states.shape[1], dtype=torch.float32, device=self.device
            )

        # If the batch itself exceeds capacity, only the newest entries matter.
        if n > self.capacity:
            states = states[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity

        # Destination slots via modular arithmetic — single GPU op.
        slots = torch.arange(self._write_ptr, self._write_ptr + n, device=self.device) % self.capacity
        self.states[slots] = states
        self.values[slots] = values

        self._write_ptr = int((self._write_ptr + n) % self.capacity)
        self._size      = min(self._size + n, self.capacity)

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        size = self._size
        # When the buffer has wrapped (size == capacity), save all slots in
        # write-order so __setstate__ can restore the ring without gaps.
        if size == self.capacity and self.states is not None:
            # Rotate so slot[0] = oldest entry (the one write_ptr points at).
            ptr = self._write_ptr
            states_np = torch.roll(self.states, -ptr, dims=0).cpu().numpy()
            values_np = torch.roll(self.values, -ptr, dims=0).cpu().numpy()
        else:
            states_np = self.states[:size].cpu().numpy() if self.states is not None and size > 0 else None
            values_np = self.values[:size].cpu().numpy() if size > 0 else np.array([], dtype=np.float64)
        return {
            "capacity":   self.capacity,
            "device":     self.device,
            "_size":      size,
            "_write_ptr": self._write_ptr,
            "states":     states_np,
            "values":     values_np,
        }

    def __setstate__(self, d: dict) -> None:
        self.capacity   = d["capacity"]
        self.device     = d["device"]
        self._size      = d["_size"]
        self._write_ptr = d["_write_ptr"]
        dev  = self.device
        size = self._size
        self.values = torch.empty(self.capacity, dtype=torch.float64, device=dev)
        if size > 0:
            self.values[:size] = torch.from_numpy(d["values"]).to(dev)
        saved = d["states"]
        if saved is not None and size > 0:
            self.states = torch.empty(self.capacity, saved.shape[1], dtype=torch.float32, device=dev)
            self.states[:size] = torch.from_numpy(saved).to(dev)
        else:
            self.states = None
