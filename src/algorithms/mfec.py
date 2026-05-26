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
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor
import faiss

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
            # Older entries are evicted when max this is reached.
            buffer_size: int = 1_000_000,
            # Number of nearest neighbours to average when estimating a Q-value for an unseen state.  Higher k → smoother estimates, higher cost.
            k: int = 11,
            # Dimensionality of the projected state embedding.  Lower values reduce memory and kNN cost; higher values preserve more structure.
            state_dim: int = 64,

            #   Discount γ — future rewards are worth γ^t times a reward t steps away.
            gamma: float = 0.99,

            #   eps-greedy exploration schedule
            # At the start of training, every action is random (eps_start=1.0). eps is linearly annealed to eps_end over annealing_frames steps.
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            annealing_frames: int = 1_000_000,

            #   Data collection
            # Number of environment steps collected before each call to step().
            frames_per_batch: int = 1_000,
            # Hard cap on episode length; -1 means no limit.
            max_frames_per_traj: int = -1,
    ) -> None:
        
        super().__init__(device)

        # Store all hyperparameters as instance attributes so setup() and step() can access them.
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

        # Running count of all environment frames seen so far.  Used as an LRU timestamp when inserting entries into the QEC buffers.
        self._collected_frames = 0

    # 
    # Setup — called once before training starts
    # 

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)

        # Strip any leading ParallelEnv batch dims — keep only (C, H, W)
        sample_shape = obs_shape[-3:]
        obs_flat_dim = int(np.prod(sample_shape))
        self.projection = np.random.randn(obs_flat_dim, self.state_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

        if self.device is not None and self.device.type == "cuda":
            gpu_id = self.device.index if self.device.index is not None else 0
        else:
            gpu_id = -1

        self.qec = QEC(range(num_actions), self.buffer_size, self.k, gpu_id=gpu_id)
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
    #   Training step — no gradient updates, memory-only update
    #

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one batch of collected transitions and update QEC memory.

        MFEC has no neural-network parameters to optimise.  The update is:
          1. Compute discounted Monte Carlo returns for each episode in the batch.
          2. Insert (state_embedding, return) pairs into the per-action QEC buffers.

        The trainer may call this with data from partial episodes (the batch
        boundary does not coincide with episode boundaries).  We detect episode
        ends via the "done" flag and process each episode independently.

        Parameters
        ----------
        batch : TensorDict
            Transitions from SyncDataCollector.  May span multiple episodes.

        Returns
        -------
        dict
            Scalar metrics logged by the trainer (epsilon, average buffer fill).
        """

        # Flatten any leading batch/time axes so we have a single sequence of transitions indexed by t = 0 … n-1.
        batch = batch.reshape(-1)
        n = batch.numel()  # total number of transitions

        # Advance the ε-greedy annealing schedule by n steps.
        self.greedy_module.step(n)
        self._collected_frames += n

        # Extract arrays from the TensorDict.  We work in numpy because the QEC memory and FAISS index are numpy objects.
        obs        = batch[self.obs_key]                                           # [n, C, H, W]
        actions_np = batch["action"].cpu().numpy().flatten().astype(int)           # [n]
        rewards_np = batch["next", "reward"].cpu().numpy().flatten().astype(np.float64)  # [n]
        dones_np   = batch["next", "done"].cpu().numpy().flatten().astype(bool)    # [n]

        # Embed all n observations in a single vectorised call to avoid the overhead of calling embed() inside the per-step loop.
        states = self.qec_policy.embed(obs)  # (n, state_dim) numpy array

        #   Episode segmentation and Monte Carlo return computation
        # Walk through the batch and split it at "done" boundaries.  For each
        # completed episode we compute the discounted return G_t for every step
        # and write the (state, return) pairs into QEC.
        episode_start = 0
        for t in range(n):
            # A "done" flag marks the *last* transition of an episode. We also flush the trailing partial episode when we reach the end
            # of the batch (t == n - 1), even if no done flag was set — the returns are still valid; the episode just continues in the next batch.
            if dones_np[t] or t == n - 1:
                ep          = slice(episode_start, t + 1)
                ep_states   = states[ep]      # (ep_len, state_dim)
                ep_actions  = actions_np[ep]  # (ep_len,)
                ep_rewards  = rewards_np[ep]  # (ep_len,)
                ep_len      = len(ep_rewards)

                # Compute discounted Monte Carlo returns by walking backward:
                #   G[T-1] = r[T-1]
                #   G[t]   = r[t] + γ · G[t+1]
                # This gives the total discounted return from each step t to the end of the episode, which is what MFEC stores in QEC.
                G = np.zeros(ep_len)
                G[-1] = ep_rewards[-1]
                for i in range(ep_len - 2, -1, -1):
                    G[i] = ep_rewards[i] + self.gamma * G[i + 1]

                # Absolute frame index of the first step in this episode. Used as an LRU timestamp so older entries are evicted first.
                frame_time = self._collected_frames - n + episode_start
                for i in range(ep_len):
                    self.qec.update(
                        ep_states[i],    # state embedding
                        ep_actions[i],   # action taken
                        G[i],            # discounted return from this step
                        frame_time + i,  # LRU timestamp
                    )

                episode_start = t + 1  # next episode begins after this done flag

        self.qec.rebuild_trees()  # one rebuild per step() call instead of per insert
        return {
            # Current ε value — tells us how often the agent acts randomly.
            "train/epsilon": float(self.greedy_module.eps),
            # Mean number of entries stored across all per-action buffers. Grows toward buffer_size, then stays flat once memory is full.
            "train/qec_size": float(np.mean([len(b) for b in self.qec.buffers])),
        }

    #
    #   Policy access — called by StepTrainer
    #

    def get_policy(self) -> TensorDictModule:
        """Return the greedy (deterministic) policy used at evaluation time."""
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        """Return the ε-greedy policy used during data collection."""
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        """Tell StepTrainer how to configure SyncDataCollector."""
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    #
    #   Checkpointing
    #

    def _get_training_state(self) -> TrainingState:
        """Serialise everything needed to resume training from a checkpoint.

        Because MFEC has no neural-network weights, the state is just the
        random projection matrix and the full contents of every ActionBuffer.
        """
        return TrainingState(
            step=0,
            policy_state_dict={},     # no learnable parameters
            optimizer_state_dict={},  # no optimiser
            extra={
                "projection": self.projection,
                # Each ActionBuffer stores three parallel lists.  We save all
                # three so the kNN tree can be rebuilt exactly after loading.
                "qec_state": [
                    {"states": b.states, "values": b.values, "times": b.times}
                    for b in self.qec.buffers
                ],
                "collected_frames": self._collected_frames,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        """Restore algorithm state from a checkpoint created by _get_training_state."""
        self.projection = state.extra["projection"]
        # Update the reference inside QECPolicy so it uses the restored matrix,
        # not the one initialised in setup().
        self.qec_policy.projection = self.projection
        self.qec_policy._proj_tensor = None  # force re-cache on next embed()
        self._collected_frames = int(state.extra["collected_frames"])

        for buf, saved in zip(self.qec.buffers, state.extra["qec_state"]):
            buf.states = saved["states"]
            buf.values = saved["values"]
            buf.times  = saved["times"]
            buf._index = None
            buf._index_size = 0
            buf._needs_rebuild = False
            # Rebuild the FAISS index so kNN lookups work immediately after loading.
            buf.sync_index()


# ---------------------------------------------------------------------------
# Policy module — converts observations to Q-value vectors
# ---------------------------------------------------------------------------

class QECPolicy(nn.Module):
    """Non-parametric nn.Module that estimates Q-values via kNN memory lookup.

    This module holds no learnable weights.  It wraps the QEC memory so that
    QValueActor can call it through TorchRL's TensorDict interface.

    embed() and forward() are split so that MFECAlgorithm.step() can call
    embed() on its own (to get embeddings for the whole batch efficiently)
    without going through the full forward pass.
    """

    def __init__(self, qec: "QEC", projection: np.ndarray, num_actions: int) -> None:
        super().__init__()
        self.qec = qec
        self.projection = projection   # fixed (H*W, state_dim) Gaussian matrix
        self.num_actions = num_actions
        self._proj_tensor: torch.Tensor | None = None  # cached on obs device

    def embed(self, obs: torch.Tensor) -> np.ndarray:
        # Keep the matmul on the same device as obs so we only transfer the
        # compact 64-dim result to CPU rather than the full pixel observation
        # (4×84×84 ≈ 112 KB per env vs 256 bytes for the embedding).
        if self._proj_tensor is None or self._proj_tensor.device != obs.device:
            self._proj_tensor = torch.tensor(
                self.projection, dtype=torch.float32, device=obs.device
            )
        flat = obs.float().reshape(-1, self._proj_tensor.shape[0])
        return (flat @ self._proj_tensor).cpu().numpy()

    def forward(self, obs):
        states = self.embed(obs)

        # One FAISS search call per action (batched over all states in obs).
        results = list(self.qec.executor.map(
            lambda a: self.qec.estimate_batch(states, a),
            range(self.num_actions),
        ))
        
        # results shape: (num_actions, batch) → transpose to (batch, num_actions)
        q_values = np.array(results, dtype=np.float32).T
        q_values = np.where(np.isinf(q_values), 1e9, q_values)
        
        out = torch.tensor(q_values, dtype=torch.float32, device=obs.device)
        leading = obs.shape[:-3]
        if leading:
            return out.reshape(*leading, self.num_actions)
        else:
            return out.squeeze(0) if out.shape[0] == 1 else out

# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

class QEC:
    """Q-value Episodic Controller.

    Holds one ActionBuffer per discrete action and exposes the two core
    operations from the MFEC paper: estimate() and update().

    Parameters
    ----------
    actions     : iterable of action indices (e.g. range(num_actions))
    buffer_size : maximum entries per ActionBuffer before LRU eviction
    k           : number of nearest neighbours for Q-value estimation
    """

    def __init__(self, actions, buffer_size: int, k: int, gpu_id: int = -1) -> None:
        # One independent buffer per action.  Using a tuple prevents accidental
        # reassignment of individual buffers.
        self.buffers = tuple(ActionBuffer(buffer_size, gpu_id=gpu_id) for _ in actions)
        self.k = k
        self.executor = ThreadPoolExecutor(max_workers=len(self.buffers))

    def rebuild_trees(self) -> None:
        for buf in self.buffers:
            buf.sync_index()

    def estimate(self, state: np.ndarray, action: int) -> float:
        """Estimate Q(state, action) from the episodic memory.

        Three cases (in priority order):
        1. Exact match  → return the stored return directly.
        2. Fewer than k entries in buffer  → return +inf (optimistic, encourages
           exploration of this action).
        3. General case → average the k nearest neighbours' returns.

        Parameters
        ----------
        state  : np.ndarray, shape (state_dim,) — embedded observation
        action : int — index into self.buffers

        Returns
        -------
        float
        """
        buffer = self.buffers[action]

        # Check for an exact state match first (O(log n) via KD-tree).
        state_index = buffer.find_state(state)
        if state_index is not None:
            # We have seen this exact state before — return the best return.
            return buffer.values[state_index]

        if len(buffer) <= self.k:
            # Not enough data to compute a meaningful kNN average.
            # +inf signals "unknown but optimistic" to the argmax selector.
            return float('inf')

        # General case: retrieve the k nearest stored states and average their
        # returns.  This is the core kNN regression step from the paper.
        neighbours = buffer.find_k_nearest(state, self.k)  # list of buffer indices
        value = sum(buffer.values[idx] for idx in neighbours)
        return value / self.k

    def estimate_batch(self, states: np.ndarray, action: int) -> list[float]:
        """Estimate Q-values for a batch of states in a single FAISS search call.

        Replaces the per-state loop in QECPolicy.forward() with one batched
        index.search() invocation, which is significantly faster for large buffers.

        Parameters
        ----------
        states : np.ndarray, shape (batch, state_dim)
        action : int

        Returns
        -------
        list[float], length batch
        """
        buffer = self.buffers[action]
        n = len(states)

        if buffer._index is None or buffer._index.ntotal == 0:
            return [float("inf")] * n

        ntotal = buffer._index.ntotal
        if ntotal <= self.k:
            # Too few entries for a meaningful kNN average.
            return [float("inf")] * n

        # Fetch k+1 neighbours so we can detect exact matches (nearest dist ≈ 0)
        # without a separate second query.
        k_fetch = min(self.k + 1, ntotal)
        queries = np.array(states, dtype=np.float32)
        _, indices = buffer._index.search(queries, k_fetch)   # (n, k_fetch)

        results: list[float] = []
        for state, idx_row in zip(states, indices):
            valid = [int(j) for j in idx_row if j != -1]
            if not valid:
                results.append(float("inf"))
                continue
            nearest = valid[0]
            if np.allclose(buffer.states[nearest], state):
                # Exact hit — return the stored best return directly.
                results.append(buffer.values[nearest])
            else:
                # Average the k nearest neighbours' returns.
                knn = valid[: self.k]
                results.append(sum(buffer.values[j] for j in knn) / len(knn))

        return results

    def update(self, state: np.ndarray, action: int, return_: float, time: int) -> None:
        """Insert or update a (state, return) entry in the action's buffer.

        If the state already exists, we keep the *maximum* of the old and new
        returns (optimistic update).  This prevents a single unlucky episode
        from overwriting previously observed high returns.

        Parameters
        ----------
        state   : np.ndarray, shape (state_dim,)
        action  : int
        return_ : discounted Monte Carlo return G_t computed for this step
        time    : absolute frame index used as an LRU eviction timestamp
        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index is not None:
            # State already in memory — update value (keep max) and timestamp
            # (keep most recent so the entry survives LRU eviction).
            max_value = max(buffer.values[state_index], return_)
            max_time  = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            # New state — add to the buffer.  If the buffer is full, the oldest
            # entry is evicted automatically inside ActionBuffer.add().
            buffer.add(state, return_, time)


class ActionBuffer:
    """Fixed-capacity nearest-neighbour memory for a single action.

    Internally maintains three parallel lists — states, values, times —
    alongside a FAISS IndexFlatL2 for fast kNN queries.

    The index is kept in sync lazily: sync_index() is called once per
    training batch (by QEC.rebuild_trees()) rather than after every insert.
    Two cases:
      * Only appends since last sync  → incremental add, O(new entries).
      * An in-place eviction occurred → full rebuild, O(n log n), but this
        only happens once the buffer has reached capacity.

    Eviction policy: when at capacity, the entry with the *smallest* timestamp
    (least recently used) is replaced by the incoming entry, provided the
    incoming entry has a more recent timestamp.

    Parameters
    ----------
    capacity : int — maximum number of (state, return) entries to store
    gpu_id   : int — CUDA device index for FAISS GPU acceleration; -1 = CPU
    """

    def __init__(self, capacity: int, gpu_id: int = -1) -> None:
        self._index: faiss.Index | None = None
        self._index_size: int = 0        # entries currently reflected in _index
        self._needs_rebuild: bool = False  # True after an in-place eviction
        self._gpu_id = gpu_id
        self._gpu_res: object | None = None  # faiss.StandardGpuResources, lazy
        self.capacity = capacity
        self.states: list[np.ndarray] = []  # list of (state_dim,) float64 arrays
        self.values: list[float] = []       # corresponding best returns
        self.times:  list[int]   = []       # corresponding LRU timestamps

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _make_index(self, state_dim: int) -> faiss.Index:
        index = faiss.IndexFlatL2(state_dim)
        if self._gpu_id >= 0 and faiss.get_num_gpus() > 0:
            if self._gpu_res is None:
                self._gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self._gpu_res, self._gpu_id, index)
        return index

    def sync_index(self) -> None:
        """Sync the FAISS index with the current states list.

        Called once per batch by QEC.rebuild_trees().
        Full rebuild only on eviction; otherwise incremental adds.
        """
        if not self.states:
            return
        state_dim = self.states[0].shape[0]

        if self._needs_rebuild or self._index is None:
            self._index = self._make_index(state_dim)
            self._index.add(np.array(self.states, dtype=np.float32))
            self._index_size = len(self.states)
            self._needs_rebuild = False
        elif len(self.states) > self._index_size:
            new = np.array(self.states[self._index_size:], dtype=np.float32)
            self._index.add(new)
            self._index_size = len(self.states)

    # ------------------------------------------------------------------
    # Lookup  (used by QEC.estimate() during the update pass in step())
    # ------------------------------------------------------------------

    def find_state(self, state: np.ndarray) -> int | None:
        """Return the index of an exactly matching stored state, or None."""
        if self._index is None or self._index.ntotal == 0:
            return None
        q = np.array([state], dtype=np.float32)
        _, indices = self._index.search(q, 1)
        idx = int(indices[0][0])
        if idx == -1:
            return None
        if np.allclose(self.states[idx], state):
            return idx
        return None

    def find_k_nearest(self, state: np.ndarray, k: int) -> list[int]:
        """Return the buffer indices of the k nearest stored states."""
        if self._index is None or self._index.ntotal == 0:
            return []
        k_eff = min(k, self._index.ntotal)
        q = np.array([state], dtype=np.float32)
        _, indices = self._index.search(q, k_eff)
        return [int(i) for i in indices[0] if i != -1]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
            # Incremental add deferred to sync_index().
        else:
            min_time_index = int(np.argmin(self.times))
            if time > self.times[min_time_index]:
                self.states[min_time_index] = state
                self.values[min_time_index] = value
                self.times[min_time_index] = time
                # In-place eviction: the slot's vector changed → must rebuild.
                self._needs_rebuild = True

    def replace(self, state, value, time, index):
        # Called on an exact-match hit: only value/time change, not the
        # state vector, so the FAISS index remains valid.
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self) -> int:
        return len(self.states)