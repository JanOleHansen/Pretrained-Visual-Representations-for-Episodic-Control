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
import heapq

import numpy as np
from scipy.signal import lfilter
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

        self._explore_policy = _SharedPolicy(self.q_actor, self.greedy_module)

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
        # lfilter([1], [1, -γ], r[::-1])[::-1] computes the discounted return
        # G[t] = r[t] + γ*r[t+1] + γ²*r[t+2] + … in a single C-level IIR pass,
        # replacing the O(ep_len) Python inner loop.
        G_all = np.empty(n, dtype=np.float64)
        ends = np.flatnonzero(dones_np)
        if len(ends) == 0 or ends[-1] != n - 1:
            ends = np.append(ends, n - 1)
        ep_start = 0
        for ep_end in ends:
            r = rewards_np[ep_start : ep_end + 1]
            G_all[ep_start : ep_end + 1] = lfilter([1.0], [1.0, -self.gamma], r[::-1])[::-1]
            ep_start = ep_end + 1

        # Absolute frame timestamp for LRU eviction.
        frame_times = (self._collected_frames - n) + np.arange(n, dtype=np.int64)

        #   Vectorised per-action batch update
        # Group transition indices by action, then do ONE batched FAISS search
        # per action (instead of n individual single-vector searches).
        # Exact-match detection and value/time updates are then fully vectorised
        # with numpy; only genuinely new states fall through to the Python loop.
        for action in range(self._num_actions):
            mask = actions_np == action
            if not mask.any():
                continue
            buf        = self.qec.buffers[action]
            act_states = states[mask].astype(np.float32)  # (m, state_dim)
            act_values = G_all[mask]                       # (m,)
            act_times  = frame_times[mask]                 # (m,)

            if buf._index is not None and buf._index.ntotal > 0:
                # One FAISS search for all m states belonging to this action.
                _, nn_idx = buf._index.search(act_states, 1)  # (m, 1)
                nn_flat = nn_idx[:, 0]                         # (m,)
                valid   = nn_flat >= 0
                safe_nn = np.where(valid, nn_flat, 0)

                # Vectorised exact-match check across all m queries at once.
                nearest_vecs = buf.states[safe_nn]             # (m, state_dim)
                tol  = 1e-5 * (1.0 + np.abs(act_states))
                exact = valid & np.all(np.abs(nearest_vecs - act_states) <= tol, axis=1)

                # Batch replace: max-aggregation for all exact-match slots.
                # np.maximum.at handles duplicate indices correctly.
                if exact.any():
                    ex_idx = safe_nn[exact]
                    np.maximum.at(buf.values, ex_idx, act_values[exact])
                    np.maximum.at(buf.times,  ex_idx, act_times[exact])
                    for slot, t in zip(ex_idx.tolist(), buf.times[ex_idx].tolist()):
                        heapq.heappush(buf._heap, (t, slot))

                # Add new (non-matching) states — eviction logic prevents batching.
                for i in np.flatnonzero(~exact):
                    buf.add(act_states[i], float(act_values[i]), int(act_times[i]))
            else:
                for i in range(len(act_states)):
                    buf.add(act_states[i], float(act_values[i]), int(act_times[i]))

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
                    {
                        "states": b.states[:b._size].copy() if b.states is not None and b._size > 0 else None,
                        "values": b.values[:b._size].copy() if b._size > 0 else np.array([], dtype=np.float64),
                        "times":  b.times[:b._size].copy()  if b._size > 0 else np.array([], dtype=np.int64),
                        "size":   b._size,
                    }
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
            size = saved["size"]
            buf._size  = size
            buf._index = None
            buf._index_size = 0
            buf._needs_rebuild = False
            buf.values = np.empty(buf.capacity, dtype=np.float64)
            buf.times  = np.empty(buf.capacity, dtype=np.int64)
            if size > 0:
                buf.values[:size] = np.asarray(saved["values"])
                buf.times[:size]  = np.asarray(saved["times"])
            if size > 0 and saved["states"] is not None:
                state_dim = saved["states"].shape[1]
                buf.states = np.empty((buf.capacity, state_dim), dtype=np.float32)
                buf.states[:size] = saved["states"]
            else:
                buf.states = None
            buf._heap = [(int(buf.times[i]), i) for i in range(size)]
            heapq.heapify(buf._heap)
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

    def __deepcopy__(self, memo):
        # TorchRL deepcopies the policy when setting up the collector (to move
        # it to the right device).  A full deepcopy would give the collector its
        # own isolated, empty QEC that never receives updates from step().
        # Instead we share self.qec by reference so the collector always reads
        # the latest memory that algorithm.step() writes.
        cls = self.__class__
        copy = cls.__new__(cls)
        memo[id(self)] = copy
        # nn.Module internals (no learnable params, but must be valid)
        super(QECPolicy, copy).__init__()
        copy.qec = self.qec                   # shared reference — intentional
        copy.projection = self.projection     # immutable numpy array
        copy.num_actions = self.num_actions
        copy._proj_tensor = None              # re-cached on first embed() call
        return copy

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["executor"]  # ThreadPoolExecutor is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.executor = ThreadPoolExecutor(max_workers=len(self.buffers))

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
        return float(buffer.values[neighbours].mean())

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
        queries = np.asarray(states, dtype=np.float32)
        _, indices = buffer._index.search(queries, k_fetch)   # (n, k_fetch)

        valid   = indices >= 0                                 # (n, k_fetch) bool
        safe    = np.where(valid, indices, 0)                  # replace -1 with 0

        nearest  = safe[:, 0]                                  # (n,) closest slot
        has_hit  = valid[:, 0]                                 # (n,) at least one result

        # Exact-match: nearest stored vector == query within floating-point tol.
        nearest_vecs = buffer.states[nearest]                  # (n, state_dim)
        tol  = 1e-5 * (1.0 + np.abs(queries))
        exact = has_hit & np.all(np.abs(nearest_vecs - queries) <= tol, axis=1)

        # kNN average for non-exact rows — fully vectorised with numpy indexing.
        k_use    = min(self.k, k_fetch)
        knn_idx  = safe[:, :k_use]                             # (n, k_use)
        knn_ok   = valid[:, :k_use]                            # (n, k_use) mask
        knn_vals = buffer.values[knn_idx]                      # (n, k_use)
        knn_vals = np.where(knn_ok, knn_vals, np.nan)
        with np.errstate(all="ignore"):
            knn_avg = np.nanmean(knn_vals, axis=1)             # (n,)

        result = np.where(exact, buffer.values[nearest], knn_avg)
        result = np.where(has_hit, result, np.inf)
        return result.astype(np.float32)

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
        self._size: int = 0
        # states is a preallocated (capacity, state_dim) float32 array, lazily
        # created on the first add() call when state_dim becomes known.
        # Keeping states as a contiguous array means sync_index() can pass a
        # zero-copy slice to FAISS instead of rebuilding from a list every time.
        self.states: np.ndarray | None = None
        # Preallocated numpy arrays — enable vectorised indexing in estimate_batch
        # and eliminate the Python-list → numpy conversion cost in np.argmin.
        self.values: np.ndarray = np.empty(capacity, dtype=np.float64)
        self.times:  np.ndarray = np.empty(capacity, dtype=np.int64)
        # Min-heap of (time, slot_index) for O(log n) LRU eviction.
        # Lazy deletion: entries become stale when times[slot] != stored time.
        self._heap: list = []

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
        """Sync the FAISS index with the current states array.

        Called once per batch by QEC.rebuild_trees().
        Full rebuild only on eviction; otherwise incremental adds.
        Passes a zero-copy slice to FAISS — no conversion overhead.
        """
        if self.states is None or self._size == 0:
            return
        state_dim = self.states.shape[1]

        if self._needs_rebuild or self._index is None:
            self._index = self._make_index(state_dim)
            self._index.add(self.states[:self._size])   # zero-copy view
            self._index_size = self._size
            self._needs_rebuild = False
        elif self._size > self._index_size:
            self._index.add(self.states[self._index_size:self._size])
            self._index_size = self._size

    # ------------------------------------------------------------------
    # Lookup  (used by QEC.estimate() during the update pass in step())
    # ------------------------------------------------------------------

    def find_state(self, state: np.ndarray) -> int | None:
        """Return the index of an exactly matching stored state, or None."""
        if self._index is None or self._index.ntotal == 0:
            return None
        q = np.asarray(state, dtype=np.float32).reshape(1, -1)
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
        q = np.asarray(state, dtype=np.float32).reshape(1, -1)
        _, indices = self._index.search(q, k_eff)
        return [int(i) for i in indices[0] if i != -1]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, state: np.ndarray, value: float, time: int) -> None:
        if self.states is None:
            self.states = np.empty((self.capacity, state.shape[0]), dtype=np.float32)
        if self._size < self.capacity:
            idx = self._size
            self.states[idx] = state
            self.values[idx] = value
            self.times[idx]  = time
            self._size += 1
            heapq.heappush(self._heap, (time, idx))
            # Incremental add deferred to sync_index().
        else:
            # O(log n) LRU eviction via min-heap with lazy deletion.
            # Pop until we find a non-stale entry (times[slot] matches heap time).
            while True:
                t, idx = heapq.heappop(self._heap)
                if self.times[idx] == t:
                    break
            if time > t:
                self.states[idx] = state
                self.values[idx] = value
                self.times[idx]  = time
                heapq.heappush(self._heap, (time, idx))
                # In-place eviction: the slot's vector changed → must rebuild.
                self._needs_rebuild = True
            else:
                # Incoming entry is older than the LRU slot — discard it.
                heapq.heappush(self._heap, (t, idx))

    def replace(self, state: np.ndarray, value: float, time: int, index: int) -> None:
        # Called on an exact-match hit: only value/time change, not the
        # state vector, so the FAISS index remains valid.
        self.states[index] = state
        self.values[index] = value
        self.times[index]  = time
        heapq.heappush(self._heap, (time, index))  # old entry becomes stale

    def __len__(self) -> int:
        return self._size

    def __getstate__(self):
        d = self.__dict__.copy()
        # FAISS index and GPU resources are not picklable; drop them.
        d["_index"] = None
        d["_gpu_res"] = None
        d["_index_size"] = 0
        d["_needs_rebuild"] = False
        # Heap is rebuilt cheaply from times on load; don't pickle it.
        d["_heap"] = None
        # Save only the active slice so we don't pickle the full preallocated
        # arrays — most of which may be uninitialised.
        d["states"] = self.states[:self._size].copy() if self.states is not None and self._size > 0 else None
        d["values"] = self.values[:self._size].copy() if self._size > 0 else np.array([], dtype=np.float64)
        d["times"]  = self.times[:self._size].copy()  if self._size > 0 else np.array([], dtype=np.int64)
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        size = self._size
        # Restore full preallocated values/times arrays from compact slices.
        saved_values = d["values"]
        saved_times  = d["times"]
        self.values = np.empty(self.capacity, dtype=np.float64)
        self.times  = np.empty(self.capacity, dtype=np.int64)
        if size > 0:
            self.values[:size] = saved_values
            self.times[:size]  = saved_times
        # Restore full preallocated states array from compact slice.
        saved_states = d["states"]
        if saved_states is not None and size > 0:
            state_dim = saved_states.shape[1]
            self.states = np.empty((self.capacity, state_dim), dtype=np.float32)
            self.states[:size] = saved_states
        else:
            self.states = None
        # Rebuild a fresh, stale-free heap from the stored timestamps.
        self._heap = [(int(self.times[i]), i) for i in range(size)]
        heapq.heapify(self._heap)
        self.sync_index()