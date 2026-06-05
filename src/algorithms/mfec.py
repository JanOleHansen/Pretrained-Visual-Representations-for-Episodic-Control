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

function UpdateMemory(s, a, G):
    if s is in Q_EC[a]:
        Q_EC[a][s] ← max(Q_EC[a][s], G)       # keep best return seen
    else:
        if |Q_EC[a]| ≥ capacity:
            evict oldest entry (ring buffer)
        insert (s, G) into Q_EC[a]
"""

from __future__ import annotations
from typing import Callable

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
            obs_key: str = "pixels",
            buffer_size: int = 1_000_000,
            k: int = 11,
            state_dim: int = 64,
            gamma: float = 0.99,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            annealing_frames: int = 1_000_000,
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
    # Setup
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

        self._buffer_device = (
            self.device if self.device is not None else torch.device("cpu")
        )

        self.qec = QEC(num_actions, self.buffer_size, self.k, self._buffer_device)
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
    #   Training step
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
            Scalar metrics logged by the trainer.
        """

        batch = batch.reshape(-1)
        n = batch.numel()

        self.greedy_module.step(n)
        self._collected_frames += n

        dev = self._buffer_device
        obs = batch[self.obs_key]

        # actions stay on GPU — no .cpu() needed.
        actions_gpu = batch["action"].to(dev).flatten().long()   # (n,)

        # rewards/dones still go through scipy on CPU (not the bottleneck).
        rewards_np = batch["next", "reward"].cpu().numpy().flatten().astype(np.float64)
        dones_np   = batch["next", "done"].cpu().numpy().flatten().astype(bool)

        # Embed all observations in one GPU matmul — result stays on device.
        states_gpu = self.qec_policy.embed(obs)   # (n, d)
        if states_gpu.device != dev:
            states_gpu = states_gpu.to(dev)

        # Discounted Monte Carlo returns via scipy IIR.
        G_all = np.empty(n, dtype=np.float64)
        ends = np.flatnonzero(dones_np)
        if len(ends) == 0 or ends[-1] != n - 1:
            ends = np.append(ends, n - 1)
        ep_start = 0
        for ep_end in ends:
            r = rewards_np[ep_start : ep_end + 1]
            G_all[ep_start : ep_end + 1] = lfilter([1.0], [1.0, -self.gamma], r[::-1])[::-1]
            ep_start = ep_end + 1
        G_gpu = torch.tensor(G_all, dtype=torch.float64, device=dev)

        # Sort transitions by action on GPU — one argsort, no per-action CPU mask.
        sorted_idx     = torch.argsort(actions_gpu, stable=True)   # (n,)
        sorted_states  = states_gpu[sorted_idx]                    # (n, d)
        sorted_values  = G_gpu[sorted_idx]                         # (n,)
        sorted_actions = actions_gpu[sorted_idx]                   # (n,) sorted

        counts  = torch.bincount(sorted_actions, minlength=self._num_actions)
        offsets = torch.zeros(self._num_actions + 1, dtype=torch.long, device=dev)
        offsets[1:] = counts.cumsum(0)
        # One small CPU sync to read segment boundaries (A+1 integers).
        offsets_cpu = offsets.cpu().tolist()

        for a in range(self._num_actions):
            s, e = offsets_cpu[a], offsets_cpu[a + 1]
            if s == e:
                continue

            act_states = sorted_states[s:e]   # view — zero-copy slice
            act_values = sorted_values[s:e]

            if self.qec._sizes[a] > 0:
                # Batched kNN for all m states of this action in one GPU call.
                dists, nn_idx = self.qec.knn_action(act_states, a, 1)
                exact = dists[:, 0] < 1e-5   # (m,) bool

                if exact.any():
                    ex_slots = nn_idx[:, 0][exact]
                    # In-place max-aggregation; rare duplicate slots overwrite (acceptable).
                    self.qec.values[a, ex_slots] = torch.maximum(
                        self.qec.values[a, ex_slots], act_values[exact]
                    )

                new_mask = ~exact
                if new_mask.any():
                    self.qec.add_batch(a, act_states[new_mask], act_values[new_mask])
            else:
                self.qec.add_batch(a, act_states, act_values)

        return {
            "train/epsilon":  float(self.greedy_module.eps),
            "train/qec_size": float(np.mean(self.qec._sizes)),
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
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    #
    #   Checkpointing
    #

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict={},
            optimizer_state_dict={},
            extra={
                "projection":       self.projection,
                "qec_state":        self.qec.__getstate__(),
                "collected_frames": self._collected_frames,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.projection = state.extra["projection"]
        self.qec_policy.projection = self.projection
        self.qec_policy._proj_tensor = None
        self._collected_frames = int(state.extra["collected_frames"])
        self.qec.__setstate__(state.extra["qec_state"])


# ---------------------------------------------------------------------------
# Policy module
# ---------------------------------------------------------------------------

class QECPolicy(nn.Module):
    """Non-parametric nn.Module that estimates Q-values via kNN memory lookup.

    embed() and forward() are split so that MFECAlgorithm.step() can embed
    the whole batch in one GPU matmul without going through the full forward pass.
    """

    def __init__(self, qec: "QEC", projection: np.ndarray, num_actions: int) -> None:
        super().__init__()
        self.qec = qec
        self.projection = projection
        self.num_actions = num_actions
        self._proj_tensor: torch.Tensor | None = None

    def __deepcopy__(self, memo):
        # TorchRL deepcopies the policy when setting up the collector.  Share
        # self.qec by reference so the collector reads the memory that step() writes.
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
        """Project observation to (batch, state_dim) — result stays on obs.device."""
        if self._proj_tensor is None or self._proj_tensor.device != obs.device:
            self._proj_tensor = torch.tensor(
                self.projection, dtype=torch.float32, device=obs.device
            )
        flat = obs.float().reshape(-1, self._proj_tensor.shape[0])
        return flat @ self._proj_tensor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        states = self.embed(obs)                   # (B, d) on obs.device

        # Single fused call — one batched cdist across ALL actions, no thread pool.
        q_values = self.qec.estimate_all(states)   # (B, A) on obs.device

        q_values = torch.where(
            torch.isinf(q_values),
            torch.full_like(q_values, 1e9),
            q_values,
        )
        leading = obs.shape[:-3]
        if leading:
            return q_values.reshape(*leading, self.num_actions)
        return q_values.squeeze(0) if q_values.shape[0] == 1 else q_values


class _SharedPolicy(TensorDictSequential):
    """Returns self on deepcopy so a single EGreedyModule is shared with the collector."""

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# ---------------------------------------------------------------------------
# Fused episodic memory — all actions in one GPU tensor pair
# ---------------------------------------------------------------------------

class QEC:
    """Q-value Episodic Controller — fused GPU tensor layout.

    All action buffers share two pre-allocated GPU tensors:

        states : (num_actions, capacity, state_dim)  float32
        values : (num_actions, capacity)             float64

    Each action has its own ring-buffer write pointer and fill count.
    All kNN queries are batched across actions in a single torch.cdist call
    (for forward) or chunked per-action (for step, where query counts vary).

    Parameters
    ----------
    num_actions : int
    capacity    : int — max entries per action
    k           : int — number of nearest neighbours
    device      : torch.device
    """

    # Maximum intermediate (queries × stored) tensor in bytes before chunking.
    # 256 MB keeps us well inside typical VRAM budgets.
    _CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(
        self, num_actions: int, capacity: int, k: int, device: torch.device
    ) -> None:
        self.num_actions = num_actions
        self.capacity    = capacity
        self.k           = k
        self.device      = device

        self.states: torch.Tensor | None = None   # (A, C, d) — lazy init
        self.values = torch.empty(num_actions, capacity, dtype=torch.float64, device=device)

        # Per-action scalars — updated at most A times per step().
        self._sizes      = [0] * num_actions
        self._write_ptrs = [0] * num_actions

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init_states(self, state_dim: int) -> None:
        if self.states is None:
            self.states = torch.empty(
                self.num_actions, self.capacity, state_dim,
                dtype=torch.float32, device=self.device,
            )

    # ------------------------------------------------------------------
    # Q-value estimation — called from QECPolicy.forward()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_all(self, queries: torch.Tensor) -> torch.Tensor:
        """Estimate Q(s, a) for all actions simultaneously.

        One batched torch.cdist replaces A separate calls and eliminates
        the thread pool that was needed to overlap them.

        Parameters
        ----------
        queries : (B, d) float32 — any device

        Returns
        -------
        (B, A) float32 — on queries.device; +inf where memory is too sparse
        """
        B   = queries.shape[0]
        A   = self.num_actions
        dev_q = queries.device

        max_size = max(self._sizes)
        if self.states is None or max_size == 0:
            return torch.full((B, A), float("inf"), dtype=torch.float32, device=dev_q)

        if queries.device != self.device:
            queries = queries.to(self.device)

        sizes_t = torch.tensor(self._sizes, dtype=torch.long, device=self.device)
        k_fetch = min(self.k + 1, max_size)

        # Expand queries to (A, B, d) — view with zero-copy stride trick.
        # .contiguous() is needed for cdist's GEMM path.
        q_exp = queries.unsqueeze(0).expand(A, B, -1).contiguous()   # (A, B, d)

        # Single batched cdist across all actions: (A, B, max_size)
        dists = torch.cdist(q_exp, self.states[:, :max_size, :])

        # Mask slots beyond each action's current fill to +inf.
        slot_range = torch.arange(max_size, device=self.device)
        invalid = slot_range[None, :] >= sizes_t[:, None]             # (A, max_size)
        dists.masked_fill_(invalid[:, None, :], float("inf"))         # broadcast over B

        top_dists, top_idx = dists.topk(k_fetch, dim=-1, largest=False)   # (A, B, k_fetch)

        # Exact match: nearest L2 distance below floating-point tolerance.
        exact = top_dists[..., 0] < 1e-5   # (A, B)

        # kNN average via gathered values.
        k_use = min(self.k, k_fetch)
        knn_idx = top_idx[..., :k_use]     # (A, B, k_use)
        a_idx   = (
            torch.arange(A, device=self.device)[:, None, None]
            .expand(A, B, k_use)
        )
        knn_vals = self.values[a_idx, knn_idx].float()   # (A, B, k_use)
        knn_avg  = knn_vals.mean(dim=-1)                 # (A, B)

        a_idx1     = torch.arange(A, device=self.device)[:, None].expand(A, B)
        exact_vals = self.values[a_idx1, top_idx[..., 0]].float()   # (A, B)

        q_values = torch.where(exact, exact_vals, knn_avg)

        # Actions with too few stored entries return +inf (optimistic exploration).
        too_few  = sizes_t[:, None] <= self.k   # (A, 1) → broadcasts to (A, B)
        q_values = torch.where(too_few, torch.full_like(q_values, float("inf")), q_values)

        return q_values.T.to(dev_q)   # (B, A)

    # ------------------------------------------------------------------
    # kNN for a single action — used in step()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_action(
        self,
        queries: torch.Tensor,   # (m, d)
        action:  int,
        k:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked kNN search for one action.

        Processes the stored vectors in chunks sized to stay within
        _CHUNK_BYTES, so the intermediate distance matrix never triggers OOM
        regardless of buffer fill or query count.

        Returns
        -------
        dists   : (m, k_eff) — L2 distances, on self.device
        indices : (m, k_eff) — slot indices into self.states[action], on self.device
        """
        size  = self._sizes[action]
        k_eff = min(k, size)

        if queries.device != self.device:
            queries = queries.to(self.device)

        m = queries.shape[0]
        d = queries.shape[1]
        # How many stored vectors fit in _CHUNK_BYTES alongside the query batch?
        chunk_size = max(1, self._CHUNK_BYTES // (m * d * 4))
        chunk_size = min(chunk_size, size)

        best_dists = torch.full((m, k_eff), float("inf"), device=self.device)
        best_idx   = torch.zeros((m, k_eff), dtype=torch.long, device=self.device)

        for cs in range(0, size, chunk_size):
            ce = min(cs + chunk_size, size)
            cd  = torch.cdist(queries, self.states[action, cs:ce])    # (m, chunk)
            ck  = min(k_eff, ce - cs)
            chd, chi = cd.topk(ck, dim=1, largest=False)
            chi = chi + cs   # shift to global slot indices

            merged_d = torch.cat([best_dists, chd], dim=1)
            merged_i = torch.cat([best_idx,   chi], dim=1)
            _, keep  = merged_d.topk(k_eff, dim=1, largest=False)
            best_dists = merged_d.gather(1, keep)
            best_idx   = merged_i.gather(1, keep)

        return best_dists, best_idx

    # ------------------------------------------------------------------
    # Insertion — ring-buffer scatter for one action
    # ------------------------------------------------------------------

    def add_batch(
        self,
        action: int,
        states: torch.Tensor,   # (n, d) float32
        values: torch.Tensor,   # (n,) float64
    ) -> None:
        n = len(states)
        if n == 0:
            return

        if states.device != self.device:
            states = states.to(self.device)
            values = values.to(self.device)

        self._init_states(states.shape[1])

        # If the batch itself exceeds capacity, keep only the newest entries.
        if n > self.capacity:
            states = states[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity

        # Destination slots via modular arithmetic — one GPU op.
        ptr   = self._write_ptrs[action]
        slots = torch.arange(ptr, ptr + n, device=self.device) % self.capacity
        self.states[action, slots] = states
        self.values[action, slots] = values

        self._write_ptrs[action] = int((ptr + n) % self.capacity)
        self._sizes[action]      = min(self._sizes[action] + n, self.capacity)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        d: dict = {
            "num_actions":  self.num_actions,
            "capacity":     self.capacity,
            "k":            self.k,
            "device":       self.device,
            "_sizes":       list(self._sizes),
            "_write_ptrs":  list(self._write_ptrs),
            "action_states": None,
            "action_values": None,
        }
        if self.states is None:
            return d

        action_states = []
        action_values = []
        for a in range(self.num_actions):
            sz  = self._sizes[a]
            ptr = self._write_ptrs[a]
            if sz == 0:
                action_states.append(None)
                action_values.append(np.array([], dtype=np.float64))
                continue
            # Rotate so index 0 = oldest entry (preserves write-order on load).
            if sz == self.capacity and ptr != 0:
                s = torch.roll(self.states[a], -ptr, dims=0)[:sz]
                v = torch.roll(self.values[a], -ptr, dims=0)[:sz]
            else:
                s = self.states[a, :sz]
                v = self.values[a, :sz]
            action_states.append(s.cpu().numpy())
            action_values.append(v.cpu().numpy())

        d["action_states"] = action_states
        d["action_values"] = action_values
        return d

    def __setstate__(self, d: dict) -> None:
        self.num_actions = d["num_actions"]
        self.capacity    = d["capacity"]
        self.k           = d["k"]
        self.device      = d["device"]
        self._sizes      = list(d["_sizes"])
        self._write_ptrs = list(d["_write_ptrs"])

        dev = self.device
        self.values = torch.empty(self.num_actions, self.capacity, dtype=torch.float64, device=dev)

        if d["action_states"] is None:
            self.states = None
            return

        # Determine state_dim from first non-empty action.
        state_dim = next(
            (s.shape[1] for s in d["action_states"] if s is not None and s.ndim == 2 and s.shape[0] > 0),
            None,
        )
        if state_dim is None:
            self.states = None
            return

        self.states = torch.empty(
            self.num_actions, self.capacity, state_dim, dtype=torch.float32, device=dev
        )
        for a, (s_np, v_np) in enumerate(zip(d["action_states"], d["action_values"])):
            sz = self._sizes[a]
            if sz > 0 and s_np is not None:
                self.states[a, :sz] = torch.from_numpy(s_np).to(dev)
                self.values[a, :sz] = torch.from_numpy(v_np).to(dev)
