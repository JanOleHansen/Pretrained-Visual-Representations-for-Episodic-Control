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
            # Quantisation precision for the exact-match hash key.
            # Each embedding coordinate is multiplied by this factor and rounded
            # to the nearest integer before being converted to bytes.  Higher
            # values preserve more precision (fewer false-negatives) but make
            # keys longer.  1e5 gives five decimal places of float32 stability.
            key_scale: float = 1e5,
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
        self.key_scale = key_scale

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

        self.qec = QEC(
            num_actions, self.buffer_size, self.k, self._buffer_device,
            key_scale=self.key_scale,
        )
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
          2. Exact-match states (hash lookup, O(1)) → max-aggregate the stored value.
          3. Novel states → insert into the ring buffer via add_batch().

        Parameters
        ----------
        batch : TensorDict
            Transitions from SyncDataCollector.  May span multiple episodes.

        Returns
        -------
        dict
            Scalar metrics including train/exact_hit_rate.
        """

        batch = batch.reshape(-1)
        n = batch.numel()

        self.greedy_module.step(n)
        self._collected_frames += n

        dev = self._buffer_device
        obs = batch[self.obs_key]

        # actions stay on GPU — no .cpu() needed.
        actions_gpu = batch["action"].to(dev).flatten().long()   # (n,)

        # rewards/dones go through scipy on CPU (not the bottleneck).
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
        sorted_idx     = torch.argsort(actions_gpu, stable=True)
        sorted_states  = states_gpu[sorted_idx]                    # (n, d)
        sorted_values  = G_gpu[sorted_idx]                         # (n,)
        sorted_actions = actions_gpu[sorted_idx]                   # (n,) sorted

        counts  = torch.bincount(sorted_actions, minlength=self._num_actions)
        offsets = torch.zeros(self._num_actions + 1, dtype=torch.long, device=dev)
        offsets[1:] = counts.cumsum(0)
        # One CPU sync to read A+1 segment boundary integers.
        offsets_cpu = offsets.cpu().tolist()

        total_queries = 0
        total_hits    = 0

        for a in range(self._num_actions):
            s, e = offsets_cpu[a], offsets_cpu[a + 1]
            if s == e:
                continue

            act_states = sorted_states[s:e]   # view — zero-copy slice
            act_values = sorted_values[s:e]
            m = e - s
            total_queries += m

            # --- Exact-match path: O(m) hash lookup, no kNN needed ------
            keys_a = self.qec._make_keys(act_states)
            k_to_s = self.qec._key_to_slot[a]
            exact_list = [key in k_to_s for key in keys_a]
            n_hits = sum(exact_list)
            total_hits += n_hits

            if n_hits:
                hit_positions = [i for i, hit in enumerate(exact_list) if hit]
                ex_slots = torch.tensor(
                    [k_to_s[keys_a[i]] for i in hit_positions],
                    dtype=torch.long, device=dev,
                )
                exact_t = torch.tensor(exact_list, dtype=torch.bool, device=dev)
                # In-place max-aggregation; duplicate slots (rare) overwrite (acceptable).
                self.qec.values[a, ex_slots] = torch.maximum(
                    self.qec.values[a, ex_slots], act_values[exact_t]
                )

            # --- Novel states → ring-buffer insertion ---------------------
            n_new = m - n_hits
            if n_new:
                if n_hits:
                    novel_mask = torch.tensor(
                        [not hit for hit in exact_list], dtype=torch.bool, device=dev
                    )
                    self.qec.add_batch(a, act_states[novel_mask], act_values[novel_mask])
                else:
                    self.qec.add_batch(a, act_states, act_values)

        hit_rate = total_hits / total_queries if total_queries > 0 else 0.0

        return {
            "train/epsilon":        float(self.greedy_module.eps),
            "train/qec_size":       float(np.mean(self.qec._sizes)),
            "train/exact_hit_rate": hit_rate,
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
        states = self.embed(obs)                    # (B, d) on obs.device
        q_values = self.qec.estimate_all(states)    # (B, A) — dict-first, kNN for misses
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
    """Q-value Episodic Controller — fused GPU tensors + per-action hash maps.

    GPU tensors (states, values) hold all action buffers in a single allocation:

        states : (num_actions, capacity, state_dim)  float32
        values : (num_actions, capacity)             float64

    A pair of Python dicts per action provides O(1) exact-match lookup,
    implementing the paper's Eq. (2) "case 1" without any distance computation:

        _key_to_slot[a] : bytes → int   (quantised embedding → slot index)
        _slot_to_key[a] : int   → bytes (slot index → key, for O(1) eviction)

    estimate_all() checks the dicts first and only falls back to kNN (cdist)
    for novel queries.  add_batch() keeps the dicts consistent with the ring
    buffer, evicting the stale key whenever a slot is overwritten.

    Parameters
    ----------
    num_actions : int
    capacity    : int — max entries per action
    k           : int — number of nearest neighbours for novel states
    device      : torch.device
    key_scale   : float — multiply embeddings by this before rounding to int32
                  for the hash key.  Higher → finer quantisation; must keep
                  values within int32 range (~2.1 × 10⁹).  Default 1e5 gives
                  five decimal places of float32 precision.
    """

    # Maximum intermediate (queries × stored) tensor in bytes before chunking.
    _CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        num_actions: int,
        capacity:    int,
        k:           int,
        device:      torch.device,
        key_scale:   float = 1e5,
    ) -> None:
        self.num_actions = num_actions
        self.capacity    = capacity
        self.k           = k
        self.device      = device
        self._key_scale  = key_scale

        self.states: torch.Tensor | None = None   # (A, C, d) — lazy init
        self.values = torch.empty(num_actions, capacity, dtype=torch.float64, device=device)

        # Per-action ring-buffer state.
        self._sizes      = [0] * num_actions
        self._write_ptrs = [0] * num_actions

        # Per-action exact-match hash maps.
        # _key_to_slot: used for O(1) lookup in estimate_all() and step().
        # _slot_to_key: used for O(1) eviction when a ring-buffer slot wraps.
        self._key_to_slot: list[dict[bytes, int]] = [{} for _ in range(num_actions)]
        self._slot_to_key: list[dict[int, bytes]] = [{} for _ in range(num_actions)]

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_keys(self, states: torch.Tensor) -> list[bytes]:
        """Convert a batch of float32 embeddings to stable bytes hash keys.

        Each embedding row is multiplied by _key_scale and rounded to the
        nearest int32, then converted to raw bytes.  This gives a key that
        is identical for the same observation (the projection is deterministic)
        and robust to float noise up to 0.5 / _key_scale per coordinate.

        The paper relies on exact state re-encounters: same pixels → same key.
        The random projection is fixed, so identical pixels always yield the
        same float32 embedding, and hence the same key.

        Parameters
        ----------
        states : (B, d) float32 — on any device

        Returns
        -------
        list of B bytes objects (one per row)
        """
        q = torch.round(states * self._key_scale).to(torch.int32)
        q_cpu = q.cpu().contiguous()
        return [q_cpu[i].numpy().tobytes() for i in range(q_cpu.shape[0])]

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
        """Estimate Q(s, a) for all actions.

        Per-query, per-action strategy (Eq. 2 of Blundell et al. 2016):
          1. Dict lookup: O(1) — return stored value directly (no cdist).
          2. kNN fallback: chunked cdist for queries not in the dict.

        Parameters
        ----------
        queries : (B, d) float32 — any device

        Returns
        -------
        (B, A) float32 — on queries.device; +inf where memory is too sparse
        """
        B     = queries.shape[0]
        A     = self.num_actions
        dev_q = queries.device

        max_size = max(self._sizes)
        if self.states is None or max_size == 0:
            return torch.full((B, A), float("inf"), dtype=torch.float32, device=dev_q)

        if queries.device != self.device:
            queries = queries.to(self.device)

        # One GPU→CPU sync to compute all keys (B rows × d int32 values).
        keys = self._make_keys(queries)   # list[bytes], length B

        result = torch.full((A, B), float("inf"), dtype=torch.float32, device=self.device)

        for a in range(A):
            size_a = self._sizes[a]
            k_to_s = self._key_to_slot[a]

            # --- Exact-match path: O(B) dict lookups, zero GPU work -----
            hit_b: list[int]   = []
            hit_slots: list[int] = []
            miss_b: list[int]  = []
            for b, key in enumerate(keys):
                slot = k_to_s.get(key)
                if slot is not None:
                    hit_b.append(b)
                    hit_slots.append(slot)
                else:
                    miss_b.append(b)

            if hit_b:
                hit_b_t = torch.tensor(hit_b,    dtype=torch.long, device=self.device)
                hit_s_t = torch.tensor(hit_slots, dtype=torch.long, device=self.device)
                result[a, hit_b_t] = self.values[a, hit_s_t].float()

            if not miss_b:
                continue   # all queries hit — no kNN needed for this action

            # +inf for actions with too few stored entries (optimistic init)
            if size_a <= self.k:
                continue   # result[a, miss_b] stays +inf

            # --- kNN fallback for misses (chunked cdist) -----------------
            miss_b_t   = torch.tensor(miss_b, dtype=torch.long, device=self.device)
            miss_q     = queries[miss_b_t]     # (miss_count, d)
            k_fetch    = min(self.k + 1, size_a)
            dists, idx = self.knn_action(miss_q, a, k_fetch)  # (miss_count, k_fetch)

            # Near-exact in the kNN result (float safety; should be rare)
            near_exact = dists[:, 0] < 1e-5
            k_use      = min(self.k, k_fetch)
            knn_vals   = self.values[a, idx[:, :k_use]].float()   # (miss_count, k_use)
            knn_avg    = knn_vals.mean(dim=-1)
            exact_vals = self.values[a, idx[:, 0]].float()
            result[a, miss_b_t] = torch.where(near_exact, exact_vals, knn_avg)

        return result.T.to(dev_q)   # (B, A)

    # ------------------------------------------------------------------
    # kNN for a single action — used by estimate_all() for misses
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_action(
        self,
        queries: torch.Tensor,   # (m, d)
        action:  int,
        k:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked kNN search for one action (exact L2, GPU-resident).

        Chunks the stored vectors to stay within _CHUNK_BYTES so the
        intermediate distance matrix never triggers OOM.

        Returns
        -------
        dists   : (m, k_eff) L2 distances — on self.device
        indices : (m, k_eff) slot indices into self.states[action]
        """
        size  = self._sizes[action]
        k_eff = min(k, size)

        if queries.device != self.device:
            queries = queries.to(self.device)

        m = queries.shape[0]
        d = queries.shape[1]
        chunk_size = max(1, self._CHUNK_BYTES // (m * d * 4))
        chunk_size = min(chunk_size, size)

        best_dists = torch.full((m, k_eff), float("inf"), device=self.device)
        best_idx   = torch.zeros((m, k_eff), dtype=torch.long,  device=self.device)

        for cs in range(0, size, chunk_size):
            ce  = min(cs + chunk_size, size)
            cd  = torch.cdist(queries, self.states[action, cs:ce])
            ck  = min(k_eff, ce - cs)
            chd, chi = cd.topk(ck, dim=1, largest=False)
            chi = chi + cs

            merged_d = torch.cat([best_dists, chd], dim=1)
            merged_i = torch.cat([best_idx,   chi], dim=1)
            _, keep  = merged_d.topk(k_eff, dim=1, largest=False)
            best_dists = merged_d.gather(1, keep)
            best_idx   = merged_i.gather(1, keep)

        return best_dists, best_idx

    # ------------------------------------------------------------------
    # Insertion — ring-buffer scatter + dict maintenance
    # ------------------------------------------------------------------

    def add_batch(
        self,
        action: int,
        states: torch.Tensor,   # (n, d) float32
        values: torch.Tensor,   # (n,) float64
    ) -> None:
        """Insert novel (state, value) pairs via ring-buffer scatter.

        Also registers each new key in _key_to_slot / _slot_to_key and
        evicts the stale key of any overwritten slot so the dicts stay
        consistent with the GPU tensors.
        """
        n = len(states)
        if n == 0:
            return

        if states.device != self.device:
            states = states.to(self.device)
            values = values.to(self.device)

        self._init_states(states.shape[1])

        if n > self.capacity:
            states = states[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity

        ptr   = self._write_ptrs[action]
        slots = torch.arange(ptr, ptr + n, device=self.device) % self.capacity
        self.states[action, slots] = states
        self.values[action, slots] = values

        # Dict maintenance: O(n) Python operations.
        new_keys   = self._make_keys(states)   # one GPU→CPU sync (n rows)
        slots_list = slots.tolist()
        k_to_s     = self._key_to_slot[action]
        s_to_k     = self._slot_to_key[action]
        for slot, key in zip(slots_list, new_keys):
            # Evict the previous occupant of this slot (ring-buffer overwrite).
            old_key = s_to_k.get(slot)
            if old_key is not None:
                k_to_s.pop(old_key, None)
            # Register the new entry.
            k_to_s[key]  = slot
            s_to_k[slot] = key

        self._write_ptrs[action] = int((ptr + n) % self.capacity)
        self._sizes[action]      = min(self._sizes[action] + n, self.capacity)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        d: dict = {
            "num_actions":   self.num_actions,
            "capacity":      self.capacity,
            "k":             self.k,
            "device":        self.device,
            "key_scale":     self._key_scale,
            "_sizes":        list(self._sizes),
            "_write_ptrs":   list(self._write_ptrs),
            "action_states": None,
            "action_values": None,
        }
        if self.states is None:
            return d

        action_states, action_values = [], []
        for a in range(self.num_actions):
            sz  = self._sizes[a]
            ptr = self._write_ptrs[a]
            if sz == 0:
                action_states.append(None)
                action_values.append(np.array([], dtype=np.float64))
                continue
            # Rotate so index 0 = oldest entry (write-order on load).
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
        self._key_scale  = d.get("key_scale", 1e5)   # default for old checkpoints
        self._sizes      = list(d["_sizes"])
        self._write_ptrs = list(d["_write_ptrs"])

        dev = self.device
        self.values = torch.empty(self.num_actions, self.capacity, dtype=torch.float64, device=dev)

        # Initialise empty dicts — rebuilt from tensors below.
        self._key_to_slot = [{} for _ in range(self.num_actions)]
        self._slot_to_key = [{} for _ in range(self.num_actions)]

        if d["action_states"] is None:
            self.states = None
            return

        state_dim = next(
            (s.shape[1] for s in d["action_states"]
             if s is not None and s.ndim == 2 and s.shape[0] > 0),
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

        # Rebuild dicts from the restored tensors (no need to pickle the dicts).
        for a in range(self.num_actions):
            sz = self._sizes[a]
            if sz == 0:
                continue
            keys_a = self._make_keys(self.states[a, :sz])
            k_to_s = self._key_to_slot[a]
            s_to_k = self._slot_to_key[a]
            for slot, key in enumerate(keys_a):
                k_to_s[key]  = slot
                s_to_k[slot] = key
