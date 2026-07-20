"""Neural Episodic Control (NEC).

Pritzel et al. (2017), "Neural Episodic Control"
https://arxiv.org/abs/1703.01988

Key ideas
---------
* **Differentiable Neural Dictionary (DND)**: one table per discrete action,
  each mapping a state *embedding* to a Q-value estimate.  Lookup produces a
  kernel-weighted average over the k nearest stored embeddings:

      Q(s, a) = Σ_i w_i Q_i / Σ_i w_i,  w_i = 1 / (‖h − h_i‖² + δ)

  where h = embedding_network(s) and δ = 1e-3 prevents division by zero.

* **Trainable CNN encoder**: unlike MFEC's fixed random projection, the
  embedding network is a NatureDQN-style ConvNet trained end-to-end.

* **N-step returns**: for step t in an episode of length T,

      Q^(N)(s_t, a_t) = Σ_{j=0}^{N-1} γ^j r_{t+j} + γ^N max_{a'} Q(s_{t+N}, a')

  Bootstrap from the DND for t + N < T; full Monte Carlo otherwise.

* **Dual DND updates**:
  (a) Online write after each episode: blend existing entries
      Q_i ← Q_i + α(Q^(N) − Q_i),  α = dnd_lr;
      insert novel embeddings via a ring buffer (LRU eviction).
  (b) Gradient descent on a regression loss (predicted Q̂ vs. N-step target)
      backpropagates through both the kernel-weighted combination and the CNN.

Algorithm pseudocode (matches the paper)
-----------------------------------------
Initialise:
    embedding_network φ    # trainable CNN
    DND[a] for each action a   # (key, value) ring-buffer with exact-match dict
    replay_buffer D

For each step:
    h_t ← φ(s_t)
    a_t ← ε-greedy w.r.t. Q̂(s_t, ·) via DND kernel lookup
    execute a_t, observe r_{t+1}

At episode end:
    Compute N-step returns Q^(N)_t for each t
    For each t:
        if h_t exact-matches a slot in DND[a_t]:
            DND[a_t][h_t] ← DND[a_t][h_t] + α(Q^(N)_t − DND[a_t][h_t])
        else:
            insert (h_t, Q^(N)_t) into DND[a_t]  # ring-buffer with LRU eviction
        append (s_t, a_t, Q^(N)_t) to D

    For each gradient step:
        (s, a, target) ← sample D
        h ← φ(s)
        find k nearest keys in DND[a]   # no_grad
        Q̂(s, a) ← Σ w_i(h) v_i / Σ w_i(h)  # v_i frozen; grad → φ via ∂w_i/∂h
        ∇ θ = (Q̂ − target)²
        update θ (embedding network only — DND values updated by blend rule)

Reference implementation deviations
------------------------------------
The GitHub repo at github.com/EndingCredits/Neural-Episodic-Control explicitly
documents these deviations from the paper (README + source comments):

1. Existing DND entries are *not* blended/updated on write.
2. DND values are *not* updated by backpropagation; only the embedding network
   is trained.
3. Slightly different episode-reset handling.

This implementation follows the paper on (1) and (3). It does **not** follow
the paper on (2): the paper's §3.4 has gradients update both the embedding
network *and* the DND keys/values (values at a lower LR than the blend rate
α). Here, ``DND.values`` is a plain (non-grad) tensor updated only by the
blend rule — see the ``DND`` class docstring for why: making ``values`` a
gradient-enabled Adam parameter conflicted with the ring-buffer's in-place
overwrites (a newly-inserted entry inherits Adam's stale per-slot momentum
from whatever it evicted) and drove values negative. This is the same
deviation the reference repo makes, for a related reason.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks import NatureEmbedding


# ---------------------------------------------------------------------------
# Differentiable Neural Dictionary
# ---------------------------------------------------------------------------

class DND:
    """Differentiable Neural Dictionary — fused GPU tensors, grad-enabled values.

    Extends the QEC ring-buffer / exact-match-dict / chunked-kNN pattern from
    mfec.py with one NEC-specific change:

    ``estimate_all()`` uses the inverse-distance kernel instead of a plain
    average: w_i = 1 / (‖h − h_i‖² + δ).

    ``values`` is a plain (no grad) tensor updated only by the in-place blend
    rule Q_i ← Q_i + α(G − Q_i).  The regression loss gradient reaches the
    CNN via ∂w_i/∂h (distance term); stored Q-values are frozen constants in
    that computation.  Including ``values`` in Adam conflicted with the blend
    writes and caused values to drift negative (stale momentum applied to
    post-blend values).

    Storage layout mirrors QEC:

        keys   : (num_actions, capacity, embedding_dim)  float32   — no grad
        values : (num_actions, capacity)                 float32   — no grad (blend-only updates)

    Parameters
    ----------
    num_actions  : int
    capacity     : int — max entries per action
    k            : int — number of nearest neighbours for kNN fallback
    kernel_delta : float — δ in the inverse-distance kernel (1e-3 per paper)
    device       : torch.device
    key_scale    : float — quantisation precision for exact-match hash keys
    """

    _CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        num_actions: int,
        capacity:    int,
        k:           int,
        kernel_delta: float,
        device:      torch.device,
        key_scale:   float = 1e5,
    ) -> None:
        self.num_actions  = num_actions
        self.capacity     = capacity
        self.k            = k
        self.kernel_delta = kernel_delta
        self.device       = device
        self._key_scale   = key_scale

        self.keys: torch.Tensor | None = None  # (A, C, d) — lazy init, no grad

        # Plain (no grad) tensor.  Values are updated only via the in-place
        # blend rule Q_i ← Q_i + α(G - Q_i) and ring-buffer inserts.
        # The gradient of the regression loss flows into the CNN embedding
        # network through the distance term ‖h − h_i‖²; the stored Q-values
        # act as frozen scalars in that computation.
        self.values = torch.zeros(num_actions, capacity, device=device)

        self._sizes      = [0] * num_actions
        self._write_ptrs = [0] * num_actions

        self._key_to_slot: list[dict[bytes, int]] = [{} for _ in range(num_actions)]
        self._slot_to_key: list[dict[int, bytes]] = [{} for _ in range(num_actions)]

    # ------------------------------------------------------------------
    # Key generation (identical to QEC._make_keys)
    # ------------------------------------------------------------------

    def _make_keys(self, states: torch.Tensor) -> list[bytes]:
        """Quantise float32 embeddings to stable bytes keys (B rows → B keys)."""
        q = torch.round(states * self._key_scale).to(torch.int32)
        q_cpu = q.cpu().contiguous()
        return [q_cpu[i].numpy().tobytes() for i in range(q_cpu.shape[0])]

    # ------------------------------------------------------------------
    # Lazy key-tensor initialisation
    # ------------------------------------------------------------------

    def _init_keys(self, embedding_dim: int) -> None:
        if self.keys is None:
            self.keys = torch.empty(
                self.num_actions, self.capacity, embedding_dim,
                dtype=torch.float32, device=self.device,
            )

    # ------------------------------------------------------------------
    # Q-value estimation — inverse-distance kernel, for policy inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_all(self, queries: torch.Tensor) -> torch.Tensor:
        """Estimate Q(s, a) for all actions via kernel-weighted kNN lookup.

        Per-query, per-action strategy:
          1. Exact-match dict: O(1) — return stored value directly.
          2. kNN + kernel weighting for queries not in the dict.
          3. +∞ for actions with too few stored entries (optimistic init).

        The kNN step is batched across all actions in a single
        :meth:`knn_all_actions` call rather than looping over actions in
        Python — see that method's docstring for why.

        Parameters
        ----------
        queries : (B, d) float32

        Returns
        -------
        (B, A) float32 — on queries.device; +inf where memory is too sparse
        """
        B     = queries.shape[0]
        A     = self.num_actions
        dev_q = queries.device

        max_size = max(self._sizes) if self._sizes else 0
        if self.keys is None or max_size == 0:
            return torch.full((B, A), float("inf"), dtype=torch.float32, device=dev_q)

        if queries.device != self.device:
            queries = queries.to(self.device)

        keys_b = self._make_keys(queries)

        # --- Exact-match lookups (cheap CPU dict ops; unchanged) -----------
        hit_mask = torch.zeros(A, B, dtype=torch.bool, device=self.device)
        hit_vals = torch.zeros(A, B, dtype=torch.float32, device=self.device)
        for a in range(A):
            k_to_s = self._key_to_slot[a]
            hit_b:    list[int] = []
            hit_slots: list[int] = []
            for b, key in enumerate(keys_b):
                slot = k_to_s.get(key)
                if slot is not None:
                    hit_b.append(b)
                    hit_slots.append(slot)
            if hit_b:
                hit_b_t = torch.tensor(hit_b,     dtype=torch.long, device=self.device)
                hit_s_t = torch.tensor(hit_slots, dtype=torch.long, device=self.device)
                hit_mask[a, hit_b_t] = True
                hit_vals[a, hit_b_t] = self.values.data[a, hit_s_t].float()

        # --- kNN, batched across ALL actions in one pass -------------------
        # k+1: column 0 is the nearest neighbour (used for the near-exact
        # check), columns 0..k-1 feed the kernel-weighted average.
        dists, idx = self.knn_all_actions(queries, self.k + 1)  # (A, B, <=k+1)
        cols  = dists.shape[-1]
        k_use = min(self.k, cols)

        a_idx    = torch.arange(A, device=self.device).view(A, 1, 1).expand_as(idx)
        knn_vals = self.values.data[a_idx, idx].float()          # (A, B, cols)

        near_exact = dists[..., 0] < 1e-5                        # (A, B)
        dists_sq   = dists[..., :k_use] ** 2                     # (A, B, k_use)
        weights    = 1.0 / (dists_sq + self.kernel_delta)
        knn_q      = (weights * knn_vals[..., :k_use]).sum(-1) / weights.sum(-1)
        exact_val  = knn_vals[..., 0]

        knn_result = torch.where(near_exact, exact_val, knn_q)   # (A, B)

        sizes_t     = torch.tensor(self._sizes, device=self.device)
        sparse_mask = (sizes_t <= self.k).view(A, 1)              # too sparse → +inf
        result = torch.where(
            sparse_mask, torch.full_like(knn_result, float("inf")), knn_result
        )
        result = torch.where(hit_mask, hit_vals, result)          # exact hits always win

        return result.T.to(dev_q)  # (B, A)

    # ------------------------------------------------------------------
    # kNN across ALL actions for a shared query set (batched)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_all_actions(
        self,
        queries: torch.Tensor,  # (B, d)
        k:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked kNN (exact L2) over stored keys, batched across actions.

        Used by ``estimate_all`` (ε-greedy action selection, called once per
        collector step): every query there is checked against every
        action's table regardless — that's what "Q(s, a) for all actions"
        means — so batching the ``num_actions`` sequential ``knn_action``
        calls into one batched ``cdist`` per capacity-chunk removes GPU
        kernel-launch/sync overhead for free, without computing anything
        extra. NOT used by ``NECAlgorithm._gradient_step``: there, each row
        only ever needs its own action's table, so broadcasting against all
        tables would mean doing up to ``num_actions``x more distance
        computations for no benefit — that call site keeps the per-action
        ``knn_action`` loop instead.

        Slots beyond an action's current size are masked to +inf distance so
        they never surface in the top-k; callers must still gate "too
        sparse" actions (``size_a <= k``) themselves, exactly as with the
        per-action ``knn_action``.

        Parameters
        ----------
        queries : (B, d) float32 — same query set evaluated against every
            action's table.
        k : requested number of neighbours (may be reduced if no action
            holds at least that many entries).

        Returns
        -------
        dists   : (A, B, k_eff) L2 distances
        indices : (A, B, k_eff) slot indices into self.keys[action]
        """
        A = self.num_actions
        B = queries.shape[0]
        d = queries.shape[1]

        if queries.device != self.device:
            queries = queries.to(self.device)

        sizes_t  = torch.tensor(self._sizes, device=self.device)
        max_size = int(sizes_t.max().item()) if self._sizes else 0
        k_eff    = min(k, max_size)

        if k_eff <= 0:
            return (
                torch.full((A, B, 0), float("inf"), device=self.device),
                torch.zeros((A, B, 0), dtype=torch.long, device=self.device),
            )

        chunk_size = max(1, self._CHUNK_BYTES // (A * B * 4))
        chunk_size = min(chunk_size, max_size)

        best_dists = torch.full((A, B, k_eff), float("inf"), device=self.device)
        best_idx   = torch.zeros((A, B, k_eff), dtype=torch.long,  device=self.device)

        q_exp = queries.unsqueeze(0).expand(A, B, d)

        for cs in range(0, max_size, chunk_size):
            ce = min(cs + chunk_size, max_size)
            cd = torch.cdist(q_exp, self.keys[:, cs:ce, :])       # (A, B, ce-cs)

            slot_idx = torch.arange(cs, ce, device=self.device).view(1, 1, -1)
            valid    = slot_idx < sizes_t.view(A, 1, 1)
            cd       = cd.masked_fill(~valid, float("inf"))

            ck = min(k_eff, ce - cs)
            chd, chi = cd.topk(ck, dim=-1, largest=False)
            chi = chi + cs

            merged_d = torch.cat([best_dists, chd], dim=-1)
            merged_i = torch.cat([best_idx,   chi], dim=-1)
            _, keep  = merged_d.topk(k_eff, dim=-1, largest=False)
            best_dists = merged_d.gather(-1, keep)
            best_idx   = merged_i.gather(-1, keep)

        return best_dists, best_idx

    # ------------------------------------------------------------------
    # kNN for a single action (identical to QEC.knn_action) — used by
    # NECAlgorithm._gradient_step, where each row only needs its own
    # action's table (see knn_all_actions' docstring for why that call
    # site does NOT use the batched-across-actions helper).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_action(
        self,
        queries: torch.Tensor,  # (m, d)
        action:  int,
        k:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked kNN (exact L2) over stored keys for one action.

        Returns
        -------
        dists   : (m, k_eff) L2 distances
        indices : (m, k_eff) slot indices into self.keys[action]
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
            cd  = torch.cdist(queries, self.keys[action, cs:ce])
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
    # Write — blend existing, insert novel (ring-buffer + dict maintenance)
    # ------------------------------------------------------------------

    def write_batch(
        self,
        action: int,
        states: torch.Tensor,  # (n, d) float32
        values: torch.Tensor,  # (n,)   float32
        dnd_lr: float,
    ) -> tuple[int, int]:
        """Write N-step return estimates into the DND for one action.

        Existing entries (exact hash match): blend with DND learning rate α.
        Novel entries: insert via ring buffer (evict oldest on overflow).

        Uses ``.data`` for all in-place tensor writes so the autograd graph
        for the gradient-step path remains intact.

        Returns
        -------
        (n_hits, n_novel)
        """
        n = len(states)
        if n == 0:
            return 0, 0

        if states.device != self.device:
            states = states.to(self.device)
        values = values.to(self.device)

        self._init_keys(states.shape[1])

        keys_b  = self._make_keys(states)
        k_to_s  = self._key_to_slot[action]

        hit_indices:  list[int] = []
        hit_slots:    list[int] = []
        novel_indices: list[int] = []

        for i, key in enumerate(keys_b):
            slot = k_to_s.get(key)
            if slot is not None:
                hit_indices.append(i)
                hit_slots.append(slot)
            else:
                novel_indices.append(i)

        # --- Blend existing entries (paper §2.3 tabular Q-learning update) --
        if hit_indices:
            hit_idx_t   = torch.tensor(hit_indices, dtype=torch.long, device=self.device)
            hit_slots_t = torch.tensor(hit_slots,   dtype=torch.long, device=self.device)
            old = self.values.data[action, hit_slots_t]
            new = values[hit_idx_t].to(old.dtype)
            self.values.data[action, hit_slots_t] = old + dnd_lr * (new - old)

        # --- Insert novel entries via ring buffer ---------------------------
        if novel_indices:
            nov_idx_t = torch.tensor(novel_indices, dtype=torch.long, device=self.device)
            self._insert_novel(action, states[nov_idx_t], values[nov_idx_t])

        return len(hit_indices), len(novel_indices)

    def _insert_novel(
        self,
        action: int,
        states: torch.Tensor,  # (n, d) float32, already on self.device
        values: torch.Tensor,  # (n,)   float32
    ) -> None:
        n = len(states)
        if n > self.capacity:
            states = states[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity

        ptr   = self._write_ptrs[action]
        slots = torch.arange(ptr, ptr + n, device=self.device) % self.capacity
        self.keys[action, slots]         = states
        self.values.data[action, slots]  = values.to(self.values.dtype)

        new_keys   = self._make_keys(states)
        slots_list = slots.tolist()
        k_to_s     = self._key_to_slot[action]
        s_to_k     = self._slot_to_key[action]
        for slot, key in zip(slots_list, new_keys):
            old_key = s_to_k.get(slot)
            if old_key is not None:
                k_to_s.pop(old_key, None)
            k_to_s[key]  = slot
            s_to_k[slot] = key

        self._write_ptrs[action] = int((ptr + n) % self.capacity)
        self._sizes[action]      = min(self._sizes[action] + n, self.capacity)

    # ------------------------------------------------------------------
    # Serialisation (mirrors QEC.__getstate__ / __setstate__)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        d: dict = {
            "num_actions":  self.num_actions,
            "capacity":     self.capacity,
            "k":            self.k,
            "kernel_delta": self.kernel_delta,
            "device":       self.device,
            "key_scale":    self._key_scale,
            "_sizes":       list(self._sizes),
            "_write_ptrs":  list(self._write_ptrs),
            "action_keys":  None,
            "action_values": None,
        }
        if self.keys is None:
            return d

        action_keys, action_values = [], []
        for a in range(self.num_actions):
            sz  = self._sizes[a]
            ptr = self._write_ptrs[a]
            if sz == 0:
                action_keys.append(None)
                action_values.append(np.array([], dtype=np.float32))
                continue
            if sz == self.capacity and ptr != 0:
                k_t = torch.roll(self.keys[a],         -ptr, dims=0)[:sz]
                v_t = torch.roll(self.values.data[a],  -ptr, dims=0)[:sz]
            else:
                k_t = self.keys[a, :sz]
                v_t = self.values.data[a, :sz]
            action_keys.append(k_t.cpu().numpy())
            action_values.append(v_t.cpu().numpy())

        d["action_keys"]   = action_keys
        d["action_values"] = action_values
        return d

    def __setstate__(self, d: dict) -> None:
        self.num_actions  = d["num_actions"]
        self.capacity     = d["capacity"]
        self.k            = d["k"]
        self.kernel_delta = d["kernel_delta"]
        self.device       = d["device"]
        self._key_scale   = d.get("key_scale", 1e5)
        self._sizes       = list(d["_sizes"])
        self._write_ptrs  = list(d["_write_ptrs"])

        dev = self.device
        self.values = torch.zeros(self.num_actions, self.capacity, device=dev)

        self._key_to_slot = [{} for _ in range(self.num_actions)]
        self._slot_to_key = [{} for _ in range(self.num_actions)]

        if d["action_keys"] is None:
            self.keys = None
            return

        embedding_dim = next(
            (k.shape[1] for k in d["action_keys"]
             if k is not None and k.ndim == 2 and k.shape[0] > 0),
            None,
        )
        if embedding_dim is None:
            self.keys = None
            return

        self.keys = torch.empty(
            self.num_actions, self.capacity, embedding_dim,
            dtype=torch.float32, device=dev,
        )
        for a, (k_np, v_np) in enumerate(zip(d["action_keys"], d["action_values"])):
            sz = self._sizes[a]
            if sz > 0 and k_np is not None:
                self.keys[a, :sz]        = torch.from_numpy(k_np).to(dev)
                self.values.data[a, :sz] = torch.from_numpy(v_np).to(dev)

        for a in range(self.num_actions):
            sz = self._sizes[a]
            if sz == 0:
                continue
            keys_a = self._make_keys(self.keys[a, :sz])
            k_to_s = self._key_to_slot[a]
            s_to_k = self._slot_to_key[a]
            for slot, key in enumerate(keys_a):
                k_to_s[key]  = slot
                s_to_k[slot] = key


# ---------------------------------------------------------------------------
# Policy module
# ---------------------------------------------------------------------------

class DNDPolicy(nn.Module):
    """nn.Module that estimates Q(s,a) for all actions via DND kernel lookup.

    ``forward()`` is decorated @torch.no_grad() for efficient action selection
    during collection.  The training gradient path uses the embedding_net and
    DND directly (not through this module) so gradients can flow.
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        dnd: DND,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.dnd           = dnd
        self.num_actions   = num_actions

    def __deepcopy__(self, memo):
        # Collector deepcopies the policy for data collection.  Return self
        # so the collector always uses the current network and DND state.
        memo[id(self)] = self
        return self

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        leading   = obs.shape[:-3]                             # dims before (C,H,W)
        obs_flat  = obs.reshape(-1, *obs.shape[-3:]).float()   # (B, C, H, W)
        h         = self.embedding_net(obs_flat)               # (B, d)
        h         = nn.functional.normalize(h, dim=-1)        # unit-norm per paper §2
        q_values  = self.dnd.estimate_all(h)                   # (B, A)
        q_values  = torch.where(
            torch.isinf(q_values),
            torch.full_like(q_values, 1e9),
            q_values,
        )
        if leading:
            return q_values.reshape(*leading, self.num_actions)
        return q_values.squeeze(0) if q_values.shape[0] == 1 else q_values


class _SharedPolicy(TensorDictSequential):
    """Returns self on deepcopy so the EGreedyModule is shared with the collector."""

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# ---------------------------------------------------------------------------
# N-step return helper
# ---------------------------------------------------------------------------

def _compute_n_step_returns(
    rewards: np.ndarray,        # (T,) float64 — complete episode
    states:  torch.Tensor,      # (T, d) float32 on buffer_device
    gamma:   float,
    n_step:  int,
    dnd:     DND,
) -> np.ndarray:
    """N-step discounted returns for a complete episode.

    For step t with t + n_step < T:
        Q^(N)_t = MC_t + γ^N (max_a Q(s_{t+N}) − MC_{t+N})

    For steps T − n_step ≤ t < T (near end of episode) or when the DND is
    too sparse (Q = +inf): fall back to the full Monte Carlo return MC_t.

    This reuses lfilter from MFEC so the computation pattern is the same.
    The DND query uses ``@torch.no_grad()`` — gradients are computed later
    from the replay buffer.
    """
    T  = len(rewards)
    mc = lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()

    if T <= n_step:
        return mc

    boot_states = states[n_step:].to(dnd.device)      # (T-n_step, d)
    with torch.no_grad():
        q_all  = dnd.estimate_all(boot_states)         # (T-n_step, A) float32
        q_max  = q_all.max(dim=-1).values.cpu().numpy().astype(np.float64)

    gamma_n = float(gamma ** n_step)
    n_step_G = mc.copy()
    valid    = ~np.isinf(q_max)
    correction = gamma_n * (q_max - mc[n_step:])
    n_step_G[:T - n_step] = np.where(
        valid, mc[:T - n_step] + correction, mc[:T - n_step]
    )
    return n_step_G


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class NECAlgorithm(BaseAlgorithm):
    """Neural Episodic Control.

    Implements the BaseAlgorithm interface for StepTrainer.  The trainer calls
    ``setup()`` once, then ``step()`` with each collected batch.

    Design notes
    ------------
    * ``embedding_network`` factory is called as
      ``embedding_network(obs_shape, embedding_dim)`` in ``setup()``.
    * ``replay_buffer`` is a no-arg factory returning a ``ReplayBuffer``.
    * The optimizer covers the CNN embedding network only.  DND values are
      updated by the in-place blend rule, not by gradient descent.
    * Per-env carry-over logic (the ``_carry`` list) is taken directly from
      ``MFECAlgorithm.step()`` and extended to also buffer raw observations so
      they can be re-embedded with the *current* network on each step call.
    * N-step returns are computed per-episode (never across episode boundaries).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Design choices: factories injected as Callables ---------------
        embedding_network: Callable[[tuple[int, ...], int], nn.Module] = (
            NatureEmbedding
        ),
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=100_000, device="cpu"),
        ),
        obs_key: str = "pixels",
        # --- DND parameters ------------------------------------------------
        embedding_dim: int = 64,
        dnd_capacity:  int = 500_000,   # entries per action
        k:             int = 50,        # nearest neighbours for kNN lookup
        kernel_delta:  float = 1e-3,    # δ in inverse-distance kernel
        dnd_lr:        float = 0.1,     # α for blending existing DND entries
        # --- N-step return -------------------------------------------------
        n_step: int = 100,
        # --- Optimisation --------------------------------------------------
        lr:            float = 1e-4,
        gamma:         float = 0.99,
        batch_size:    int   = 32,
        max_grad_norm: float = 10.0,
        # --- Exploration ---------------------------------------------------
        eps_start:        float = 1.0,
        eps_end:          float = 0.01,
        annealing_frames: int   = 4_000_000,
        # --- Data collection -----------------------------------------------
        frames_per_batch:    int = 1_600,
        init_random_frames:  int = 50_000,
        max_frames_per_traj: int = -1,
        num_updates:         int = 100,
    ) -> None:
        super().__init__(device)
        self._make_embedding_network = embedding_network
        self._make_replay_buffer     = replay_buffer
        self.obs_key         = obs_key
        self.embedding_dim   = embedding_dim
        self.dnd_capacity    = dnd_capacity
        self.k               = k
        self.kernel_delta    = kernel_delta
        self.dnd_lr          = dnd_lr
        self.n_step          = n_step
        self.lr              = lr
        self.gamma           = gamma
        self.batch_size      = batch_size
        self.max_grad_norm   = max_grad_norm
        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.annealing_frames = annealing_frames
        self.frames_per_batch    = frames_per_batch
        self.init_random_frames  = init_random_frames
        self.max_frames_per_traj = max_frames_per_traj
        self.num_updates         = num_updates
        self._collected_frames   = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env   = make_env()
        # Strip leading parallel-env batch dims so _obs_shape is the per-sample
        # frame shape (e.g. (4, 84, 84)), not (num_envs, 4, 84, 84).
        # Same pattern as MFECAlgorithm.setup().
        env_bs      = proof_env.batch_size
        obs_shape   = tuple(proof_env.observation_spec[self.obs_key].shape)[len(env_bs):]
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)
        self._num_actions = num_actions
        self._obs_shape   = obs_shape

        buf_dev = self.device if self.device is not None else torch.device("cpu")
        self._buffer_device = buf_dev

        # 1. Embedding network (CNN trunk + dense layer)
        self.embedding_net = self._make_embedding_network(
            obs_shape, self.embedding_dim
        ).to(self.device)

        # 2. DND (one set of ring-buffer + exact-match dict per action)
        self.dnd = DND(
            num_actions=num_actions,
            capacity=self.dnd_capacity,
            k=self.k,
            kernel_delta=self.kernel_delta,
            device=buf_dev,
        )

        # 3. Policy wiring: DNDPolicy → QValueActor → EGreedyModule
        self._dnd_policy = DNDPolicy(self.embedding_net, self.dnd, num_actions)
        self.q_actor = QValueActor(
            module=self._dnd_policy,
            spec=action_spec,
            in_keys=[self.obs_key],
        )
        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=buf_dev,
        )
        self._explore_policy = _SharedPolicy(self.q_actor, self.greedy_module)

        # 4. Replay buffer — stores (obs, action, n_step_return)
        self.replay_buffer = self._make_replay_buffer()

        # 5. Optimizer — CNN embedding network only.
        # DND values are updated exclusively by the in-place blend rule;
        # including them in Adam causes the momentum state to conflict with
        # the blend writes and drives stored Q-values negative.
        self.optimizer = torch.optim.Adam(
            self.embedding_net.parameters(),
            lr=self.lr,
        )

        # 6. Per-env carry buffers for partial episodes across batch boundaries
        self._num_envs: int = int(env_bs[0]) if len(env_bs) == 1 else 1
        self._carry: list[dict | None] = [None] * self._num_envs

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: anneal ε, compute N-step returns, update DND,
        fill replay buffer, and run gradient steps after warm-up.

        The (E, T) per-env structure from mfec.py is preserved exactly:
        - Returns are computed only over COMPLETE episodes.
        - Partial trailing episodes are buffered in ``_carry`` and prepended
          to the corresponding env's data in the next call.
        - Raw observations are stored (not embeddings) so the carry can be
          re-embedded with the CURRENT network on each step() call.
        - N-step returns are computed per-episode via lfilter + DND bootstrap.
        """
        bs = batch.batch_size
        if len(bs) == 2:
            E, T = int(bs[0]), int(bs[1])
        elif len(bs) == 1:
            E, T = 1, int(bs[0])
        else:
            raise ValueError(f"Unexpected batch shape: {bs}")
        n = E * T

        self.greedy_module.step(n)
        self._collected_frames += n

        dev = self._buffer_device

        # Retrieve raw observations — shape (E, T, *obs_shape) or (T, *obs_shape)
        obs_batch = batch[self.obs_key]
        obs_shape = tuple(obs_batch.shape[len(bs):])
        obs_2d    = obs_batch.reshape(E, T, *obs_shape)

        rewards_2d = (batch["next", "reward"].cpu().numpy()
                      .flatten().astype(np.float64).reshape(E, T))
        dones_2d   = (batch["next", "done"].cpu().numpy()
                      .flatten().astype(bool).reshape(E, T))
        actions_2d = batch["action"].to(dev).reshape(n).long().reshape(E, T)

        if len(self._carry) != E:
            self._carry = [None] * E

        # --- Per-env episode processing -----------------------------------
        # Mirrors the (E, T) pattern from MFECAlgorithm.step() with the
        # additional carry of raw observations for fresh re-embedding.

        collect_obs:     list[torch.Tensor] = []
        collect_actions: list[torch.Tensor] = []
        collect_returns: list[np.ndarray]   = []

        total_hits   = 0
        total_writes = 0

        for env_idx in range(E):
            obs_e     = obs_2d[env_idx].cpu()                   # (T, *obs_shape)
            rewards_e = rewards_2d[env_idx]                     # (T,) f64 numpy
            dones_e   = dones_2d[env_idx]                       # (T,) bool numpy
            actions_e = actions_2d[env_idx]                     # (T,) long

            carry = self._carry[env_idx]
            if carry is not None:
                obs_e     = torch.cat([carry["obs"], obs_e], dim=0)
                rewards_e = np.concatenate([carry["rewards"], rewards_e])
                dones_e   = np.concatenate([carry["dones"],   dones_e])
                actions_e = torch.cat([carry["actions"], actions_e], dim=0)

            ends = np.flatnonzero(dones_e)

            if len(ends) == 0:
                self._carry[env_idx] = {
                    "obs":     obs_e,                  # intentionally CPU — raw pixels, re-embedded each step
                    "rewards": rewards_e,
                    "dones":   dones_e,
                    "actions": actions_e.to(dev),
                }
                continue

            last = int(ends[-1])

            # Embed the complete portion with the current network
            obs_complete = obs_e[:last + 1].to(self.device).float()  # (T_c, *obs)
            with torch.no_grad():
                h_complete = self.embedding_net(
                    obs_complete.reshape(last + 1, *obs_shape)
                ).to(dev)                                          # (T_c, d)
                h_complete = nn.functional.normalize(h_complete, dim=-1)  # unit-norm per paper §2

            # Process each complete episode individually
            ep_start = 0
            for ep_end in ends:
                ep_end = int(ep_end)

                r_ep = rewards_e[ep_start: ep_end + 1]             # (L,) f64
                h_ep = h_complete[ep_start: ep_end + 1]            # (L, d)
                a_ep = actions_e[ep_start: ep_end + 1]             # (L,) long

                # N-step returns for this episode
                nsr = _compute_n_step_returns(
                    r_ep, h_ep, self.gamma, self.n_step, self.dnd
                )  # (L,) f64

                # Write N-step returns into the DND (blend/insert per action)
                nsr_t = torch.tensor(nsr, dtype=torch.float32, device=dev)
                sorted_idx     = torch.argsort(a_ep, stable=True)
                sorted_states  = h_ep[sorted_idx]
                sorted_values  = nsr_t[sorted_idx]
                sorted_actions = a_ep[sorted_idx]

                counts  = torch.bincount(sorted_actions, minlength=self._num_actions)
                offsets = torch.zeros(self._num_actions + 1, dtype=torch.long, device=dev)
                offsets[1:] = counts.cumsum(0)
                offsets_cpu = offsets.cpu().tolist()

                for a in range(self._num_actions):
                    seg_s, seg_e = offsets_cpu[a], offsets_cpu[a + 1]
                    if seg_s == seg_e:
                        continue
                    h_i, n_h = self.dnd.write_batch(
                        a,
                        sorted_states[seg_s:seg_e],
                        sorted_values[seg_s:seg_e],
                        self.dnd_lr,
                    )
                    total_hits   += h_i
                    total_writes += n_h

                # Collect for replay buffer storage
                collect_obs.append(obs_e[ep_start: ep_end + 1].cpu())
                collect_actions.append(a_ep.cpu())
                collect_returns.append(nsr)

                ep_start = ep_end + 1

            # Buffer trailing partial episode
            if last < len(obs_e) - 1:
                self._carry[env_idx] = {
                    "obs":     obs_e[last + 1:],              # intentionally CPU — raw pixels, re-embedded each step
                    "rewards": rewards_e[last + 1:],
                    "dones":   dones_e[last + 1:],
                    "actions": actions_e[last + 1:].to(dev),
                }
            else:
                self._carry[env_idx] = None

        # --- Store complete episodes in replay buffer ----------------------
        for ep_obs, ep_act, ep_ret in zip(collect_obs, collect_actions, collect_returns):
            T_ep = len(ep_obs)
            episode_td = TensorDict(
                {
                    self.obs_key:   ep_obs,
                    "action":       ep_act.long(),
                    "n_step_return": torch.tensor(ep_ret, dtype=torch.float32),
                },
                batch_size=[T_ep],
                device="cpu",
            )
            self.replay_buffer.extend(episode_td)

        # --- Early exit if no complete episode or still in warm-up ----------
        dnd_size = float(np.mean(self._sizes_summary()))
        base_metrics = {
            "train/epsilon":   float(self.greedy_module.eps),
            "train/dnd_size":  dnd_size,
            "train/dnd_blend_rate": (
                total_hits / (total_hits + total_writes)
                if (total_hits + total_writes) > 0
                else 0.0
            ),
        }

        if self._collected_frames < self.init_random_frames:
            return base_metrics

        # --- Gradient steps ------------------------------------------------
        losses  = torch.zeros(self.num_updates, device=self.device)
        q_vals  = torch.zeros(self.num_updates, device=self.device)
        for j in range(self.num_updates):
            loss_val, mean_q = self._gradient_step()
            losses[j] = loss_val
            q_vals[j]  = mean_q

        return {
            **base_metrics,
            "train/q_loss":   losses.mean().item(),
            "train/q_values": q_vals.mean().item(),
        }

    def _gradient_step(self) -> tuple[float, float]:
        """One minibatch gradient update on the embedding network.

        Samples (obs, action, n_step_return) from the replay buffer, re-embeds
        observations with the *current* embedding network, computes the
        kernel-weighted Q̂ via differentiable indexing into DND.values, and
        minimises MSE(Q̂, n_step_return_target).

        Gradients flow through the embedding network (CNN parameters) via the
        distance term ‖h − h_i‖² where h = embedding_net(obs).  The stored
        DND values Q_i are frozen constants here; they are updated separately
        by the in-place blend rule in step().

        Returns
        -------
        (loss, mean_q) — scalar MSE loss and mean kernel-weighted Q-estimate
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        sample  = self.replay_buffer.sample(self.batch_size)
        obs     = sample[self.obs_key].to(self.device).float()
        actions = sample["action"].long().flatten().to(self.device)
        targets = sample["n_step_return"].float().flatten().to(self.device)

        obs_flat = obs.reshape(-1, *self._obs_shape)

        # Forward through embedding network — gradients enabled
        h = self.embedding_net(obs_flat)                      # (B, embedding_dim)
        h = nn.functional.normalize(h, dim=-1)               # unit-norm per paper §2

        # NOTE: this stays a per-action Python loop (unlike estimate_all,
        # which batches across actions via dnd.knn_all_actions). Batching it
        # the same way would mean broadcasting every row's query against
        # EVERY action's table instead of just the table for its own action
        # — an up-to-num_actions-fold increase in raw distance computations
        # against tables that can hold up to dnd_capacity entries each. That
        # FLOP blow-up is not just a constant-overhead cost (unlike the
        # kernel-launch overhead estimate_all's batching removes), so it can
        # easily cost more than the saved launches recover. Measured ~3.8x
        # slower on CPU when tried; left as the per-action loop here.
        q_hat_parts:  list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []

        for a in range(self._num_actions):
            mask = (actions == a)
            n_a  = int(mask.sum())
            if n_a == 0:
                continue
            if self.dnd._sizes[a] <= self.k:
                continue  # too sparse to compute a kernel-weighted Q

            h_a = h[mask]   # (n_a, embedding_dim)
            t_a = targets[mask]

            k_use = min(self.k, self.dnd._sizes[a])

            # Find neighbour indices without tracking gradients
            with torch.no_grad():
                _, indices = self.dnd.knn_action(h_a.detach(), a, k_use)  # (n_a, k_use)

            # Differentiable kernel-weighted combination
            # keys[a, indices]: (n_a, k_use, d) — frozen; grad flows via distances
            neighbor_keys = self.dnd.keys[a, indices]              # (n_a, k_use, d)
            diffs         = h_a.unsqueeze(1) - neighbor_keys       # (n_a, k_use, d)
            dists_sq      = (diffs ** 2).sum(dim=-1)               # (n_a, k_use)
            weights       = 1.0 / (dists_sq + self.kernel_delta)   # (n_a, k_use)
            weights       = weights / weights.sum(dim=-1, keepdim=True)

            # dnd.values[a, indices]: (n_a, k_use) — frozen; no grad into values
            neighbor_vals = self.dnd.values[a, indices]            # (n_a, k_use)
            q_hat_a       = (weights * neighbor_vals).sum(dim=-1)  # (n_a,)

            q_hat_parts.append(q_hat_a)
            target_parts.append(t_a)

        if not q_hat_parts:
            return 0.0, 0.0

        q_hat  = torch.cat(q_hat_parts)
        tgt    = torch.cat(target_parts)
        loss   = ((q_hat - tgt) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.embedding_net.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()

        mean_q = float(q_hat.detach().mean())
        return float(loss.detach()), mean_q

    def _sizes_summary(self) -> list[int]:
        return list(self.dnd._sizes)

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.embedding_net.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={
                "dnd_state":        self.dnd.__getstate__(),
                "collected_frames": self._collected_frames,
                "carry":            self._serialise_carry(),
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.embedding_net.load_state_dict(state.policy_state_dict)
        self.dnd.__setstate__(state.extra["dnd_state"])
        self.optimizer = torch.optim.Adam(
            self.embedding_net.parameters(),
            lr=self.lr,
        )
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self._collected_frames = int(state.extra["collected_frames"])
        if "carry" in state.extra:
            self._deserialise_carry(state.extra["carry"])
        else:
            self._carry = [None] * self._num_envs

    def _serialise_carry(self) -> list:
        out = []
        for c in self._carry:
            if c is None:
                out.append(None)
            else:
                out.append({
                    "obs":     c["obs"].cpu().numpy(),
                    "rewards": c["rewards"],
                    "dones":   c["dones"],
                    "actions": c["actions"].cpu().numpy().astype(np.int64),
                })
        return out

    def _deserialise_carry(self, data: list) -> None:
        self._carry = []
        for c in data:
            if c is None:
                self._carry.append(None)
            else:
                self._carry.append({
                    "obs":     torch.from_numpy(c["obs"]),
                    "rewards": c["rewards"],
                    "dones":   c["dones"],
                    "actions": torch.from_numpy(c["actions"]).long().to(self._buffer_device),
                })
