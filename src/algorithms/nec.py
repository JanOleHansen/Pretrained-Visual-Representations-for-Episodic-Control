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

  Bootstrap from the DND for t + N < T; full Monte Carlo otherwise.  If the
  episode ended by truncation (e.g. a StepCounter cutoff) rather than a true
  terminal, the last N steps additionally bootstrap γ^(T−t) max_a Q(s_T, ·)
  from the DND instead of assuming zero future reward.

* **Dual DND updates**:
  (a) Online write after each episode: blend existing entries
      Q_i ← Q_i + α(Q^(N) − Q_i),  α = dnd_lr;
      insert novel embeddings via a ring buffer (FIFO eviction of the
      oldest entry; the paper specifies LRU — see deviations below).
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
        (truncated episodes bootstrap the tail from Q(s_T, ·); true
        terminals use the plain Monte Carlo tail)
    For each t:
        if h_t exact-matches a slot in DND[a_t]:
            DND[a_t][h_t] ← DND[a_t][h_t] + α(Q^(N)_t − DND[a_t][h_t])
        else:
            insert (h_t, Q^(N)_t) into DND[a_t]  # ring-buffer, FIFO eviction
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

This implementation follows the paper on all three. In particular it does
**not** take the reference repo's deviation (2): gradients DO flow into
``DND.keys`` and ``DND.values``, as paper Figure 2 requires ("Gradients flow
through the entire architecture").

An earlier version of this file froze both, because making them
gradient-enabled Adam parameters conflicted with the ring-buffer's in-place
overwrites — a newly inserted entry inherited Adam's stale per-slot momentum
from whatever it evicted, which drove stored values negative. That is fixed
by construction now: ``DND.apply_gradient`` does a **stateless sparse SGD**
step on only the slots a minibatch retrieved (``dnd_key_lr`` /
``dnd_value_lr``), so there is no per-slot optimiser state to go stale and
untouched slots are left bit-identical. See that method for the two
invariants it restores (unit-norm keys, exact-match hash validity).

Why this matters more than it looks: stored keys were previously write-once
and never refreshed, so with ``dnd_capacity`` = 5e5 and ~178 inserts per
action per collector batch an entry survived ~2800 batches — over a million
gradient steps of CNN drift — while the kNN kept retrieving it as though it
still lived in the current embedding space.

One remaining deviation from the paper: DND eviction is FIFO (ring-buffer
insertion order), not LRU as §3.1 specifies — nothing updates recency on
lookup.

The old justification for FIFO ("stored keys go stale, so evicting the
oldest evicts the stalest") no longer applies now that gradients refresh the
keys. It is left as FIFO on cost/benefit grounds, not principle: **eviction
policy has no effect at all until a table is full**, and at
``dnd_capacity`` = 5e5 with ~178 inserts per action per collector batch that
is ~2800 batches (~4.5M agent steps). Below that the two policies are
bit-identical, so switching would change nothing for any run shorter than
that, while touching the ring-buffer serialisation
(``__getstate__``/``__setstate__`` rotate by ``_write_ptrs``) and the
recency bookkeeping on the hottest path. Worth doing before a full-length
40M-frame run; not worth it to debug one.

One further caveat, caught in review:

4. The exact-match blend rule (a) above is **largely inert in practice**.
   ``write_batch`` blends only on a bit-level hash match of the quantised
   64-d embedding, and the CNN takes ``num_updates`` Adam steps between
   successive ``step()`` calls, so a state re-encountered in a later batch
   essentially never re-hashes to its stored key. Blends therefore only
   happen between duplicate frames embedded within one ``step()`` call, and
   the DND behaves close to an insert-only FIFO log. Making the blend rule
   fire across batches would mean matching within a radius rather than
   exactly, which is a design change, not a bug fix, and is deliberately NOT
   done here.

   ``train/dnd_blend_rate`` measures the residual. An earlier version of this
   note said "expect it near 0"; that is **wrong for Ms. Pac-Man** and reading
   a healthy run as broken is the cost. Atari has long stretches of
   bit-identical frames — the opening "ready" freeze and the pause after each
   death — and those duplicate observations produce duplicate embeddings
   *within* the same ``step()`` call, which is exactly the case the rule does
   fire on. Measured on a 1500-step Ms. Pac-Man rollout: 264/1500 = 17.6% of
   observations are byte-identical to an earlier frame, and the quantised
   embeddings collapse at precisely the same rate (1236 distinct keys for 1236
   distinct frames — no additional collapse from the encoder). A blend rate of
   0.1-0.5 on this game is therefore the expected reading, not evidence of an
   embedding that has stopped discriminating. Watch ``eval/dnd_top_weight``
   for that instead (see ``NECAlgorithm.eval_metrics``).

Where the paper's hyperparameters actually come from
----------------------------------------------------
Pritzel et al. (2017) has no hyperparameter table — its only appendix is
"A. Scores on Atari Games". Everything is prose in §4. Stated there:
p = 50 neighbours, δ = 10⁻³, N = 100, 5×10⁵ memories per action, replay
buffer of the last 10⁵ states, minibatch 32, one replay update per 16
observed frames, γ = 0.99, action repeat 4, and no reward clipping.
Explicitly **swept and never reported**: the SGD learning rate, the
fast-update rate α (``dnd_lr``), the embedding dimensionality, and the
ε-greedy exploration rate. Config comments must not claim paper authority
for those four.
"""

from __future__ import annotations

import warnings
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
from src.algorithms.eval_policy import EvalEGreedyModule
from src.networks import NatureEmbedding


def _topk_l2_unit(
    queries: torch.Tensor,   # (m, d) unit-norm
    keys:    torch.Tensor,   # (n, d) unit-norm
    k:       int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k by L2 distance, via dot products.

    For unit-norm vectors ``‖q − k‖² = 2 − 2·q·k``, so ordering by ascending
    distance is exactly ordering by descending dot product.  That turns the
    search into one GEMM plus a top-k, skipping ``torch.cdist``'s norm
    computation over the whole key block on every single call.

    This is **not** an approximation — the returned distances match
    ``torch.cdist`` to float32 tolerance at every table size tested.  It is
    only valid because NEC normalises every embedding before it touches the
    DND (``NECAlgorithm._embed``, and ``DND.apply_gradient`` re-projects keys
    after each update); ``DND(unit_norm_keys=False)`` restores the cdist path
    for any caller that cannot guarantee that.

    Measured against cdist for k=50 on unit-norm data, at m=4 queries — the
    regime ``_gradient_step`` actually runs in (batch 32 over 9 actions):

        n=10,000  3.9x     n=200,000  5.1x
        n=50,000  3.3x     n=500,000  5.2x

    Numerics: ``2 − 2·sim`` loses relative precision as sim -> 1, but only for
    distances far below ``kernel_delta`` = 1e-3, where the kernel weight is
    delta-dominated anyway and the Q estimate is unaffected.
    """
    sim = queries @ keys.T                       # (m, n)
    top_sim, idx = sim.topk(k, dim=1, largest=True)
    dist = (2.0 - 2.0 * top_sim).clamp_min_(0.0).sqrt_()
    return dist, idx


# ---------------------------------------------------------------------------
# Differentiable Neural Dictionary
# ---------------------------------------------------------------------------

class DND:
    """Differentiable Neural Dictionary — fused GPU tensors, frozen values.

    Extends the QEC ring-buffer / exact-match-dict / chunked-kNN pattern from
    mfec.py with two NEC-specific changes:

    1. ``estimate_all()`` uses the inverse-distance kernel instead of a plain
       average: w_i = 1 / (‖h − h_i‖² + δ).
    2. ``estimate_all()`` has **no exact-match shortcut** — the kernel sum is
       the only Q definition, because ``NECAlgorithm._gradient_step`` has to
       be able to reproduce it differentiably. ``_key_to_slot`` /
       ``_slot_to_key`` are write-path structures here, used only by
       ``write_batch``'s blend rule and ring-buffer eviction. See
       ``estimate_all``'s docstring.

    ``keys`` and ``values`` are plain tensors (no autograd), but they ARE
    updated by the regression loss — see :meth:`apply_gradient`, which does a
    stateless sparse SGD step on just the slots a minibatch retrieved.  Making
    them autograd leaves instead would allocate a dense gradient the size of
    the whole table on every backward (1.15 GB at the Atari defaults).
    ``values`` additionally moves by the in-place blend rule
    Q_i ← Q_i + α(G − Q_i) at episode end.

    Storage layout mirrors QEC:

        keys   : (num_actions, capacity, embedding_dim)  float32
        values : (num_actions, capacity)                 float32

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
        unit_norm_keys: bool = True,
    ) -> None:
        self.num_actions  = num_actions
        self.capacity     = capacity
        self.k            = k
        self.kernel_delta = kernel_delta
        self.device       = device
        self._key_scale   = key_scale
        # Every key and query NEC puts in here is L2-normalised, which lets the
        # kNN use the dot-product identity in _topk_l2_unit (exact, ~5x faster
        # at Atari table sizes).  Set False to force the general cdist path.
        self.unit_norm_keys = unit_norm_keys

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

        # Slots whose key was moved by apply_gradient() and whose exact-match
        # hash is therefore stale; drained by flush_moved_slots().
        self._moved_slots: list[list[torch.Tensor]] = [[] for _ in range(num_actions)]

        self.reset_lookup_stats()

    # ------------------------------------------------------------------
    # Read-path instrumentation
    # ------------------------------------------------------------------

    def reset_lookup_stats(self) -> None:
        """Zero the counters :meth:`lookup_stats` reports and start recording.

        Accumulation is **off by default** and only switched on here, because
        ``estimate_all`` is the ε-greedy hot path — one call per env step per
        collector batch.  ``NECAlgorithm.reset_eval_metrics`` turns it on for
        the duration of an evaluation rollout and ``eval_metrics`` turns it
        back off.

        The running sums stay device tensors and are converted exactly once,
        in :meth:`lookup_stats`.  Reading them back per call would put a forced
        GPU sync on the policy's hot path — the same reason
        ``QEC.reset_lookup_stats`` in mfec.py keeps its near-exact tally on
        device.
        """
        self._record_lookups = False
        self._lookup_queries = 0
        self._lookup_optimistic = 0
        self._lookup_nn_dist = torch.zeros((), dtype=torch.float64, device=self.device)
        self._lookup_top_weight = torch.zeros((), dtype=torch.float64, device=self.device)

    def lookup_stats(self) -> dict[str, float]:
        """Kernel-lookup diagnostics since the last :meth:`reset_lookup_stats`.

        One "query" is one *(state, action)* pair asked of
        :meth:`estimate_all`, i.e. ``|A|`` per frame the policy sees — every
        candidate action, not only the one taken.  Same denominator as
        ``MFECAlgorithm.eval_metrics``, and deliberately not the same as any
        ``train/*`` counter.

        Returns
        -------
        ``{}`` when nothing was recorded, otherwise

        ``queries``
            total (state, action) pairs looked up.
        ``optimistic_rate``
            fraction answered with the ``+inf`` sentinel (action's table still
            at/below ``k``).  Above ~0 late in a run means some action is
            still starved and ``argmax`` is chasing the sentinel.
        ``nn_dist``
            mean L2 distance from the query embedding to its *nearest* stored
            key, over non-sentinel queries.  Every embedding is unit-norm, so
            this lives in ``[0, 2]``; a value that drifts up over training is
            the memory going stale relative to the CNN.
        ``top_weight``
            mean share of the kernel mass carried by that nearest neighbour,
            i.e. ``max_i w_i / Σ_i w_i``.  This is the number that separates
            "the memory holds a bad policy" from "the memory is not being used
            at all": at ``1/k`` = 0.02 the kernel is a flat average over all
            ``k`` neighbours, every action of a state scores alike, and the
            argmax is noise however full the tables are.
        """
        if not self._lookup_queries:
            return {}
        graded = self._lookup_queries - self._lookup_optimistic
        out = {
            "queries":         float(self._lookup_queries),
            "optimistic_rate": self._lookup_optimistic / self._lookup_queries,
        }
        if graded > 0:
            out["nn_dist"]    = float(self._lookup_nn_dist) / graded
            out["top_weight"] = float(self._lookup_top_weight) / graded
        return out

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

        Implements the paper's Eq. (4)/(5) verbatim over the k nearest keys:

            Q(s, a) = Σ_i w_i Q_i / Σ_i w_i,   w_i = 1 / (‖h − h_i‖² + δ)

        Actions holding ``<= k`` entries return +∞ (optimistic init).

        No exact-match shortcut
        -----------------------
        Unlike :meth:`QEC.estimate_all` in mfec.py, this does **not** consult
        ``_key_to_slot`` to short-circuit an exact re-encounter to its stored
        value, and does not special-case a near-zero nearest distance.  Those
        shortcuts exist in QEC because MFEC's Eq. (2) genuinely defines an
        exact hit as a separate case; NEC has no such case — its Q is always
        the kernel sum.  More importantly, ``NECAlgorithm._gradient_step``
        cannot reproduce a shortcut (returning a stored constant kills the
        gradient), so any shortcut here would make the network act under a
        different Q-function than the one it is regressed onto.  Measured
        divergence when the shortcut was present: ~0.3% on exact hits.

        The kernel needs no shortcut anyway: an exact re-encounter has
        distance 0 and therefore weight 1/δ = 1000, which already dominates
        every genuine neighbour (~1–2 on the unit sphere).

        ``_key_to_slot`` remains a write-path structure, used by
        :meth:`write_batch` for the blend rule.

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
        # Nothing stored, or every action still at/below the kNN threshold:
        # the answer is +inf everywhere, so skip the kNN sweep entirely.
        # (QEC gets this early-out per action via `continue`; the batched
        # kNN here can only take it for the whole table at once.)
        if self.keys is None or max_size <= self.k:
            # Counted, not skipped: "every query was a sentinel" is exactly
            # what optimistic_rate exists to report.
            if self._record_lookups:
                self._lookup_queries    += B * A
                self._lookup_optimistic += B * A
            return torch.full((B, A), float("inf"), dtype=torch.float32, device=dev_q)

        if queries.device != self.device:
            queries = queries.to(self.device)

        dists, idx = self.knn_all_actions(queries, self.k)        # (A, B, k_use)

        a_idx    = torch.arange(A, device=self.device).view(A, 1, 1).expand_as(idx)
        knn_vals = self.values[a_idx, idx].float()                # (A, B, k_use)

        weights = 1.0 / (dists ** 2 + self.kernel_delta)          # (A, B, k_use)
        knn_q   = (weights * knn_vals).sum(-1) / weights.sum(-1)  # (A, B)

        # Sparse actions: their padded +inf distances make knn_q meaningless
        # (possibly NaN when every slot is padding), so overwrite wholesale.
        sizes_t     = torch.tensor(self._sizes, device=self.device)
        sparse_mask = (sizes_t <= self.k).view(A, 1)
        result = torch.where(
            sparse_mask, torch.full_like(knn_q, float("inf")), knn_q
        )

        if self._record_lookups:
            # `dists` comes from a `largest=False` top-k, so column 0 is the
            # nearest neighbour and carries the largest kernel weight.  Both
            # sums stay on device (see reset_lookup_stats).
            graded = (~sparse_mask).expand(A, B)
            n_graded = int(graded.sum())
            self._lookup_queries    += A * B
            self._lookup_optimistic += A * B - n_graded
            if n_graded:
                share = weights[..., 0] / weights.sum(-1)          # (A, B)
                self._lookup_nn_dist    += dists[..., 0][graded].double().sum()
                self._lookup_top_weight += share[graded].double().sum()

        return result.T.to(dev_q)  # (B, A)

    def _pairwise(self, q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Batched (A, b, n) L2 distances between ``q`` (A, b, d) and ``keys``.

        Same unit-norm identity as :func:`_topk_l2_unit`, but this path keeps
        the full distance matrix because ``knn_all_actions`` has to mask dead
        slots to +inf before the top-k.  ``baddbmm`` computes ``2 - 2 q·k`` in
        one fused call.
        """
        if not self.unit_norm_keys:
            return torch.cdist(q, keys)
        sim = torch.bmm(q, keys.transpose(1, 2))
        return (2.0 - 2.0 * sim).clamp_min_(0.0).sqrt_()

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

        Chunking
        --------
        Two levels, **queries outermost**.  The outer loop splits the query
        set so one iteration's distance matrix fits ``_CHUNK_BYTES``; the
        inner loop splits the key table only when even a single query chunk
        still cannot fit.  The ordering is the point: with queries outermost
        the inner loop runs exactly once for any realistic
        (num_actions, capacity), which skips the cat/topk/gather merge
        entirely.

        The previous capacity-outermost version ran that merge once per
        capacity chunk, and its byte budget counted only the ``cdist``
        output.  The merge materialises ``(A, B, k_eff + chunk)`` in float32
        *and* int64, so the true peak was 3-4x the budget: for the
        episode-end bootstrap on H.E.R.O. (A=18, B≈4400, capacity=5e5) that
        was 591 iterations of ~2.0 GB each, on top of the 2.3 GB key table
        already resident on the same device.

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

        if k_eff <= 0 or B == 0:
            return (
                torch.full((A, B, 0), float("inf"), device=self.device),
                torch.zeros((A, B, 0), dtype=torch.long, device=self.device),
            )

        out_d = torch.empty((A, B, k_eff), device=self.device)
        out_i = torch.empty((A, B, k_eff), dtype=torch.long, device=self.device)

        # (A, 1, max_size): which slots hold a live entry, broadcast over queries.
        slot_idx = torch.arange(max_size, device=self.device).view(1, 1, -1)
        valid    = slot_idx < sizes_t.view(A, 1, 1)

        q_chunk = max(1, self._CHUNK_BYTES // (A * max_size * 4))
        q_chunk = min(q_chunk, B)

        for qs in range(0, B, q_chunk):
            qe    = min(qs + q_chunk, B)
            b     = qe - qs
            q_exp = queries[qs:qe].unsqueeze(0).expand(A, b, d)

            k_chunk = max(1, self._CHUNK_BYTES // (A * b * 4))
            k_chunk = min(k_chunk, max_size)

            if k_chunk >= max_size:
                # Common path: whole table in one shot, no merge.
                cd = self._pairwise(q_exp, self.keys[:, :max_size, :])
                cd = cd.masked_fill(~valid, float("inf"))
                bd, bi = cd.topk(k_eff, dim=-1, largest=False)
            else:
                bd = torch.full((A, b, k_eff), float("inf"), device=self.device)
                bi = torch.zeros((A, b, k_eff), dtype=torch.long, device=self.device)
                for cs in range(0, max_size, k_chunk):
                    ce = min(cs + k_chunk, max_size)
                    cd = self._pairwise(q_exp, self.keys[:, cs:ce, :])
                    cd = cd.masked_fill(~valid[..., cs:ce], float("inf"))

                    ck = min(k_eff, ce - cs)
                    chd, chi = cd.topk(ck, dim=-1, largest=False)
                    chi = chi + cs

                    merged_d = torch.cat([bd, chd], dim=-1)
                    merged_i = torch.cat([bi, chi], dim=-1)
                    bd, keep = merged_d.topk(k_eff, dim=-1, largest=False)
                    bi = merged_i.gather(-1, keep)

            out_d[:, qs:qe] = bd
            out_i[:, qs:qe] = bi

        return out_d, out_i

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

        # An empty table (or query set) has no neighbours to return.  Without
        # this, `chunk_size` collapses to 0 below and the loop raises
        # `ValueError: range() arg 3 must not be zero`.  Every current caller
        # gates on `_sizes[a] > k` first, but this is a public method.
        if k_eff <= 0 or m == 0:
            return (
                torch.full((m, 0), float("inf"), device=self.device),
                torch.zeros((m, 0), dtype=torch.long, device=self.device),
            )

        # Bytes per stored slot: the (m, chunk) cdist output plus the
        # cat/topk/gather merge, which holds (m, k_eff + chunk) in float32 and
        # int64.  (QEC's formula divides by the embedding dim instead, which is
        # unrelated to the allocation size — conservative in practice, but wrong.)
        chunk_size = max(1, self._CHUNK_BYTES // (m * (4 + 4 + 8)))
        chunk_size = min(chunk_size, size)

        best_dists = torch.full((m, k_eff), float("inf"), device=self.device)
        best_idx   = torch.zeros((m, k_eff), dtype=torch.long,  device=self.device)

        for cs in range(0, size, chunk_size):
            ce  = min(cs + chunk_size, size)
            ck  = min(k_eff, ce - cs)
            block = self.keys[action, cs:ce]
            if self.unit_norm_keys:
                chd, chi = _topk_l2_unit(queries, block, ck)
            else:
                chd, chi = torch.cdist(queries, block).topk(
                    ck, dim=1, largest=False
                )
            chi = chi + cs

            merged_d = torch.cat([best_dists, chd], dim=1)
            merged_i = torch.cat([best_idx,   chi], dim=1)
            best_dists, keep = merged_d.topk(k_eff, dim=1, largest=False)
            best_idx   = merged_i.gather(1, keep)

        return best_dists, best_idx

    # ------------------------------------------------------------------
    # Gradient update of stored keys/values (paper Fig. 2, §3.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_gradient(
        self,
        action:     int,
        indices:    torch.Tensor,          # (m, k) slots that were read
        key_grad:   torch.Tensor | None,   # (m, k, d) dL/d(neighbour keys)
        value_grad: torch.Tensor | None,   # (m, k)    dL/d(neighbour values)
        key_lr:     float,
        value_lr:   float,
    ) -> None:
        """Stateless SGD step on the slots a minibatch actually retrieved.

        Paper Figure 2: "Gradients flow through the entire architecture."  The
        regression loss therefore updates the stored keys and values, not just
        the CNN.  ``NECAlgorithm._gradient_step`` gathers the retrieved
        neighbours into small leaf tensors and hands their ``.grad`` here, so
        this is a sparse scatter rather than a dense pass over the table.

        Two invariants have to be restored afterwards:

        1. **Unit norm.**  Every key is written L2-normalised (the kernel
           collapses otherwise — see ``tests/test_nec_kernel_scale.py``).  A
           raw gradient step moves keys off the unit sphere, so touched rows
           are re-projected onto it.
        2. **Hash validity.**  ``_key_to_slot`` maps a *quantised copy of the
           stored key* to its slot.  Once a key moves, that entry is stale and
           a later ``write_batch`` could blend a value into a slot whose key is
           no longer the one that hashed there.  Moved slots are recorded and
           delisted by :meth:`flush_moved_slots`.  Re-hashing instead would
           cost a GPU->CPU sync per touched slot per update.

        ``indices`` may repeat a slot (several queries can share a neighbour);
        ``index_put_(..., accumulate=True)`` sums those contributions, which is
        what the maths requires.
        """
        # A zero learning rate must be a true no-op, not "add zero then
        # re-normalise": re-projecting an already-unit-norm key still perturbs
        # it at float32 rounding level, and that noise would accumulate over
        # num_updates x batches. Setting both rates to 0 has to reproduce the
        # frozen-DND behaviour bit-for-bit so the change can be A/B'd.
        do_keys = key_grad is not None and key_lr != 0.0
        do_vals = value_grad is not None and value_lr != 0.0
        if not do_keys and not do_vals:
            return

        flat_idx = indices.reshape(-1)

        if do_keys:
            d = self.keys.shape[-1]
            self.keys[action].index_put_(
                (flat_idx,), -key_lr * key_grad.reshape(-1, d), accumulate=True
            )
        if do_vals:
            self.values[action].index_put_(
                (flat_idx,), -value_lr * value_grad.reshape(-1), accumulate=True
            )

        if do_keys:
            touched = torch.unique(flat_idx)
            # Invariant 1: back onto the unit sphere.
            self.keys[action, touched] = nn.functional.normalize(
                self.keys[action, touched], dim=-1
            )
            # Invariant 2: remember to delist (done in bulk, see flush_moved_slots).
            self._moved_slots[action].append(touched)

    def flush_moved_slots(self) -> int:
        """Delist every slot whose key was moved by :meth:`apply_gradient`.

        Called once per collector batch rather than per gradient step: the
        Python dict work is proportional to the number of DISTINCT slots
        touched, so batching collapses the 400 per-step passes into one.

        Returns the number of slots delisted (exposed as ``train/dnd_delisted``).
        """
        total = 0
        for a in range(self.num_actions):
            chunks = self._moved_slots[a]
            if not chunks:
                continue
            slots = torch.unique(torch.cat(chunks)).cpu().tolist()
            self._moved_slots[a] = []

            k_to_s = self._key_to_slot[a]
            s_to_k = self._slot_to_key[a]
            for slot in slots:
                old_key = s_to_k.pop(slot, None)
                if old_key is not None:
                    k_to_s.pop(old_key, None)
                    total += 1
        return total

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

        Rows are processed with sequential semantics, so duplicate keys
        within one call behave like repeated writes in the paper: the first
        occurrence of a novel key inserts; every later occurrence of the
        same key — whether it is already stored or still pending in this
        batch — blends with α.  (Without this, two identical embeddings in
        one episode would consume two ring-buffer slots and register
        conflicting entries in the exact-match dicts; a later ring-buffer
        eviction of either slot would then delist the survivor's key,
        breaking the slot↔key invariant.)

        Uses ``.data`` for all in-place tensor writes so the autograd graph
        for the gradient-step path remains intact.

        Returns
        -------
        (n_hits, n_novel) — blended occurrences vs. distinct new entries
        """
        n = len(states)
        if n == 0:
            return 0, 0

        if states.device != self.device:
            states = states.to(self.device)
        values = values.to(self.device)

        self._init_keys(states.shape[1])

        keys_b = self._make_keys(states)
        k_to_s = self._key_to_slot[action]

        vals   = values.tolist()
        n_hits = 0
        # Sequential semantics: first occurrence of a novel key inserts;
        # every later occurrence of the same key (stored or pending) blends.
        slot_updates: dict[int, float] = {}   # existing slot -> blended value
        pending_pos:  dict[bytes, int] = {}   # pending novel key -> novel_rows index
        novel_rows:   list[int]   = []
        novel_vals:   list[float] = []

        for i, key in enumerate(keys_b):
            slot = k_to_s.get(key)
            if slot is not None:
                old = slot_updates.get(slot, float(self.values.data[action, slot]))
                slot_updates[slot] = old + dnd_lr * (vals[i] - old)
                n_hits += 1
            elif key in pending_pos:
                j = pending_pos[key]
                novel_vals[j] = novel_vals[j] + dnd_lr * (vals[i] - novel_vals[j])
                n_hits += 1
            else:
                pending_pos[key] = len(novel_rows)
                novel_rows.append(i)
                novel_vals.append(vals[i])

        # --- Blend existing entries (paper §2.3 tabular Q-learning update) --
        if slot_updates:
            slots_t = torch.tensor(list(slot_updates.keys()),
                                   dtype=torch.long, device=self.device)
            vals_t  = torch.tensor(list(slot_updates.values()),
                                   dtype=self.values.dtype, device=self.device)
            self.values.data[action, slots_t] = vals_t

        # --- Insert novel entries via ring buffer ---------------------------
        if novel_rows:
            nov_idx_t = torch.tensor(novel_rows, dtype=torch.long, device=self.device)
            nov_val_t = torch.tensor(novel_vals, dtype=torch.float32, device=self.device)
            self._insert_novel(action, states[nov_idx_t], nov_val_t)

        return n_hits, len(novel_rows)

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
        # __getstate__ rotates full ring buffers so that slot 0 holds the
        # OLDEST entry; after restore the write pointer must therefore point
        # at slot 0 (full buffer) or at the append position == size (buffer
        # never wrapped; ptr == size is an invariant of _insert_novel).
        # Restoring the raw saved pointer would evict entries in a rotated
        # (wrong) order for a full capacity cycle.
        self._write_ptrs  = [
            0 if sz == self.capacity else sz for sz in self._sizes
        ]

        dev = self.device
        self.values = torch.zeros(self.num_actions, self.capacity, device=dev)

        self._key_to_slot = [{} for _ in range(self.num_actions)]
        self._slot_to_key = [{} for _ in range(self.num_actions)]
        # Rebuilt empty: the dicts below are regenerated from the restored
        # keys, so nothing is pending delisting at load time.
        self._moved_slots = [[] for _ in range(self.num_actions)]
        # __setstate__ bypasses __init__, so the read-path counters (and the
        # `_record_lookups` flag estimate_all branches on) have to be created
        # here too or the first lookup after a resume raises AttributeError.
        # They are per-rollout diagnostics and deliberately not checkpointed.
        self.reset_lookup_stats()

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
    final_state: torch.Tensor | None = None,  # (1, d) embedding of s_T when the
                                              # episode was TRUNCATED (not terminal)
) -> np.ndarray:
    """N-step discounted returns for a complete episode.

    For step t with t + n_step < T:
        Q^(N)_t = MC_t + γ^N (max_a Q(s_{t+N}) − MC_{t+N})

    For steps T − n_step ≤ t < T (near end of episode) or when the DND is
    too sparse (Q = +inf): fall back to the full Monte Carlo return MC_t.

    If ``final_state`` is given (the episode ended by TRUNCATION — e.g. a
    StepCounter cutoff — rather than a true terminal), every step whose
    N-step window crosses the cutoff additionally bootstraps
    γ^(T−t) max_a Q(s_T, a) from the DND instead of assuming zero future
    reward.  True terminals must pass ``final_state=None`` so no value is
    bootstrapped past a real done.

    This reuses lfilter from MFEC so the computation pattern is the same.
    The DND query uses ``@torch.no_grad()`` — gradients are computed later
    from the replay buffer.
    """
    T  = len(rewards)
    mc = lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()

    # Truncation bootstrap for the last min(T, n_step) steps.  Applied to the
    # OUTPUT only — `mc` must stay a pure reward sum for the correction
    # identity below to hold.
    tail_boot = np.zeros(T, dtype=np.float64)
    if final_state is not None:
        with torch.no_grad():
            # float(): _max_finite_q returns an array, and final_state is a
            # single (1, d) row.  Without the cast this stays shape (1,) and
            # silently broadcasts — harmless today, wrong the moment anyone
            # passes more than one final state.
            q_T = float(_max_finite_q(dnd.estimate_all(final_state.to(dnd.device)))[0][0])
        if np.isfinite(q_T):
            t_idx = np.arange(max(0, T - n_step), T)
            tail_boot[t_idx] = (gamma ** (T - t_idx)) * q_T

    if T <= n_step:
        return mc + tail_boot

    boot_states = states[n_step:].to(dnd.device)      # (T-n_step, d)
    with torch.no_grad():
        q_all        = dnd.estimate_all(boot_states)   # (T-n_step, A) float32
        q_max, valid = _max_finite_q(q_all)

    gamma_n = float(gamma ** n_step)
    n_step_G = mc.copy()
    correction = gamma_n * (q_max - mc[n_step:])
    n_step_G[:T - n_step] = np.where(
        valid, mc[:T - n_step] + correction, mc[:T - n_step]
    )
    return n_step_G + tail_boot


def _max_finite_q(q_all: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``max_a Q(s, a)`` over the actions whose DND table is populated.

    ``DND.estimate_all`` returns +inf for actions holding ``<= k`` entries
    (optimistic initialisation).  A plain ``.max(dim=-1)`` therefore lets a
    SINGLE under-populated action poison ``max_a Q`` for *every* state in the
    episode, silently dropping the whole episode back to plain Monte Carlo —
    with no metric revealing it.  On H.E.R.O. (18 actions, k=50) one rarely
    chosen action can sit below 51 entries for millions of frames once epsilon
    has annealed, which would disable N-step bootstrapping for the entire run.

    Masking the sentinels lets the bootstrap use the actions that do have
    enough data; only a state with no usable action at all falls back to MC.

    Returns
    -------
    (q_max, valid) — float64 ``(B,)`` maxima (0.0 where unusable) and a bool
    ``(B,)`` mask of rows with at least one populated action.
    """
    finite = torch.isfinite(q_all)
    q_max  = (q_all.masked_fill(~finite, float("-inf"))
              .max(dim=-1).values.cpu().numpy().astype(np.float64))
    valid  = finite.any(dim=-1).cpu().numpy()
    # -inf would propagate through the (discarded) invalid branch of np.where.
    return np.where(valid, q_max, 0.0), valid


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
      ``embedding_network(obs_shape, embedding_dim)`` in ``setup()``.  Its
      full contract lives in ``src.networks.NECEmbeddingNetwork``; the
      choice is a Hydra config group
      (``configs/algorithm/embedding_network/``), so it is swapped with
      ``algorithm/embedding_network=<name>`` rather than by editing nested
      YAML.  Unlike MFEC's frozen ``src.encoders.Encoder``, this network is
      trained end-to-end — every parameter it returns goes into Adam below.
    * ``replay_buffer`` is a no-arg factory returning a ``ReplayBuffer``.
    * The optimizer covers the CNN embedding network only.  DND values are
      updated by the in-place blend rule, not by gradient descent.
    * Per-env carry-over (the ``_carry`` list) started as
      ``MFECAlgorithm.step()``'s, but NEC's is a **bounded sliding window**:
      a step's N-step return needs only ``r_t..r_{t+n-1}`` and the bootstrap
      state ``s_{t+n}``, not the episode end, so only the last ``n_step`` raw
      frames are retained.  See ``step()``.
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
        # δ in the inverse-distance kernel.  The paper says 1e-3, but δ is
        # dimensionally a SQUARED DISTANCE, so it is only meaningful relative
        # to the embedding scale — and this implementation L2-normalises,
        # which the paper and the reference implementation
        # (github.com/EndingCredits/Neural-Episodic-Control) do not.
        #
        # Measured on 6000 real Ms. Pac-Man frames, mean squared distance to
        # the k=50 retrieved neighbours, and δ as a fraction of it:
        #
        #   reference: no normalisation, trunc_normal(0, 0.1) init
        #                                      d² = 1.7e+1   δ/d² = 0.00003
        #   ours:      L2-normalised, default torch init
        #                                      d² = 5.8e-3   δ/d² = 0.17
        #
        # At δ/d² = 0.17 the paper's division-by-zero *guard* is instead the
        # dominant term: every one of the k weights is ~1/δ regardless of the
        # true distance, so Q(s,a) degenerates to the mean of action a's
        # table — a per-action constant — and argmax returns one fixed action
        # for every state.  An exact re-encounter, the entire point of
        # episodic control, collected only 5.6% of the kernel mass against a
        # uniform floor of 1/k = 2.0%; at 1e-5 it collects 45%.
        #
        # 1e-5 is the smallest safe value: `_topk_l2_unit`'s `2 - 2·sim` fast
        # path has a measured float32 error of 4.5e-7 in d², so δ must stay
        # well above that or near-exact matches become numerical noise.  1e-5
        # keeps a 22x margin.
        kernel_delta:  float = 1e-5,
        dnd_lr:        float = 0.1,     # α for blending existing DND entries
        # Gradient (not blend) learning rates for the stored keys/values.
        # Paper Fig. 2 — "gradients flow through the entire architecture" — so
        # the regression loss updates the DND as well as the CNN.  §3.4 puts
        # the value rate BELOW the fast-update rate α; neither number is
        # published (§4 sweeps the SGD learning rate without reporting it).
        dnd_key_lr:    float = 1e-4,
        dnd_value_lr:  float = 1e-5,
        # --- N-step return -------------------------------------------------
        n_step: int = 100,
        # --- Optimisation --------------------------------------------------
        # RMSProp settings.  §4 says only "we used the RMSProp algorithm";
        # every number below is from the reference implementation
        # (github.com/EndingCredits/Neural-Episodic-Control, NECAgent.py),
        # which uses `RMSPropOptimizer(1e-5, decay=0.9, epsilon=0.01)` — the
        # DeepMind trio.  This used to be `torch.optim.RMSprop(params, lr=1e-4)`,
        # i.e. lr 10x higher on top of PyTorch's defaults alpha=0.99 and
        # eps=1e-8 — a stabiliser 1e6x smaller than the reference's.
        #
        # That combination is not a cosmetic difference.  RMSProp's step is
        # lr·g/(sqrt(v) + eps); with eps=1e-8 it degenerates towards
        # lr·sign(g) no matter how small the gradient is, so the CNN moves at
        # a near-constant rate every one of the `num_updates` steps per batch.
        # A DND key is written once and then read for thousands of batches, so
        # what matters is how far the embedding drifts per batch relative to
        # how far apart distinct states are.  Measured over one 400-update
        # batch on real Ms. Pac-Man frames (drift / state-spread):
        #
        #   lr=1e-4, alpha=0.99, eps=1e-8  (was)   8.7x
        #   lr=1e-5, alpha=0.9,  eps=0.01  (ref)   3.1x
        #
        # At 8.7x every stored key is stale before the next batch even reads
        # it, which is directly observable: the nearest stored key sat FURTHER
        # from a query (0.13–0.18) than an unrelated current frame did (0.094).
        lr:            float = 1e-5,
        rmsprop_alpha: float = 0.9,
        rmsprop_eps:   float = 0.01,
        gamma:         float = 0.99,
        batch_size:    int   = 32,
        max_grad_norm: float = 10.0,
        # --- Exploration ---------------------------------------------------
        eps_start:        float = 1.0,
        eps_end:          float = 0.01,
        annealing_frames: int   = 4_000_000,
        # Exploration rate used by get_policy() (evaluation).  NOT the
        # annealed training floor — this has to be big enough to actually
        # decorrelate episodes.  With repeat_action_probability=0.0 the ALE is
        # deterministic and NoopResetEnv does not perturb Ms. Pac-Man's
        # opening, so an episode of length L is bit-identical to pure argmax
        # with probability (1 - eval_eps)^L:
        #
        #     eval_eps   L=600   random actions/episode
        #     0.001      54.9%     0.6      <- was this; eval/return_min ~= max
        #     0.005       4.9%     3.0
        #     0.05        0.0%    30.0      <- Mnih et al. 2015 eval protocol
        #
        # At 0.001 more than half of all eval episodes replay the SAME
        # trajectory, so num_eval_episodes=5 costs 5x for ~1 effective sample
        # and the reported score is one fragile deterministic rollout.  0.05 is
        # the standard Atari evaluation rate (Mnih et al. 2015), whose
        # preprocessing Pritzel et al. §4 explicitly adopt; the paper itself
        # never states an evaluation epsilon.  0.0 restores pure argmax.
        eval_eps:         float = 0.05,
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
        self.dnd_key_lr      = dnd_key_lr
        self.dnd_value_lr    = dnd_value_lr
        self.n_step          = n_step
        self.lr              = lr
        self.rmsprop_alpha   = rmsprop_alpha
        self.rmsprop_eps     = rmsprop_eps
        self.gamma           = gamma
        self.batch_size      = batch_size
        self.max_grad_norm   = max_grad_norm
        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.eval_eps        = eval_eps
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

        # Both e-greedy modules need an *unbatched* action spec.  proof_env is
        # the TRAINING env, so with num_envs=8 its action_spec carries that
        # batch dim (shape [8]) — but BaseTrainer.evaluate() builds a single
        # env via make_env(num_envs=1), which factory.make_env returns bare
        # (no ParallelEnv), so the eval action is a scalar of shape [].
        # EGreedyModule.forward only auto-expands a spec when it is unbatched
        # (`not len(spec.shape)`); an [8] spec meeting a [] action raises
        # "Action spec shape does not match the action shape" at the first
        # evaluate().  An unbatched spec is correct for both callers: it
        # expands to [8] under the collector and compares equal at eval.
        #
        # Identical to the fix MFEC needed (commit a33fee3). NEC hit it only
        # once get_policy() gained an e-greedy tail — a bare QValueActor never
        # checks the spec, so the latent mismatch was invisible.
        action_spec_unbatched = action_spec
        for _ in range(len(env_bs)):
            action_spec_unbatched = action_spec_unbatched[0]

        self.greedy_module = EGreedyModule(
            spec=action_spec_unbatched,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=buf_dev,
        )
        self._explore_policy = _SharedPolicy(self.q_actor, self.greedy_module)

        # Evaluation policy: constant eps, NOT annealed and never `.step()`ed.
        # EvalEGreedyModule (not a stock EGreedyModule) because evaluate()
        # runs under ExplorationType.MODE, which gates a stock one off
        # entirely — it would look wired up and silently do nothing.
        self.eval_greedy_module = EvalEGreedyModule(
            spec=action_spec_unbatched,
            eps_init=self.eval_eps,
            eps_end=self.eval_eps,
            annealing_num_steps=1,
            device=buf_dev,
        )
        self._policy = _SharedPolicy(self.q_actor, self.eval_greedy_module)

        # `eval_eps` and `eps_end` are independent knobs that nothing couples,
        # and they live in different config files (nec_atari.yaml vs the
        # per-experiment override), so they can silently drift apart across a
        # pair of commits.  When they do, the run trains one policy and scores
        # a different one: eval/return_mean comes in well below
        # train/episode_reward at the SAME episode length, and eval/return_min
        # sticks at random-play level, while every learning curve looks
        # healthy.  Nothing else in the logs makes that recoverable, hence a
        # warning at setup and `eval/epsilon` in eval_metrics().
        #
        # An order of magnitude is the threshold because the intended gap is
        # small: eval ε is meant to sit at or slightly above the training
        # floor purely to decorrelate episodes on a deterministic ALE.
        if self.eval_eps > 10 * self.eps_end or self.eps_end > 10 * self.eval_eps:
            warnings.warn(
                f"NEC exploration mismatch: training anneals to eps_end="
                f"{self.eps_end:g} but evaluation runs at eval_eps="
                f"{self.eval_eps:g}. eval/* metrics then describe a different "
                f"policy than train/* and are not comparable to each other or "
                f"to published scores.",
                UserWarning,
                stacklevel=2,
            )

        # 4. Replay buffer — stores (obs, action, n_step_return)
        self.replay_buffer = self._make_replay_buffer()

        # 5. Optimizer — CNN embedding network only.
        # RMSProp per paper §4: "We used the RMSProp algorithm for gradient
        # descent training."  (This was Adam until the paper was re-checked.)
        # The DND's own keys/values are NOT in here: they are updated by a
        # stateless sparse SGD step in _gradient_step, because a stateful
        # optimiser's per-slot moments go stale when the ring buffer overwrites
        # a slot.  See DND.apply_gradient.
        self.optimizer = torch.optim.RMSprop(
            self.embedding_net.parameters(),
            lr=self.lr,
            alpha=self.rmsprop_alpha,
            eps=self.rmsprop_eps,
        )

        # 6. Per-env carry buffers for partial episodes across batch boundaries
        self._num_envs: int = int(env_bs[0]) if len(env_bs) == 1 else 1
        self._carry: list[dict | None] = [None] * self._num_envs

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    #: Frames per forward pass in :meth:`_embed`.  Bounds peak activation
    #: memory independently of how long the episode being embedded is.
    _EMBED_CHUNK = 256

    @torch.no_grad()
    def _embed(self, obs: torch.Tensor, out_device: torch.device) -> torch.Tensor:
        """Embed ``(N, *obs_shape)`` observations, L2-normalised, in chunks.

        Every DND read and write goes through here, so the unit-norm
        projection lives in exactly one place (``tests/test_nec_kernel_scale.py``
        explains why it must never be skipped: without it the inverse-distance
        kernel is dominated by ``kernel_delta`` and collapses).

        Chunked because a completed episode arrives in a single call and can
        be as long as ``StepCounter``'s ``max_steps`` — 4500 on the Atari
        configs.  One un-chunked pass over 4500 x 4 x 84 x 84 float32 frames
        costs ~0.5 GB of input plus ~0.5 GB of conv activations, on top of a
        DND key table that is already 1-2 GB on the same device.
        """
        n = obs.shape[0]
        if n == 0:
            return torch.empty(0, self.embedding_dim, device=out_device)

        parts: list[torch.Tensor] = []
        for i in range(0, n, self._EMBED_CHUNK):
            block = obs[i: i + self._EMBED_CHUNK].to(self.device).float()
            h = self.embedding_net(block)
            parts.append(nn.functional.normalize(h, dim=-1).to(out_device))
        return torch.cat(parts, dim=0)

    def _flush_pending_to_dnd(
        self, pending: list[tuple]
    ) -> tuple[int, int]:
        """Write one finished episode's ``(h, action, return)`` triples.

        The sliding window in ``step()`` finalises a step's return as soon as
        it matures (n_step later), but the triples are held here until the
        episode actually ends, so DND writes stay episode-end as the paper
        specifies.  Grouped by action with a single host sync, mirroring
        ``_gradient_step``.

        Returns ``(n_hits, n_novel)`` summed over actions.
        """
        if not pending:
            return 0, 0

        dev  = self._buffer_device
        h    = torch.cat([p[0] for p in pending], dim=0)
        acts = torch.cat([p[1] for p in pending], dim=0)
        vals = torch.tensor(
            np.concatenate([p[2] for p in pending]),
            dtype=torch.float32, device=dev,
        )

        order   = torch.argsort(acts, stable=True)
        counts  = torch.bincount(acts, minlength=self._num_actions)
        offsets = torch.zeros(self._num_actions + 1, dtype=torch.long, device=dev)
        offsets[1:] = counts.cumsum(0)
        offsets_cpu = offsets.cpu().tolist()

        h_sorted = h[order]
        v_sorted = vals[order]

        total_hits = total_writes = 0
        for a in range(self._num_actions):
            seg_s, seg_e = offsets_cpu[a], offsets_cpu[a + 1]
            if seg_s == seg_e:
                continue
            n_hit, n_new = self.dnd.write_batch(
                a, h_sorted[seg_s:seg_e], v_sorted[seg_s:seg_e], self.dnd_lr
            )
            total_hits   += n_hit
            total_writes += n_new
        return total_hits, total_writes

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: anneal ε, compute N-step returns, update DND,
        fill replay buffer, and run gradient steps after warm-up.

        The (E, T) per-env structure from mfec.py is preserved exactly:
        - Returns are computed only over COMPLETE episodes.
        - Returns are finalised on a SLIDING WINDOW, not at episode end.
          Step t is finalised as soon as t + n_step is in hand, because
          Q^(N)_t needs only r_t..r_{t+n-1} and the bootstrap state s_{t+n}.
          Only the last n_step steps must wait for the episode to end.
          ``_carry`` therefore retains at most n_step raw frames per env
          (~11 MB) instead of a whole episode (3 GB at max_steps=27_000, i.e.
          24 GB across 8 envs in a 32 GB container).
        - The finalised ``(h, action, return)`` triples are buffered in
          ``_carry["pending"]`` at ~264 B/step and written to the DND at
          EPISODE END, as the paper specifies — the window changes when
          returns are *computed*, not when they are written.
        - Raw observations are stored (not embeddings) so the window can be
          re-embedded with the CURRENT network on each step() call.
        - Consequence: the bootstrap Q and the embedding for step t are taken
          ~n_step steps after t rather than at episode end, so both are
          fresher than before. Same formula, different (better-conditioned)
          inputs — returns are NOT bit-identical to the pre-window code.
          Guarded by ``tests/test_nec_sliding_window.py``.
        - N-step returns are computed per-episode via lfilter + DND bootstrap.
        - Episodes ended by TRUNCATION (``done`` without ``terminated``, e.g.
          a StepCounter cutoff) bootstrap their return tail from the DND at
          the post-cutoff state; true terminals keep the Monte Carlo tail.
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
        # Next-state observations — only needed at episode-end positions, to
        # bootstrap the return tail of TRUNCATED episodes.
        next_obs_2d = batch["next", self.obs_key].reshape(E, T, *obs_shape)

        rewards_2d = (batch["next", "reward"].cpu().numpy()
                      .flatten().astype(np.float64).reshape(E, T))
        dones_2d   = (batch["next", "done"].cpu().numpy()
                      .flatten().astype(bool).reshape(E, T))
        term_2d    = (batch["next", "terminated"].cpu().numpy()
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
            # Raw frames stay host-side: they are the bulk of the window and
            # only need to reach the accelerator one _EMBED_CHUNK at a time.
            obs_e     = obs_2d[env_idx].cpu()                   # (T, *obs_shape)
            rewards_e = rewards_2d[env_idx]                     # (T,) f64 numpy
            dones_e   = dones_2d[env_idx]                       # (T,) bool numpy
            actions_e = actions_2d[env_idx]                     # (T,) long

            carry = self._carry[env_idx]
            # The retained window never contains a done (every done is drained
            # below in the same call), so episode ends always lie in the
            # CURRENT batch at index (d - carry_len) — used to look up
            # terminated/next-obs for the truncation bootstrap.
            carry_len = 0 if carry is None else len(carry["dones"])
            pending: list[tuple] = [] if carry is None else carry["pending"]
            if carry is not None:
                obs_e     = torch.cat([carry["obs"], obs_e], dim=0)
                rewards_e = np.concatenate([carry["rewards"], rewards_e])
                dones_e   = np.concatenate([carry["dones"],   dones_e])
                actions_e = torch.cat([carry["actions"], actions_e], dim=0)

            # Embed the whole window once with the CURRENT network.
            h_e = self._embed(obs_e.reshape(len(obs_e), *obs_shape), dev)

            cursor = 0          # start of the not-yet-finalised region
            while True:
                rel = np.flatnonzero(dones_e[cursor:])

                if len(rel):
                    # --- Episode ends here: finalise its tail and flush ------
                    d       = cursor + int(rel[0])
                    raw_end = d - carry_len        # position in the current batch

                    h_final = None
                    if not term_2d[env_idx, raw_end]:
                        # Truncated (StepCounter / collector cutoff), not a real
                        # terminal: bootstrap the return tail from the state
                        # after the cutoff instead of assuming zero future
                        # reward.  True terminals keep h_final=None.
                        h_final = self._embed(
                            next_obs_2d[env_idx, raw_end].reshape(1, *obs_shape), dev
                        )

                    seg = slice(cursor, d + 1)
                    # Passing only the tail is exact: return-to-go looks
                    # FORWARD only, and every earlier step of this episode was
                    # already finalised with its own n-step bootstrap.
                    nsr = _compute_n_step_returns(
                        rewards_e[seg], h_e[seg], self.gamma, self.n_step,
                        self.dnd, final_state=h_final,
                    )
                    pending.append((h_e[seg], actions_e[seg], nsr))
                    collect_obs.append(obs_e[seg])
                    collect_actions.append(actions_e[seg].cpu())
                    collect_returns.append(nsr)

                    # DND writes happen at EPISODE END, as the paper specifies
                    # — the sliding window changes when returns are *computed*,
                    # not when they are written.
                    h_i, n_h = self._flush_pending_to_dnd(pending)
                    total_hits   += h_i
                    total_writes += n_h
                    pending = []

                    cursor = d + 1
                    continue

                # --- No episode end in the window: finalise matured steps ----
                # Step i is finalisable once i + n_step is in hand: its return
                # needs rewards r_i..r_{i+n-1} and the bootstrap state s_{i+n},
                # NOT the episode end.  Only the last n_step steps must wait.
                avail    = len(obs_e) - cursor
                n_mature = avail - self.n_step
                if n_mature > 0:
                    seg = slice(cursor, len(obs_e))
                    # _compute_n_step_returns bootstraps exactly states[n_step:],
                    # i.e. the n_mature entries we keep; the trailing MC-tail
                    # entries it also returns are for steps whose episode has
                    # not ended, so they are discarded.
                    nsr = _compute_n_step_returns(
                        rewards_e[seg], h_e[seg], self.gamma, self.n_step,
                        self.dnd, final_state=None,
                    )[:n_mature]

                    mat = slice(cursor, cursor + n_mature)
                    pending.append((h_e[mat], actions_e[mat], nsr))
                    collect_obs.append(obs_e[mat])
                    collect_actions.append(actions_e[mat].cpu())
                    collect_returns.append(nsr)
                    cursor += n_mature
                break

            # --- Retain only the immature tail --------------------------------
            # At most n_step raw frames survive to the next call (plus whatever
            # one batch adds), instead of a whole episode.  With
            # StepCounter.max_steps=27_000 that is the difference between
            # ~11 MB and ~3 GB per env.  `pending` holds the finalised
            # (h, action, return) triples of the in-flight episode at ~264 B
            # per step — 7 MB for a full 27_000-step episode.
            if cursor >= len(obs_e) and not pending:
                self._carry[env_idx] = None
            else:
                self._carry[env_idx] = {
                    "obs":     obs_e[cursor:],   # CPU — raw pixels, re-embedded each call
                    "rewards": rewards_e[cursor:],
                    "dones":   dones_e[cursor:],
                    "actions": actions_e[cursor:].to(dev),
                    "pending": pending,
                }

        # --- Store finalised transitions in the replay buffer ---------------
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
        # Only updates that actually ran are averaged.  `_gradient_step`
        # returns None when it skipped (replay buffer below batch_size, or
        # every sampled action's DND table still too sparse); folding those in
        # as 0.0 would report a fabricated `train/q_loss` of exactly 0.0,
        # which reads as "converged" rather than "never ran".  That is not
        # hypothetical: the replay buffer is NOT checkpointed (see
        # `_load_training_state`), so the first batches after every resume
        # skip every single update.
        losses: list[torch.Tensor] = []
        q_vals: list[torch.Tensor] = []
        for _ in range(self.num_updates):
            result = self._gradient_step()
            if result is None:
                continue
            loss_val, mean_q = result
            losses.append(loss_val)
            q_vals.append(mean_q)

        # Gradient steps moved stored keys, so their exact-match hashes are
        # stale.  Delist them once per batch rather than once per update — the
        # cost is proportional to DISTINCT slots touched, so batching collapses
        # num_updates passes into one.
        delisted = self.dnd.flush_moved_slots()

        metrics = {
            **base_metrics,
            "train/updates":      float(len(losses)),
            "train/dnd_delisted": float(delisted),
        }
        if losses:
            # Two host syncs per collector batch instead of two per update.
            metrics["train/q_loss"]   = float(torch.stack(losses).mean())
            metrics["train/q_values"] = float(torch.stack(q_vals).mean())
        return metrics

    def _gradient_step(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One minibatch gradient update on the embedding network.

        Samples (obs, action, n_step_return) from the replay buffer, re-embeds
        observations with the *current* embedding network, computes the
        kernel-weighted Q̂ via differentiable indexing into DND.values, and
        minimises MSE(Q̂, n_step_return_target).

        Gradients flow through the embedding network (CNN parameters) via the
        distance term ‖h − h_i‖² where h = embedding_net(obs).  The stored
        DND values Q_i are frozen constants here; they are updated separately
        by the in-place blend rule in step().

        Known weakness (inherent to the paper's design, not a bug here): every
        state sampled from the replay buffer was ALSO written into DND[a] at
        episode end with a value equal to its own target.  That entry is then
        the query's own nearest neighbour at distance ~0, i.e. weight
        1/δ = 1000 against ~1-2 for genuine neighbours, so Q̂ reproduces the
        target almost exactly and the loss is tiny (measured ~2e-06 on a
        synthetic fixture).  Real learning signal only appears once the CNN
        has drifted enough, or the entry has been evicted, to break the
        self-match.  Watch the magnitude of ``train/q_loss``: a curve pinned
        around 1e-6 means the network is not actually being pushed anywhere.

        Returns
        -------
        ``(loss, mean_q)`` as 0-dim tensors on ``self.device`` for an update
        that actually ran, or ``None`` when this step was skipped — replay
        buffer below ``batch_size``, or every sampled action's DND table still
        at/below ``k`` entries.  ``None`` rather than ``(0.0, 0.0)``: a skipped
        step is not a zero-loss step, and ``step()`` must not average it into
        ``train/q_loss``.  Tensors rather than floats so the caller can defer
        the host sync to once per collector batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

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
        # (action, slot indices, neighbour-key leaf, neighbour-value leaf) per
        # action, kept so the DND can be updated from their .grad after
        # backward().  See the scatter block below.
        dnd_pending: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # Group the minibatch by action with ONE host sync.
        #
        # This used to be `mask = (actions == a); n_a = int(mask.sum())` inside
        # the loop, i.e. one GPU->CPU synchronisation *per action per update* —
        # 3,600 per collector batch at num_actions=9, num_updates=400. Each one
        # drains the CUDA pipeline behind whatever is queued (a knn_action that
        # reads the whole 64 MB key table, plus the CNN backward), so the CPU
        # can never run ahead and the per-update cost collapses to pure
        # round-trip latency. Same argsort/bincount/offsets idiom step() uses
        # for write_batch.
        order   = torch.argsort(actions, stable=True)
        counts  = torch.bincount(actions, minlength=self._num_actions)
        offsets = torch.zeros(
            self._num_actions + 1, dtype=torch.long, device=actions.device
        )
        offsets[1:] = counts.cumsum(0)
        offsets_cpu = offsets.cpu().tolist()      # <- the only sync in this loop

        h_sorted = h[order]
        t_sorted = targets[order]

        for a in range(self._num_actions):
            seg_s, seg_e = offsets_cpu[a], offsets_cpu[a + 1]
            if seg_s == seg_e:
                continue
            if self.dnd._sizes[a] <= self.k:
                continue  # too sparse to compute a kernel-weighted Q

            h_a = h_sorted[seg_s:seg_e]   # (n_a, embedding_dim)
            t_a = t_sorted[seg_s:seg_e]

            k_use = min(self.k, self.dnd._sizes[a])

            # Find neighbour indices without tracking gradients
            with torch.no_grad():
                _, indices = self.dnd.knn_action(h_a.detach(), a, k_use)  # (n_a, k_use)

            # Gather the retrieved neighbours into SMALL leaf tensors.
            #
            # Paper Figure 2: "Gradients flow through the entire architecture"
            # — the loss updates the stored keys h_i and values Q_i as well as
            # the CNN.  Doing that by making self.dnd.keys/.values autograd
            # leaves would make every backward() allocate a DENSE gradient the
            # size of the whole table (num_actions x capacity x d = 1.15 GB at
            # the Atari defaults, per update, 400 updates per batch).  Gathering
            # first keeps the graph at (n_a, k_use, d) and lets the update be
            # scattered back into only the slots this minibatch actually read.
            neighbor_keys = self.dnd.keys[a, indices].detach().requires_grad_(True)
            neighbor_vals = self.dnd.values[a, indices].detach().requires_grad_(True)

            diffs    = h_a.unsqueeze(1) - neighbor_keys        # (n_a, k_use, d)
            dists_sq = (diffs ** 2).sum(dim=-1)                # (n_a, k_use)
            weights  = 1.0 / (dists_sq + self.kernel_delta)    # (n_a, k_use)
            weights  = weights / weights.sum(dim=-1, keepdim=True)

            q_hat_a  = (weights * neighbor_vals).sum(dim=-1)   # (n_a,)

            q_hat_parts.append(q_hat_a)
            target_parts.append(t_a)
            dnd_pending.append((a, indices, neighbor_keys, neighbor_vals))

        if not q_hat_parts:
            return None

        q_hat  = torch.cat(q_hat_parts)
        tgt    = torch.cat(target_parts)
        # SUM, not mean — the reference implementation uses
        # `tf.reduce_sum(tf.square(td_err))`.  This is not free to change
        # independently of the optimiser: for RMSProp, scaling the loss by a
        # constant c is equivalent to dividing `eps` by c, so the reduction and
        # (lr, alpha, eps) only transfer together.  Keeping `mean` while
        # adopting the reference's eps=0.01 would damp every step by up to
        # `batch_size`.  `train/q_loss` still reports the MEAN so the metric
        # stays comparable across batch sizes.
        sq_err = (q_hat - tgt) ** 2
        loss   = sq_err.sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.embedding_net.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # --- Apply the DND half of the gradient -----------------------------
        # Plain SGD, deliberately: a stateful optimiser (Adam/RMSProp) keeps
        # per-slot momentum, and the ring buffer overwrites slots underneath
        # it, so a freshly inserted entry would inherit the moments of whatever
        # it evicted.  That is the failure that made an earlier attempt drive
        # stored values negative.  Stateless SGD has nothing to go stale, and
        # it leaves untouched slots bit-identical instead of decaying all
        # num_actions x capacity of them on every step.
        for a, indices, nk, nv in dnd_pending:
            self.dnd.apply_gradient(
                a, indices,
                key_grad=nk.grad, value_grad=nv.grad,
                key_lr=self.dnd_key_lr, value_lr=self.dnd_value_lr,
            )

        # Returned as 0-dim TENSORS, not floats: `float(...)` here would force
        # two more host syncs per update (800 per collector batch). step()
        # stacks them and reduces once.
        return sq_err.detach().mean(), q_hat.detach().mean()

    def _sizes_summary(self) -> list[int]:
        return list(self.dnd._sizes)

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        # Ends in EvalEGreedyModule (constant eval_eps), NOT a bare argmax —
        # see that class for why a deterministic eval policy makes
        # eval/return_std identically 0.
        return self._policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Evaluation diagnostics
    # ------------------------------------------------------------------

    def reset_eval_metrics(self) -> None:
        self.dnd.reset_lookup_stats()
        self.dnd._record_lookups = True

    def eval_metrics(self) -> dict[str, float]:
        """What the DND actually did during the evaluation rollout.

        ``eval/return_mean`` on its own cannot distinguish the three ways NEC
        evaluates badly, and they need opposite fixes:

        1. **The memory holds a bad policy.** Returns are low, the kernel is
           peaked, the neighbours are close. Train longer / tune the DND.
        2. **The memory is not being used.** Judge this from
           ``dnd_top_weight`` only against the calibration below — **not**
           against ``1/k``. An earlier version of this docstring claimed that
           ``top_weight`` near ``1/k`` (0.02 at the paper's k=50) meant the
           kernel had degenerated to a flat mean; that is wrong, and acting on
           it costs a wrong diagnosis. Measured on 6000 real Ms. Pac-Man
           frames, predicting held-out discounted return-to-go:

               retriever                     Pearson r   top_weight
               NEC embedding (shipped)         +0.50       0.025
               raw pixels (k-NN reference)     +0.60       0.029

           A retriever that beats the predict-the-mean baseline by 15% still
           sits at 0.029, because with ``k`` = 50 in 64 dimensions the 1st and
           50th neighbour are always at nearly the same distance. Near-``1/k``
           is the *normal* reading. What ``top_weight`` genuinely detects is
           the exact-re-encounter case: a stored state queried back should
           dominate its neighbours, and if that never lifts the average then
           the DND is being written and read in different embedding spaces.
        3. **The evaluation policy is not the policy that was trained.** This
           is why ``eval/epsilon`` is reported here, right next to the
           returns. ``eval_eps`` and the annealed training floor ``eps_end``
           are two independent knobs that nothing couples, and a run whose
           eval ε is an order of magnitude above its training ε posts a
           depressed ``eval/return_mean`` and a ``eval/return_min`` pinned at
           random-play level while ``train/episode_reward`` climbs normally.
           There is no other logged quantity from which that is recoverable
           after the fact.

        ``eval/dnd_nn_dist`` supports (2): embeddings are unit-norm, so a
        nearest-neighbour distance that drifts upward over training means the
        stored keys are going stale relative to the CNN faster than
        ``dnd_key_lr`` refreshes the ones the kNN happens to touch.

        Returns ``{}`` when the rollout recorded no lookups, matching
        ``MFECAlgorithm.eval_metrics`` — an absent metric leaves a gap in the
        chart rather than a fabricated zero.
        """
        stats = self.dnd.lookup_stats()
        self.dnd._record_lookups = False
        if not stats:
            return {}

        metrics = {
            "eval/epsilon":             float(self.eval_greedy_module.eps),
            "eval/dnd_optimistic_rate": stats["optimistic_rate"],
        }
        if "nn_dist" in stats:
            metrics["eval/dnd_nn_dist"]    = stats["nn_dist"]
            metrics["eval/dnd_top_weight"] = stats["top_weight"]
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        """Snapshot everything that is not cheaply recomputable.

        NOT checkpointed, deliberately:

        * **The replay buffer.**  1e5 float32 pixel transitions is 11.3 GB;
          writing that on every checkpoint is not viable.  On resume the
          gradient path therefore no-ops until the buffer refills — `step()`
          reports `train/updates` so that gap is visible rather than
          masquerading as a zero loss.
        * **The per-env `_carry`.**  It holds RAW pixels for the in-flight
          episode (up to `StepCounter.max_steps` = 4500 frames = 508 MB per
          env; ~4 GB across 8 envs), and it is the one piece of state whose
          loss is nearly free: returns are computed backwards from the episode
          end, so a partial episode that starts mid-stream still yields
          correct return-to-go for every step it contains.  Dropping it costs
          at most `num_envs` partial episodes per process restart.
        """
        return TrainingState(
            step=0,
            policy_state_dict=self.embedding_net.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={
                "dnd_state":        self.dnd.__getstate__(),
                "collected_frames": self._collected_frames,
                # EGreedyModule keeps the live epsilon in its own buffer, which
                # is NOT part of embedding_net.state_dict().  Without this a
                # resumed run silently restarts exploration at eps_init and
                # re-anneals from scratch, even though _collected_frames says
                # the anneal finished millions of frames ago.
                "greedy_state":     self.greedy_module.state_dict(),
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.embedding_net.load_state_dict(state.policy_state_dict)

        # torch.load's map_location does not rewrite the pickled torch.device
        # inside dnd_state, so a checkpoint written on cuda:0 would rebuild the
        # DND on cuda:0 regardless of what this run resolved to (a hard failure
        # on a CPU-only host, a silent cross-device copy on a different GPU).
        # Re-pin to the device this algorithm actually owns.
        dnd_state = dict(state.extra["dnd_state"])
        dnd_state["device"] = self._buffer_device
        self.dnd.__setstate__(dnd_state)

        self.optimizer = torch.optim.RMSprop(
            self.embedding_net.parameters(),
            lr=self.lr,
            alpha=self.rmsprop_alpha,
            eps=self.rmsprop_eps,
        )
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self._collected_frames = int(state.extra["collected_frames"])

        greedy_state = state.extra.get("greedy_state")
        if greedy_state is not None:
            self.greedy_module.load_state_dict(greedy_state)

        # See _get_training_state: the carry is intentionally not persisted.
        self._carry = [None] * self._num_envs