"""Regression tests for MFEC exact-match dict correctness.

Two properties are verified:

1. **Embed determinism** — embed() and _make_keys() produce bit-identical
   output for bit-identical input on CPU.  If this were ever non-deterministic
   (e.g. due to GPU matmul ordering), the same game state seen twice would
   produce different keys and the exact-match dict would never hit.

2. **Exact-hit-rate non-zero after re-encounter** — feeding the same
   observation in two successive step() calls must register an exact hit on
   the second call (hit_rate > 0).  This is the key property required by
   Blundell et al. (2016) Eq. (2): re-encountered states must be resolved
   via O(1) dict lookup, not kNN.

Root-cause note: both tests would fail (or exact_hit_rate would stay 0) if
VecNorm is present in the environment transform stack, because VecNorm's
shifting running statistics change the normalization applied to each frame,
making the same raw pixel frame produce different float32 values at different
timesteps, and thus different hash keys.  MFEC env configs must NOT include
VecNorm (see configs/environment/qbert_train.yaml for the detailed comment).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from src.algorithms.mfec import MFECAlgorithm, QEC, QECPolicy
from src.encoders.random_projectins import RandomProjectionEncoder


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from test_mfec_returns.py)
# ---------------------------------------------------------------------------

class _MockGreedy:
    eps = 0.5
    def step(self, n: int) -> None:
        pass


def _make_algorithm(
    *,
    E: int = 1,
    num_actions: int = 3,
    state_dim: int = 4,
    obs_flat_dim: int = 4,
    buffer_size: int = 200,
    k: int = 1,
    gamma: float = 0.9,
) -> MFECAlgorithm:
    """Build a MFECAlgorithm with internals wired directly (no real TorchRL env)."""
    alg = MFECAlgorithm(
        device=None,
        obs_key="obs",
        buffer_size=buffer_size,
        k=k,
        state_dim=state_dim,
        gamma=gamma,
        frames_per_batch=10,
        key_scale=1e5,
    )
    dev = torch.device("cpu")
    alg._buffer_device    = dev
    alg._num_actions      = num_actions
    alg._num_envs         = E
    alg._carry            = [None] * E
    alg._collected_frames = 0

    # Identity projection: embedding = first state_dim pixels (exact, no noise).
    proj = np.zeros((obs_flat_dim, state_dim), dtype=np.float32)
    for i in range(min(obs_flat_dim, state_dim)):
        proj[i, i] = 1.0
    encoder = RandomProjectionEncoder(obs_flat_dim, state_dim, seed=0)
    encoder.projection = proj
    alg.encoder = encoder

    alg.qec          = QEC(num_actions, buffer_size, k, dev, key_scale=1e5)
    alg.qec_policy   = QECPolicy(alg.qec, alg.encoder, num_actions)
    alg.greedy_module = _MockGreedy()
    return alg


def _make_batch(
    obs:     torch.Tensor,
    rewards: torch.Tensor,
    dones:   torch.Tensor,
    actions: torch.Tensor,
) -> TensorDict:
    bs = obs.shape[:-1]
    return TensorDict(
        {
            "obs":    obs,
            "action": actions,
            "next": TensorDict(
                {
                    "reward": rewards.unsqueeze(-1).float(),
                    "done":   dones.unsqueeze(-1),
                },
                batch_size=list(bs),
            ),
        },
        batch_size=list(bs),
    )


# ---------------------------------------------------------------------------
# Test 1 — embed() and _make_keys() are deterministic for identical inputs
# ---------------------------------------------------------------------------

def test_embed_determinism():
    """embed() must produce bit-identical float32 output for identical input tensors.

    If this property ever broke (e.g. due to GPU reduction ordering), the same
    game state seen at two different timesteps would produce different keys and
    the exact-match dict would never fire.
    """
    np.random.seed(42)
    proj = np.random.randn(16, 4).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0)
    encoder = RandomProjectionEncoder(16, 4, seed=42)
    encoder.projection = proj

    qec    = QEC(3, 100, 1, torch.device("cpu"), key_scale=1e5)
    policy = QECPolicy(qec, encoder, 3)

    obs = torch.randn(8, 16)   # 8 arbitrary observations, each with 16 "pixels"

    state1 = policy.embed(obs)
    state2 = policy.embed(obs)

    assert torch.equal(state1, state2), (
        "embed() returned different float32 values for the same input tensor. "
        "Exact-match dict hits require bit-identical embeddings."
    )

    keys1 = qec._make_keys(state1)
    keys2 = qec._make_keys(state2)
    assert keys1 == keys2, (
        "_make_keys() produced different bytes for identical float32 states. "
        "This would prevent any dict hit even with a deterministic embed()."
    )


# ---------------------------------------------------------------------------
# Test 2 — same observation fed twice → exact hit on second step() call
# ---------------------------------------------------------------------------

def test_exact_match_hit_rate_nonzero():
    """The same observation in two consecutive step() calls must register as an
    exact dict hit on the second call (hit_rate > 0).

    This is the regression guard for the VecNorm bug: with VecNorm present in
    the env transforms, the same raw pixel frame produces different normalized
    values at different timesteps, making keys non-reproducible and
    exact_hit_rate permanently 0.  Without VecNorm, the fixed projection is
    deterministic and re-encountered states get O(1) dict resolution.
    """
    E, T = 1, 3

    # All pixels set to 0.5; with the identity projection, the embedded state
    # is [0.5, 0.5, 0.5, 0.5] for every timestep — a perfectly reproducible key.
    obs     = torch.full((E, T, 4), 0.5)
    rewards = torch.ones(E, T)
    dones   = torch.zeros(E, T, dtype=torch.bool)
    dones[0, T - 1] = True          # one complete episode per batch
    actions = torch.zeros(E, T, dtype=torch.long)

    alg   = _make_algorithm(E=E, state_dim=4, obs_flat_dim=4, k=1)
    batch = _make_batch(obs, rewards, dones, actions)

    # First step: QEC is empty — all states are novel, no hits.
    m1 = alg.step(batch)
    assert m1["train/exact_hit_rate"] == 0.0, (
        "First step must have 0 exact hits (QEC is empty)"
    )

    # Second step: same observation → same key → must hit the dict.
    m2 = alg.step(batch)
    assert m2["train/exact_hit_rate"] > 0.0, (
        f"Expected exact_hit_rate > 0 on second step with identical observation, "
        f"got {m2['train/exact_hit_rate']:.4f}. "
        "This indicates keys are not reproducible — likely VecNorm is present "
        "in the env transform stack or embed() is non-deterministic."
    )


# ---------------------------------------------------------------------------
# Test 3 — add_batch / dict consistency: insert then re-insert same key
# ---------------------------------------------------------------------------

def test_qec_add_batch_dict_consistency():
    """Inserting a state, then presenting the same state again in step(), must
    be recognised as an exact dict hit rather than treated as novel.
    """
    dev    = torch.device("cpu")
    qec    = QEC(num_actions=2, capacity=50, k=1, device=dev, key_scale=1e5)
    state  = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    value  = torch.tensor([5.0], dtype=torch.float64)

    # Insert the state for action 0.
    qec.add_batch(0, state, value)

    # The key must now be registered, pointing at a live slot.
    keys = qec._make_keys(state)
    assert keys[0] in qec._key_to_slot[0], (
        "Key not registered in _key_to_slot after add_batch"
    )
    slot = qec._key_to_slot[0][keys[0]]
    assert 0 <= slot < qec._sizes[0], (
        f"Slot {slot} is outside the live range [0, {qec._sizes[0]}) after add_batch"
    )
    assert qec.values[0, slot].item() == pytest.approx(5.0), (
        "Value tensor does not hold the inserted value at the registered slot"
    )

    # Present the same state again — must be an exact match (not novel).
    same_key = qec._make_keys(state)[0]
    assert same_key in qec._key_to_slot[0], (
        "Same state not found in dict on second lookup — dict is not consistent"
    )


# ---------------------------------------------------------------------------
# Test 4 — eviction is LRU (least recently *updated*), not FIFO
# ---------------------------------------------------------------------------

def test_qec_eviction_is_least_recently_updated():
    """Blundell et al. (2016) §2: eviction removes the least recently *updated*
    entry, not the least recently inserted one.

    A plain FIFO ring buffer would discard the oldest insertion, which on Atari
    is exactly the early-level state the agent re-visits on every episode.
    Here: fill the buffer, re-update the oldest entry, then overflow — the
    touched entry must survive and the next-oldest must be the one dropped.
    """
    dev = torch.device("cpu")
    qec = QEC(num_actions=1, capacity=3, k=1, device=dev, key_scale=1e5)

    states = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32
    )
    qec.add_batch(0, states, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert qec._sizes[0] == 3

    keys = qec._make_keys(states)

    # Touch the oldest entry (states[0]) — simulates an Eq. (1) max-update.
    qec.touch(0, [keys[0]])

    # Insert a fourth state: the buffer is full, so one entry must be evicted.
    new_state = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    qec.add_batch(0, new_state, torch.tensor([4.0], dtype=torch.float64))

    k_to_s = qec._key_to_slot[0]
    assert keys[0] in k_to_s, (
        "The touched (most recently updated) entry was evicted — eviction is "
        "still FIFO rather than LRU."
    )
    assert keys[1] not in k_to_s, (
        "Expected the least recently updated entry (states[1]) to be evicted, "
        f"but it is still present. Dict holds {len(k_to_s)} entries."
    )
    assert qec._make_keys(new_state)[0] in k_to_s, "New entry was not registered"
    assert len(k_to_s) == 3, f"Buffer overflowed capacity: {len(k_to_s)} entries"

    # The evicted entry's slot must have been reused, not leaked.
    assert qec._sizes[0] == 3


# ---------------------------------------------------------------------------
# Test 5 — keys are invariant to batch shape
# ---------------------------------------------------------------------------

def test_embed_key_invariant_to_batch_shape():
    """The same observation must hash to the same key whether it is embedded
    alone or as part of a larger batch.

    Training embeds ``num_envs`` rows at a time; ``BaseTrainer.evaluate`` builds
    a single env and therefore embeds 1 row.  A float32 matmul picks a different
    reduction order for those two shapes, and at key_scale=1e5 the resulting
    ~1e-6 error is the same order as the quantisation step — which would make
    evaluation silently miss every exact match.  RandomProjectionEncoder
    accumulates in float64 to keep the key shape-invariant.
    """
    obs_dim, state_dim = 7056, 64        # paper's single 84x84 frame
    encoder = RandomProjectionEncoder(obs_dim, state_dim, seed=0)
    qec     = QEC(1, 100, 1, torch.device("cpu"), key_scale=1e5)
    policy  = QECPolicy(qec, encoder, 1)

    torch.manual_seed(0)
    batch = torch.rand(16, obs_dim)      # pixel-like values in [0, 1]

    keys_batched = qec._make_keys(policy.embed(batch))
    keys_single  = [qec._make_keys(policy.embed(batch[i : i + 1]))[0] for i in range(16)]

    mismatches = [i for i in range(16) if keys_batched[i] != keys_single[i]]
    assert not mismatches, (
        f"{len(mismatches)}/16 observations hashed differently at batch size 1 "
        f"vs batch size 16 (rows {mismatches[:5]}). Exact-match lookups would "
        "fail during evaluation."
    )