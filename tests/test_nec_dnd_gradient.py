"""Gradients must reach the DND, not just the CNN (Pritzel et al. 2017, Fig. 2).

Figure 2's caption is explicit: "Gradients flow through the entire
architecture."  An earlier version of `nec.py` froze `DND.keys` and
`DND.values` and trained only the embedding network, which meant a stored key
was written once and never refreshed.  With `dnd_capacity=5e5` and ~178
inserts per action per collector batch, an entry survived ~2800 batches —
over a million gradient steps of CNN drift — while the kNN kept retrieving it
as though it still lived in the current embedding space.

`DND.apply_gradient` fixes that with a stateless sparse SGD step on the slots
a minibatch actually retrieved.  These tests pin the four properties that
make it correct, each of which is a bug if it regresses:

1. keys AND values move (the fix does what it claims);
2. slots the minibatch did not touch stay bit-identical (no dense pass over
   the table, and no optimiser state silently decaying 5e5 entries per step);
3. keys stay on the unit sphere (the kernel collapses otherwise —
   see test_nec_kernel_scale.py);
4. the exact-match hash never points at a key that has since moved.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from src.algorithms.nec import DND, NECAlgorithm
from src.networks import NatureEmbedding


OBS_SHAPE = (4, 84, 84)
EMBED_DIM = 64


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """``torch.equal`` that also holds for NaN.

    ``DND`` allocates its tables with ``torch.empty``, so every slot beyond
    ``_sizes[a]`` holds uninitialised memory — and whenever that garbage
    happens to be a NaN bit pattern, ``torch.equal`` returns False no matter
    what, because NaN != NaN. The "were these bytes left alone?" assertions
    below were therefore passing on luck: they flip to failing as soon as
    something earlier in the session changes what the allocator hands back
    (running the multiprocessing-based env tests first is enough to do it).

    Comparing the raw bits answers the question actually being asked.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    bits = torch.int64 if a.element_size() == 8 else torch.int32
    return torch.equal(a.view(bits), b.view(bits))


def _make_algorithm(*, key_lr: float, value_lr: float) -> NECAlgorithm:
    """NEC wired directly (no env), seeded with 3 distinct frames per action."""
    torch.manual_seed(0)
    np.random.seed(0)

    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=NatureEmbedding,
        obs_key="obs",
        embedding_dim=EMBED_DIM,
        dnd_capacity=200,
        k=2,
        kernel_delta=1e-3,
        dnd_lr=0.1,
        dnd_key_lr=key_lr,
        dnd_value_lr=value_lr,
        n_step=5,
        lr=1e-2,
        batch_size=6,
        init_random_frames=0,
        num_updates=1,
    )
    alg._obs_shape = OBS_SHAPE
    alg._num_actions = 2
    alg._buffer_device = torch.device("cpu")
    alg.embedding_net = NatureEmbedding(OBS_SHAPE, EMBED_DIM)
    alg.dnd = DND(2, 200, 2, 1e-3, torch.device("cpu"))
    alg.optimizer = torch.optim.RMSprop(alg.embedding_net.parameters(), lr=alg.lr)
    alg.replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=100, device="cpu")
    )

    # Six distinct frames: a static "paddle" plus a moving 2x2 block. Distinct
    # matters — write_batch de-duplicates by quantised key, and _gradient_step
    # skips any action with _sizes[a] <= k, so repeats would silently make the
    # whole loss computation a no-op.
    obs = torch.zeros(6, *OBS_SHAPE)
    obs[:, :, 40:44, 10:12] = 1.0
    for i in range(6):
        obs[i, :, 10 + 12 * i: 12 + 12 * i, 50:52] = 1.0

    actions = torch.tensor([0, 0, 0, 1, 1, 1])
    targets = torch.tensor([1.0, -1.0, 0.0, -1.0, 1.0, 0.0])

    alg.replay_buffer.extend(
        TensorDict(
            {"obs": obs, "action": actions, "n_step_return": targets},
            batch_size=[6],
        )
    )
    with torch.no_grad():
        h0 = nn.functional.normalize(alg.embedding_net(obs), dim=-1)
    alg.dnd.write_batch(0, h0[0:3], targets[0:3], 1.0)
    alg.dnd.write_batch(1, h0[3:6], targets[3:6], 1.0)
    assert all(s > alg.k for s in alg.dnd._sizes), (
        f"DND sizes {alg.dnd._sizes} <= k={alg.k}: _gradient_step would skip "
        "every action and these tests would assert on nothing"
    )
    return alg


def test_gradient_reaches_keys_and_values_and_cnn():
    """Paper Fig. 2: gradients flow through the ENTIRE architecture."""
    alg = _make_algorithm(key_lr=1e-2, value_lr=1e-3)

    keys_before = alg.dnd.keys.clone()
    values_before = alg.dnd.values.clone()
    cnn_before = next(alg.embedding_net.parameters()).clone()

    result = alg._gradient_step()
    assert result is not None, "gradient step was skipped; nothing was exercised"

    assert not torch.equal(cnn_before, next(alg.embedding_net.parameters())), (
        "embedding network did not move"
    )
    assert not torch.equal(keys_before, alg.dnd.keys), (
        "DND.keys did not move — gradients are not reaching the stored keys, "
        "so they will go stale as the CNN drifts (paper Fig. 2 requires this)"
    )
    assert not torch.equal(values_before, alg.dnd.values), (
        "DND.values did not move — gradients are not reaching the stored values"
    )


def test_untouched_slots_are_bit_identical():
    """Only retrieved slots may change.

    Guards two regressions at once: making keys/values autograd leaves (which
    would allocate a dense num_actions x capacity x d gradient), and using a
    stateful optimiser (whose momentum would decay every slot in the table on
    every step, and whose per-slot state goes stale when the ring buffer
    overwrites a slot — the bug that previously drove stored values negative).
    """
    alg = _make_algorithm(key_lr=1e-2, value_lr=1e-3)
    keys_before = alg.dnd.keys.clone()
    values_before = alg.dnd.values.clone()

    assert alg._gradient_step() is not None

    for a in range(alg._num_actions):
        dead = slice(alg.dnd._sizes[a], alg.dnd.capacity)
        assert _bitwise_equal(alg.dnd.keys[a, dead], keys_before[a, dead]), (
            f"action {a}: slots beyond _sizes were modified"
        )
        assert _bitwise_equal(alg.dnd.values[a, dead], values_before[a, dead]), (
            f"action {a}: values beyond _sizes were modified"
        )


def test_keys_stay_unit_norm_after_gradient():
    """A raw gradient step pushes keys off the unit sphere.

    Every key is written L2-normalised; if an update leaves them off the
    sphere the inverse-distance kernel is dominated by kernel_delta again —
    the exact failure tests/test_nec_kernel_scale.py exists to prevent.
    Uses a deliberately large key_lr so any missing re-projection is obvious.
    """
    alg = _make_algorithm(key_lr=1.0, value_lr=1e-3)
    assert alg._gradient_step() is not None

    live = torch.cat(
        [alg.dnd.keys[a, : alg.dnd._sizes[a]] for a in range(alg._num_actions)]
    )
    norms = live.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_exact_match_hash_never_points_at_a_moved_key():
    """`_key_to_slot` maps a quantised copy of a stored key to its slot.

    Once a gradient moves that key the entry is stale, and a later
    `write_batch` would blend a value into a slot whose key is no longer the
    one that hashed there. `flush_moved_slots()` delists them.
    """
    alg = _make_algorithm(key_lr=1e-2, value_lr=1e-3)
    assert alg._gradient_step() is not None

    delisted = alg.dnd.flush_moved_slots()
    assert delisted > 0, "no slots were delisted despite keys having moved"

    stale = []
    for a in range(alg._num_actions):
        for key, slot in alg.dnd._key_to_slot[a].items():
            current = alg.dnd._make_keys(alg.dnd.keys[a, slot: slot + 1])[0]
            if current != key:
                stale.append((a, slot))
    assert not stale, (
        f"{len(stale)} exact-match entries still hash to a key that has moved: "
        f"{stale[:5]}. write_batch would blend into the wrong slot."
    )

    # Draining twice must be idempotent, not double-delist.
    assert alg.dnd.flush_moved_slots() == 0


def test_zero_learning_rates_leave_the_dnd_untouched():
    """A safety valve: dnd_key_lr=dnd_value_lr=0 recovers the old frozen-DND
    behaviour exactly, so the change can be A/B'd against previous runs."""
    alg = _make_algorithm(key_lr=0.0, value_lr=0.0)
    keys_before = alg.dnd.keys.clone()
    values_before = alg.dnd.values.clone()

    assert alg._gradient_step() is not None

    assert _bitwise_equal(keys_before, alg.dnd.keys)
    assert _bitwise_equal(values_before, alg.dnd.values)
