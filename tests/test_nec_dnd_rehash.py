"""Regression tests: gradient-moved DND slots must be RE-HASHED, not delisted.

``apply_gradient`` moves a retrieved key, which invalidates the quantised
exact-match hash stored in ``_key_to_slot``.  ``flush_moved_slots`` used to
handle that by dropping the slot from the dict permanently.  Because the kNN
eventually retrieves essentially every entry, that removed essentially every
entry from the exact-match dict over a run, which silently disabled the paper's
blend rule (§3.3, Eq. 4) — NEC's headline "rapidly updated estimates of the
value function" — and made re-encounters insert duplicates instead of updating.

These tests pin the repaired behaviour: after a gradient step, the moved slot
is still reachable by its CURRENT key, so a later write blends into it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.nec import DND

A, D, K = 3, 8, 2


def _fresh(capacity: int = 64) -> DND:
    torch.manual_seed(0)
    return DND(
        num_actions=A,
        capacity=capacity,
        k=K,
        kernel_delta=1e-3,
        device=torch.device("cpu"),
    )


def _unit(n: int) -> torch.Tensor:
    return nn.functional.normalize(torch.randn(n, D), dim=-1)


def _invariants_hold(dnd: DND) -> bool:
    """``_key_to_slot`` and ``_slot_to_key`` must agree wherever both exist."""
    for a in range(dnd.num_actions):
        k_to_s, s_to_k = dnd._key_to_slot[a], dnd._slot_to_key[a]
        for key, slot in k_to_s.items():
            if s_to_k.get(slot) != key:
                return False
        for slot, key in s_to_k.items():
            if k_to_s.get(key) != slot:
                return False
    return True


def test_moved_slot_is_rehashed_not_dropped():
    dnd = _fresh()
    keys = _unit(5)
    dnd.write_batch(0, keys, torch.zeros(5), dnd_lr=0.1)
    assert len(dnd._key_to_slot[0]) == 5

    # Move slots 1 and 3 by a gradient step large enough to change the hash.
    idx = torch.tensor([[1, 3]])
    grad = torch.full((1, 2, D), 5.0)
    dnd.apply_gradient(0, idx, key_grad=grad, value_grad=None,
                       key_lr=1e-1, value_lr=0.0)
    n = dnd.flush_moved_slots()

    assert n == 2, "both moved slots should be reported as re-hashed"
    assert len(dnd._key_to_slot[0]) == 5, (
        "re-hashing must preserve dict size; delisting would shrink it"
    )
    assert _invariants_hold(dnd)

    # The moved slots must be reachable by their CURRENT stored keys.
    for slot in (1, 3):
        cur = dnd._make_keys(dnd.keys[0, slot: slot + 1])[0]
        assert dnd._key_to_slot[0].get(cur) == slot


def test_blend_rule_still_fires_after_a_gradient_step():
    """The point of re-hashing: a re-encounter updates rather than duplicates."""
    dnd = _fresh()
    keys = _unit(4)
    dnd.write_batch(0, keys, torch.tensor([10.0, 20.0, 30.0, 40.0]), dnd_lr=0.1)
    size_before = dnd._sizes[0]

    idx = torch.tensor([[2]])
    dnd.apply_gradient(0, idx, key_grad=torch.full((1, 1, D), 5.0),
                       value_grad=None, key_lr=1e-1, value_lr=0.0)
    dnd.flush_moved_slots()

    moved_key = dnd.keys[0, 2:3].clone()
    before = float(dnd.values[0, 2])

    hits, novel = dnd.write_batch(0, moved_key, torch.tensor([100.0]), dnd_lr=0.5)

    assert (hits, novel) == (1, 0), "re-encounter must blend, not insert"
    assert dnd._sizes[0] == size_before, "no new slot may be consumed"
    assert float(dnd.values[0, 2]) == before + 0.5 * (100.0 - before)


def test_hash_collision_keeps_mapping_consistent():
    """Two slots quantising to the same key must not corrupt the dicts."""
    dnd = _fresh()
    keys = _unit(3)
    dnd.write_batch(0, keys, torch.zeros(3), dnd_lr=0.1)

    # Force slots 0 and 1 onto the same stored key, then re-hash both.
    dnd.keys[0, 1] = dnd.keys[0, 0]
    dnd._moved_slots[0].append(torch.tensor([0, 1]))
    dnd.flush_moved_slots()

    assert _invariants_hold(dnd), "collision must leave a consistent mapping"
    # Exactly one of the two colliding slots may own the shared key.
    shared = dnd._make_keys(dnd.keys[0, 0:1])[0]
    assert dnd._key_to_slot[0][shared] in (0, 1)


def test_zero_key_lr_leaves_dicts_untouched():
    """``dnd_key_lr=0`` must remain a bit-exact no-op, including for hashes."""
    dnd = _fresh()
    keys = _unit(4)
    dnd.write_batch(0, keys, torch.zeros(4), dnd_lr=0.1)
    snapshot = dict(dnd._key_to_slot[0])

    dnd.apply_gradient(0, torch.tensor([[0, 1]]),
                       key_grad=torch.ones(1, 2, D), value_grad=None,
                       key_lr=0.0, value_lr=0.0)
    assert dnd.flush_moved_slots() == 0
    assert dnd._key_to_slot[0] == snapshot
