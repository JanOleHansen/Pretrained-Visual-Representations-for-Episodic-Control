"""Regression tests for LRU eviction in the DND (paper §3.3).

"We overwrite the item that has least recently shown up as a neighbour when we
reach the memory's maximum capacity."

Eviction was FIFO until ``dnd_capacity`` was cut 5e5 -> 5e4 for throughput,
which made the eviction regime reachable inside a normal run (~450k agent
steps) for the first time.  FIFO is specifically wrong once the blend rule is
live: a frequently re-encountered state occupies one slot whose value is
refreshed but whose insertion order is not, so FIFO discards the
best-estimated, most-reused entries on a fixed timer.

These tests pin: recency is stamped on retrieval, cold slots are the ones
evicted, hot slots survive, the slot<->key dicts stay consistent across an
eviction, and recency round-trips through a checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.nec import DND

D = 8


def _dnd(capacity: int, num_actions: int = 2, k: int = 2) -> DND:
    torch.manual_seed(0)
    return DND(
        num_actions=num_actions,
        capacity=capacity,
        k=k,
        kernel_delta=1e-3,
        device=torch.device("cpu"),
    )


def _unit(n: int) -> torch.Tensor:
    return nn.functional.normalize(torch.randn(n, D), dim=-1)


def _invariants_hold(dnd: DND) -> bool:
    for a in range(dnd.num_actions):
        k_to_s, s_to_k = dnd._key_to_slot[a], dnd._slot_to_key[a]
        for key, slot in k_to_s.items():
            if s_to_k.get(slot) != key:
                return False
        for slot, key in s_to_k.items():
            if k_to_s.get(key) != slot:
                return False
    return True


def test_no_eviction_while_the_table_has_room():
    dnd = _dnd(capacity=10)
    dnd.write_batch(0, _unit(4), torch.arange(4.0), dnd_lr=0.1)
    dnd.write_batch(0, _unit(3), torch.arange(3.0), dnd_lr=0.1)
    assert dnd._sizes[0] == 7
    assert dnd._write_ptrs[0] == 7, "ptr == size is the fill-phase invariant"
    assert len(dnd._key_to_slot[0]) == 7


def test_retrieval_stamps_recency():
    dnd = _dnd(capacity=16)
    dnd.write_batch(0, _unit(8), torch.arange(8.0), dnd_lr=0.1)
    before = dnd._lru[0, :8].clone()

    dnd.knn_action(dnd.keys[0, 3:4].clone(), 0, k=2)

    after = dnd._lru[0, :8]
    assert after[3] > before[3], "the queried slot must be marked recent"
    assert int((after > before).sum()) == 2, "exactly k slots were returned"


def test_lru_evicts_cold_slots_and_keeps_hot_ones():
    cap = 8
    dnd = _dnd(capacity=cap)
    keys = _unit(cap)
    dnd.write_batch(0, keys, torch.arange(float(cap)), dnd_lr=0.1)
    assert dnd._sizes[0] == cap

    # Slots 0 and 1 are kept hot; 2..7 are never touched again.
    hot_keys = dnd.keys[0, 0:2].clone()
    for _ in range(3):
        dnd.knn_action(hot_keys, 0, k=1)

    hot_before = hot_keys.clone()
    dnd.write_batch(0, _unit(4), torch.full((4,), 99.0), dnd_lr=0.1)

    assert dnd._sizes[0] == cap, "capacity must not be exceeded"
    assert _invariants_hold(dnd)

    # The two hot keys must still be present somewhere in the table.
    for row in hot_before:
        d = (dnd.keys[0, :cap] - row).norm(dim=-1)
        assert float(d.min()) < 1e-5, "an actively retrieved entry was evicted"


def test_freshly_inserted_entries_are_not_immediately_evicted():
    """Without stamping on write, new rows inherit the victim's cold stamp."""
    cap = 6
    dnd = _dnd(capacity=cap)
    dnd.write_batch(0, _unit(cap), torch.zeros(cap), dnd_lr=0.1)

    first = _unit(2)
    dnd.write_batch(0, first, torch.full((2,), 1.0), dnd_lr=0.1)
    second = _unit(2)
    dnd.write_batch(0, second, torch.full((2,), 2.0), dnd_lr=0.1)

    for row in first:
        d = (dnd.keys[0, :cap] - row).norm(dim=-1)
        assert float(d.min()) < 1e-5, (
            "the previous insert was evicted by the very next one"
        )


def test_eviction_keeps_slot_key_dicts_consistent():
    cap = 5
    dnd = _dnd(capacity=cap)
    dnd.write_batch(0, _unit(cap), torch.zeros(cap), dnd_lr=0.1)
    for _ in range(4):
        dnd.write_batch(0, _unit(3), torch.ones(3), dnd_lr=0.1)
        assert dnd._sizes[0] == cap
        assert _invariants_hold(dnd)
        assert len(dnd._key_to_slot[0]) <= cap
        assert len(dnd._slot_to_key[0]) <= cap


def test_oversized_insert_is_truncated_to_capacity():
    cap = 4
    dnd = _dnd(capacity=cap)
    dnd.write_batch(0, _unit(cap), torch.zeros(cap), dnd_lr=0.1)
    dnd.write_batch(0, _unit(10), torch.ones(10), dnd_lr=0.1)
    assert dnd._sizes[0] == cap
    assert _invariants_hold(dnd)


def test_recency_survives_a_checkpoint_round_trip():
    dnd = _dnd(capacity=8)
    dnd.write_batch(0, _unit(8), torch.arange(8.0), dnd_lr=0.1)
    dnd.knn_action(dnd.keys[0, 5:6].clone(), 0, k=1)
    saved_lru = dnd._lru[0, :8].clone()
    saved_tick = dnd._tick

    restored = _dnd(capacity=8)
    restored.__setstate__(dnd.__getstate__())

    assert restored._tick == saved_tick
    assert torch.equal(restored._lru[0, :8], saved_lru), (
        "recency must round-trip, or a resume evicts near-randomly"
    )


def test_pre_lru_checkpoint_loads_without_recency():
    """Old checkpoints have no 'action_lru'; they must still restore."""
    dnd = _dnd(capacity=8)
    dnd.write_batch(0, _unit(8), torch.arange(8.0), dnd_lr=0.1)
    state = dnd.__getstate__()
    del state["action_lru"]
    del state["_tick"]

    restored = _dnd(capacity=8)
    restored.__setstate__(state)

    assert restored._sizes[0] == 8
    assert int(restored._lru[0, :8].max()) == -1, "unknown recency stays cold"
    assert _invariants_hold(restored)
