"""The kNN uses a dot-product identity that is only valid for unit-norm data.

`_topk_l2_unit` exploits `||q - k||^2 = 2 - 2*q.k`, so ordering by ascending
distance is ordering by descending dot product. That skips `torch.cdist`'s
norm computation over the whole key block on every call — ~5x at Atari table
sizes, in the m~4 regime `_gradient_step` runs in (batch 32 over 9 actions),
where the kNN is ~89% of a gradient update.

It is exact, NOT an approximation, but only because NEC normalises every
embedding before it reaches the DND. These tests pin both halves of that: the
results match `cdist` bit-closely on unit-norm data, and `unit_norm_keys=False`
still gives the general path for anything that cannot guarantee it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.nec import DND, _topk_l2_unit


D, K = 16, 5


def _unit(n, d=D):
    return nn.functional.normalize(torch.randn(n, d), dim=-1)


def _fill(dnd, per_action, seed=0):
    torch.manual_seed(seed)
    for a in range(dnd.num_actions):
        dnd.write_batch(a, _unit(per_action), torch.randn(per_action), 1.0)


def test_helper_matches_cdist_on_unit_norm_data():
    torch.manual_seed(0)
    for n in (50, 500, 5000):
        keys, q = _unit(n), _unit(7)
        d_ref, i_ref = torch.cdist(q, keys).topk(K, dim=1, largest=False)
        d_got, i_got = _topk_l2_unit(q, keys, K)
        torch.testing.assert_close(d_got, d_ref, atol=1e-4, rtol=1e-4)
        # Indices may differ only on exact ties; the distances must not.
        torch.testing.assert_close(
            keys[i_got].sub(q.unsqueeze(1)).pow(2).sum(-1).sqrt(),
            d_ref, atol=1e-4, rtol=1e-4,
        )


def test_knn_action_identical_with_and_without_the_identity():
    torch.manual_seed(1)
    fast = DND(3, 2000, K, 1e-3, torch.device("cpu"), unit_norm_keys=True)
    slow = DND(3, 2000, K, 1e-3, torch.device("cpu"), unit_norm_keys=False)
    _fill(fast, 400); _fill(slow, 400)
    # Only LIVE slots: the tables are allocated with torch.empty, so the dead
    # tail is uninitialised memory and differs between instances.
    for a in range(3):
        assert torch.equal(
            fast.keys[a, : fast._sizes[a]], slow.keys[a, : slow._sizes[a]]
        )

    q = _unit(11)
    for a in range(3):
        d_f, i_f = fast.knn_action(q, a, K)
        d_s, i_s = slow.knn_action(q, a, K)
        torch.testing.assert_close(d_f, d_s, atol=1e-4, rtol=1e-4)


def test_estimate_all_identical_with_and_without_the_identity():
    """The acting policy's Q-values must not shift because of this."""
    torch.manual_seed(2)
    fast = DND(3, 2000, K, 1e-3, torch.device("cpu"), unit_norm_keys=True)
    slow = DND(3, 2000, K, 1e-3, torch.device("cpu"), unit_norm_keys=False)
    _fill(fast, 400); _fill(slow, 400)

    q = _unit(9)
    torch.testing.assert_close(
        fast.estimate_all(q), slow.estimate_all(q), atol=1e-4, rtol=1e-4
    )


def test_multi_chunk_path_is_also_exact():
    """Force the chunked merge and the batched masking branch."""
    torch.manual_seed(3)
    fast = DND(2, 4000, K, 1e-3, torch.device("cpu"), unit_norm_keys=True)
    slow = DND(2, 4000, K, 1e-3, torch.device("cpu"), unit_norm_keys=False)
    _fill(fast, 900); _fill(slow, 900)

    orig = DND._CHUNK_BYTES
    try:
        DND._CHUNK_BYTES = 2000          # tiny -> many chunks, exercises the merge
        q = _unit(6)
        for a in range(2):
            d_f, _ = fast.knn_action(q, a, K)
            d_s, _ = slow.knn_action(q, a, K)
            torch.testing.assert_close(d_f, d_s, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            fast.estimate_all(q), slow.estimate_all(q), atol=1e-4, rtol=1e-4
        )
    finally:
        DND._CHUNK_BYTES = orig
