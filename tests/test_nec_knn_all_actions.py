"""Regression tests for ``DND.knn_all_actions`` / ``DND._block_topk``.

``_block_topk`` ranks candidates by *similarity* and only converts the ``k``
winners to distances, instead of materialising ``2 - 2*sim`` -> ``clamp`` ->
``sqrt`` -> ``masked_fill`` over the whole ``(A, b, n)`` matrix.  For unit-norm
vectors ``‖q - h‖² = 2 - 2·q·h`` is monotone decreasing in ``q·h``, so this is
a reassociation and must return **exactly** the same neighbours and distances
as ranking on distance directly.

These tests pin that equivalence against a brute-force ``torch.cdist``
reference, because the rewrite is on the ε-greedy hot path (one call per env
per collector batch) and a silent regression there degrades the DND lookup
without failing anything else.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.algorithms.nec import DND

A, D, K = 5, 16, 8


def _build(sizes: list[int], capacity: int, *, unit_norm: bool = True) -> DND:
    torch.manual_seed(0)
    dnd = DND(
        num_actions=len(sizes),
        capacity=capacity,
        k=K,
        kernel_delta=1e-3,
        device=torch.device("cpu"),
        unit_norm_keys=unit_norm,
    )
    dnd._init_keys(D)
    for a, s in enumerate(sizes):
        if s:
            keys = torch.randn(s, D)
            if unit_norm:
                keys = nn.functional.normalize(keys, dim=-1)
            dnd.keys[a, :s] = keys
            dnd.values[a, :s] = torch.randn(s)
        dnd._sizes[a] = s
    return dnd


def _reference(dnd: DND, queries: torch.Tensor, k_eff: int, action: int):
    """Brute-force top-k over the live slots of one action."""
    size = dnd._sizes[action]
    m = queries.shape[0]
    if size == 0:
        return torch.full((m, k_eff), float("inf")), None
    d = torch.cdist(queries, dnd.keys[action, :size])
    if size < k_eff:  # pad dead slots exactly as the masking does
        d = torch.cat([d, torch.full((m, k_eff - size), float("inf"))], dim=1)
    return d.topk(k_eff, dim=1, largest=False)


@pytest.mark.parametrize(
    "sizes, capacity, tag",
    [
        ([30, 25, 40, 9, 12], 40, "ragged, some below k"),
        ([0, 3, 9, 10, 40], 40, "empty and near-threshold tables"),
        ([64] * A, 64, "uniform, all full"),
        ([1, 1, 1, 1, 1], 8, "single entry per action"),
    ],
)
def test_knn_all_actions_matches_bruteforce(sizes, capacity, tag):
    dnd = _build(sizes, capacity)
    queries = nn.functional.normalize(torch.randn(13, D), dim=-1)

    got_d, got_i = dnd.knn_all_actions(queries, K)
    k_eff = min(K, max(sizes))
    assert got_d.shape == (len(sizes), 13, k_eff), tag

    for a, size in enumerate(sizes):
        ref_d, _ = _reference(dnd, queries, k_eff, a)

        finite = torch.isfinite(ref_d)
        assert torch.equal(finite, torch.isfinite(got_d[a])), (
            f"{tag}: +inf padding disagrees for action {a}"
        )
        assert torch.allclose(ref_d[finite], got_d[a][finite], atol=1e-5), (
            f"{tag}: distances disagree for action {a}"
        )

        # Returned indices must actually address the returned distances.
        if size:
            recomputed = torch.cdist(queries, dnd.keys[a, :size])
            live = torch.isfinite(got_d[a])
            picked = recomputed.gather(1, got_i[a].clamp(max=size - 1))
            assert torch.allclose(picked[live], got_d[a][live], atol=1e-5), (
                f"{tag}: indices do not match distances for action {a}"
            )


def test_knn_all_actions_chunked_path_matches_unchunked():
    """Force the multi-chunk merge branch and check it agrees with one shot."""
    sizes = [200] * A
    dnd = _build(sizes, 200)
    queries = nn.functional.normalize(torch.randn(9, D), dim=-1)

    full_d, _ = dnd.knn_all_actions(queries, K)

    original = DND._CHUNK_BYTES
    try:
        # Small enough that k_chunk < max_size, exercising the cat/topk/gather
        # merge that the single-shot path skips.
        DND._CHUNK_BYTES = A * 9 * 4 * 3
        chunked_d, _ = dnd.knn_all_actions(queries, K)
    finally:
        DND._CHUNK_BYTES = original

    assert torch.allclose(full_d, chunked_d, atol=1e-5)


def test_non_unit_norm_path_still_uses_cdist():
    """``unit_norm_keys=False`` must fall back to the general metric."""
    sizes = [30] * A
    dnd = _build(sizes, 30, unit_norm=False)
    queries = torch.randn(7, D) * 5.0  # deliberately far off the unit sphere

    got_d, _ = dnd.knn_all_actions(queries, K)
    for a in range(A):
        ref_d, _ = _reference(dnd, queries, K, a)
        assert torch.allclose(ref_d, got_d[a], atol=1e-4)


def test_estimate_all_unaffected_by_rewrite():
    """End-to-end: kernel-weighted Q must match a hand-rolled computation."""
    sizes = [60] * A
    dnd = _build(sizes, 60)
    queries = nn.functional.normalize(torch.randn(4, D), dim=-1)

    got = dnd.estimate_all(queries)          # (B, A)

    for a in range(A):
        d = torch.cdist(queries, dnd.keys[a, :sizes[a]])
        nd, ni = d.topk(K, dim=1, largest=False)
        w = 1.0 / (nd ** 2 + dnd.kernel_delta)
        expected = (w * dnd.values[a, :sizes[a]][ni]).sum(1) / w.sum(1)
        assert torch.allclose(got[:, a], expected, atol=1e-4)
