"""NEC must report what its DND did during an evaluation rollout.

`eval/return_mean` alone cannot separate the three ways NEC evaluates badly,
and they need opposite fixes:

1. the memory holds a bad policy;
2. the memory is not being *used* — the inverse-distance kernel degenerates to
   a flat mean over all `k` neighbours, so every action of a state scores
   alike and the argmax is noise however full the tables are;
3. the evaluation policy is not the policy that was trained — `eval_eps` and
   the annealed training floor `eps_end` are independent knobs living in
   different config files, and when they drift apart the run trains one policy
   and scores another.

MFEC has reported (1) vs (2) via `eval/exact_hit_rate` since it was written;
NEC shipped with `BaseAlgorithm`'s no-op `eval_metrics()` and reported none of
it, which is why a real Ms. Pac-Man run showing `eval/return_mean` at half of
`train/episode_reward` — at the same episode length — could not be diagnosed
from its logs.

These tests pin the read-path counters and the mismatch guard.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.nec import DND, NECAlgorithm
from src.networks import NatureEmbedding


OBS_SHAPE = (4, 84, 84)
NUM_ACTIONS = 6
EMB_DIM = 8
K = 4


class _MockAtariEnv:
    """NECAlgorithm.setup() only reads these three attributes."""

    def __init__(self) -> None:
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=OBS_SHAPE, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=NUM_ACTIONS)
        self.batch_size = torch.Size([])


def _make(eval_eps: float = 0.05, eps_end: float = 0.05) -> NECAlgorithm:
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=NatureEmbedding,
        obs_key="pixels",
        embedding_dim=EMB_DIM,
        dnd_capacity=200,
        k=K,
        eps_end=eps_end,
        eval_eps=eval_eps,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        alg.setup(_MockAtariEnv)
    return alg


def _filled_dnd(spread: float, n: int = 50) -> DND:
    """A DND whose keys sit in a cone of the given angular ``spread``.

    ``spread`` is the knob the ``top_weight`` metric has to be sensitive to: a
    tight cone means every one of the ``k`` neighbours is about equally far
    away, so the kernel weights are near-uniform (top share -> 1/k) even though
    the table is completely full.
    """
    dnd = DND(NUM_ACTIONS, 200, K, 1e-3, torch.device("cpu"))
    dnd._init_keys(EMB_DIM)
    base = nn.functional.normalize(torch.randn(EMB_DIM), dim=-1)
    for a in range(NUM_ACTIONS):
        keys = nn.functional.normalize(
            base + spread * torch.randn(n, EMB_DIM), dim=-1
        )
        dnd.keys[a, :n] = keys
        dnd.values[a, :n] = torch.randn(n)
        dnd._sizes[a] = n
    return dnd


def _query(dnd: DND, b: int = 16) -> None:
    dnd.estimate_all(nn.functional.normalize(torch.randn(b, EMB_DIM), dim=-1))


def test_recording_is_off_until_reset_eval_metrics() -> None:
    """The collector hot path must not pay for eval instrumentation."""
    dnd = _filled_dnd(spread=0.5)
    _query(dnd)
    assert dnd.lookup_stats() == {}


def test_counts_every_state_action_pair() -> None:
    dnd = _filled_dnd(spread=0.5)
    dnd.reset_lookup_stats()
    dnd._record_lookups = True
    _query(dnd, b=16)
    stats = dnd.lookup_stats()
    # |A| queries per frame, not one — same denominator as MFEC's hit rate.
    assert stats["queries"] == 16 * NUM_ACTIONS
    assert stats["optimistic_rate"] == 0.0


def test_sparse_tables_count_as_optimistic() -> None:
    """Actions at/below k answer +inf; that has to show up, not vanish."""
    dnd = _filled_dnd(spread=0.5)
    dnd._sizes[0] = K  # <= k -> sentinel for this action only
    dnd.reset_lookup_stats()
    dnd._record_lookups = True
    _query(dnd, b=10)
    stats = dnd.lookup_stats()
    assert stats["optimistic_rate"] == pytest.approx(1.0 / NUM_ACTIONS)


def test_empty_table_is_all_optimistic() -> None:
    dnd = DND(NUM_ACTIONS, 200, K, 1e-3, torch.device("cpu"))
    dnd.reset_lookup_stats()
    dnd._record_lookups = True
    _query(dnd, b=5)
    stats = dnd.lookup_stats()
    assert stats["queries"] == 5 * NUM_ACTIONS
    assert stats["optimistic_rate"] == 1.0
    # Nothing was graded, so no distance/weight numbers are invented.
    assert "nn_dist" not in stats


def test_top_weight_detects_a_degenerate_kernel() -> None:
    """The metric that separates "bad policy" from "memory unused".

    A tight cone of keys makes every neighbour equidistant, so the kernel is a
    flat mean (top share -> 1/k) and the argmax over actions is noise.  A wide
    spread lets the nearest neighbour dominate.
    """
    torch.manual_seed(0)

    def top_weight(spread: float) -> float:
        dnd = _filled_dnd(spread=spread)
        dnd.reset_lookup_stats()
        dnd._record_lookups = True
        _query(dnd, b=32)
        return dnd.lookup_stats()["top_weight"]

    flat = top_weight(spread=0.001)
    peaked = top_weight(spread=1.0)

    assert flat == pytest.approx(1.0 / K, abs=0.02), (
        "a collapsed embedding space must read as a uniform kernel"
    )
    assert peaked > flat


def test_nn_dist_tracks_key_staleness() -> None:
    """Unit-norm embeddings, so nn_dist is a bounded, comparable number."""
    torch.manual_seed(0)
    dnd = _filled_dnd(spread=0.05)
    dnd.reset_lookup_stats()
    dnd._record_lookups = True
    # Queries drawn from the SAME cone as the keys -> near neighbours.
    base = nn.functional.normalize(dnd.keys[0, 0], dim=-1)
    dnd.estimate_all(
        nn.functional.normalize(base + 0.05 * torch.randn(32, EMB_DIM), dim=-1)
    )
    near = dnd.lookup_stats()["nn_dist"]

    dnd.reset_lookup_stats()
    dnd._record_lookups = True
    _query(dnd, b=32)  # queries from all over the sphere -> far neighbours
    far = dnd.lookup_stats()["nn_dist"]

    assert 0.0 <= near < far <= 2.0


def test_algorithm_reports_eval_epsilon_next_to_the_returns() -> None:
    """The one number that makes a train/eval exploration mismatch visible."""
    alg = _make(eval_eps=0.05, eps_end=0.05)
    alg.dnd = _filled_dnd(spread=0.5)
    alg.reset_eval_metrics()
    _query(alg.dnd, b=8)
    metrics = alg.eval_metrics()

    assert metrics["eval/epsilon"] == pytest.approx(0.05)
    assert "eval/dnd_top_weight" in metrics
    assert "eval/dnd_optimistic_rate" in metrics
    # Recording is switched back off so the collector does not keep paying.
    assert alg.dnd._record_lookups is False


def test_eval_metrics_empty_when_nothing_was_queried() -> None:
    """An absent metric leaves a gap in the chart; a zero would be a lie."""
    alg = _make()
    alg.reset_eval_metrics()
    assert alg.eval_metrics() == {}


def test_setup_warns_when_eval_and_train_epsilon_diverge() -> None:
    """The exact drift that produced the unexplained Ms. Pac-Man eval gap."""
    with pytest.warns(UserWarning, match="exploration mismatch"):
        NECAlgorithm(
            device=torch.device("cpu"),
            embedding_network=NatureEmbedding,
            obs_key="pixels",
            embedding_dim=EMB_DIM,
            dnd_capacity=200,
            k=K,
            eps_end=0.001,
            eval_eps=0.05,
        ).setup(_MockAtariEnv)


def test_setup_is_silent_when_they_agree() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _make(eval_eps=0.05, eps_end=0.05)


def test_lookup_counters_survive_a_checkpoint_round_trip() -> None:
    """__setstate__ bypasses __init__; the read path must not break on resume."""
    dnd = _filled_dnd(spread=0.5)
    restored = DND(NUM_ACTIONS, 200, K, 1e-3, torch.device("cpu"))
    restored.__setstate__(dnd.__getstate__())

    _query(restored, b=4)  # would raise AttributeError without the re-init
    assert restored.lookup_stats() == {}

    restored.reset_lookup_stats()
    restored._record_lookups = True
    _query(restored, b=4)
    assert restored.lookup_stats()["queries"] == 4 * NUM_ACTIONS
