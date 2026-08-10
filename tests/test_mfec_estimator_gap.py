"""MFEC evaluates greedily, and reports the Eq. (2) estimator gap.

Two related regressions are guarded here.

1. ``eval_eps`` defaults to 0.0.
   It used to be the paper's 0.005, so that ``num_eval_episodes`` produced
   more than one distinct sample on a deterministic ALE.  Measured cost of
   that (Ms. Pac-Man, one QEC, 6 episodes each)::

       eval_eps=0.000   [1440, 1440, 1440, 1440, 1440, 1440]   mean 1440
       eval_eps=0.005   [ 490, 1440,  870,  550, 1440, 1440]   mean 1038

   MFEC replays a memorised trajectory and cannot absorb a single
   off-trajectory action, so ε at evaluation understates the policy ~30% and
   pins ``eval/return_min`` near the floor *permanently* — it becomes the
   worst of N ε-derailments rather than anything about the memory.  The
   ε=0.005 score the paper reports is the collector's, logged as
   ``train/episode_reward``.

2. ``eval/exact_minus_knn_value`` is reported.
   Eq. (2) answers a query with a stored value (a max over realised returns)
   on an exact match and a k-neighbour mean otherwise, and ``argmax`` then
   compares the two.  On Ms. Pac-Man the exact branch runs ~540 points above
   the kNN branch for the *same* (s, a), so a tried action beats an untried
   one on estimator bias alone.  The metric makes that visible per encoder.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from src.algorithms.mfec import QEC, MFECAlgorithm

NUM_ACTIONS = 4
STATE_DIM = 3


class _ProofEnv(EnvBase):
    batch_locked = False

    def __init__(self):
        super().__init__(device="cpu", batch_size=torch.Size([]))
        self.observation_spec = Composite(
            pixels=Unbounded(shape=torch.Size([1, 4, 4]), dtype=torch.float32)
        )
        self.action_spec = Categorical(NUM_ACTIONS)
        self.reward_spec = Unbounded(shape=torch.Size([1]))

    def _reset(self, tensordict=None, **kw):
        raise NotImplementedError

    def _step(self, tensordict):
        raise NotImplementedError

    def _set_seed(self, seed):
        return seed


def test_eval_eps_defaults_to_greedy():
    algorithm = MFECAlgorithm(device=torch.device("cpu"), state_dim=STATE_DIM)
    assert algorithm.eval_eps == 0.0, (
        "eval_eps is non-zero again: eval/return_min then measures the worst "
        "epsilon-derailment out of num_eval_episodes, not the policy."
    )


def test_eval_policy_is_deterministic_at_the_default():
    algorithm = MFECAlgorithm(
        device=torch.device("cpu"), state_dim=STATE_DIM, k=1, buffer_size=32,
        frames_per_batch=8,
    )
    algorithm.setup(lambda: _ProofEnv())
    assert float(algorithm.eval_greedy_module.eps) == 0.0
    # The module must still be in the chain, so raising eval_eps has an effect
    # (a stock EGreedyModule would be a no-op under ExplorationType.MODE).
    from src.algorithms.eval_policy import EvalEGreedyModule

    assert isinstance(list(algorithm.get_policy().module)[-1], EvalEGreedyModule)


def _filled_qec() -> QEC:
    """A QEC whose exact values sit far above its neighbourhood means."""
    qec = QEC(NUM_ACTIONS, capacity=256, k=2, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    states = torch.tensor(rng.normal(size=(64, STATE_DIM)), dtype=torch.float32)
    for a in range(NUM_ACTIONS):
        qec.add_batch(a, states, torch.full((64,), 100.0, dtype=torch.float64))
    return qec, states


def test_value_stats_expose_the_exact_vs_knn_gap():
    qec, states = _filled_qec()
    # A query that is stored (exact branch) and one that is not (kNN branch).
    qec.reset_lookup_stats()
    qec.estimate_all(states[:8])                       # all exact hits
    e_sum, e_n, _, k_n = qec.value_stats()
    assert e_n == 8 * NUM_ACTIONS and k_n == 0
    assert e_sum / e_n == pytest.approx(100.0)

    qec.reset_lookup_stats()
    novel = states[:8] + 5.0                           # far from every entry
    qec.estimate_all(novel)
    e_sum, e_n, k_sum, k_n = qec.value_stats()
    assert e_n == 0 and k_n == 8 * NUM_ACTIONS
    assert k_sum / k_n == pytest.approx(100.0)


def test_eval_metrics_reports_the_gap():
    algorithm = MFECAlgorithm(
        device=torch.device("cpu"), state_dim=STATE_DIM, k=2, buffer_size=256,
        frames_per_batch=8,
    )
    algorithm.setup(lambda: _ProofEnv())
    qec, states = _filled_qec()
    algorithm.qec = qec

    algorithm.reset_eval_metrics()
    assert algorithm.eval_metrics() == {}, "no queries yet -> no metrics"

    # Exact entries are worth 100.  Plant a tight, *distinct-keyed* cluster far
    # from them, worth 10, so a query inside it has both its k=2 neighbours in
    # the cluster and the kNN branch returns 10 rather than a blend.
    far = torch.full((4, STATE_DIM), 50.0) + 0.01 * torch.arange(4).unsqueeze(1)
    for a in range(NUM_ACTIONS):
        qec.add_batch(a, far, torch.full((4,), 10.0, dtype=torch.float64))

    algorithm.reset_eval_metrics()
    # +0.05 keeps the query clear of the near-exact tolerance (~2.6e-3 at this
    # norm) while staying orders of magnitude closer to the cluster than to the
    # unit-scale entries.
    qec.estimate_all(torch.cat([states[:8], far[:1] + 0.05]))
    metrics = algorithm.eval_metrics()

    assert "eval/exact_minus_knn_value" in metrics
    assert metrics["eval/exact_minus_knn_value"] == pytest.approx(90.0)
    assert metrics["eval/exact_hit_rate"] == pytest.approx(8 / 9)
