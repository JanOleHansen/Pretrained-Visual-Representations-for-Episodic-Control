"""NEC's evaluation policy must be stochastic, and must stay stochastic.

`BaseTrainer.evaluate()` runs under `set_exploration_type(ExplorationType.MODE)`,
and torchrl's `EGreedyModule.forward` is gated on
`exploration_type() in (ExplorationType.RANDOM, None)`.  So dropping a *stock*
`EGreedyModule` into the eval chain looks correct and is silently a no-op —
`src/algorithms/eval_policy.py::EvalEGreedyModule` exists to force that gate.

Why it matters for NEC: the eval env sets `repeat_action_probability=0.0`, ALE
is deterministic, and `NoopResetEnv` does not perturb Ms. Pac-Man's opening
enough to matter.  A deterministic policy therefore replays the same trajectory
every episode, `eval/return_std` is identically 0, and `num_eval_episodes`
silently collapses to a single sample at N times the cost.  Observed on a real
2.25M-step run before this was wired up.

These tests are cheap and catch the regression that a torchrl upgrade (changing
the exploration gate) or a refactor of `get_policy()` would otherwise hide.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms.eval_policy import EvalEGreedyModule
from src.algorithms.nec import NECAlgorithm
from src.networks import NatureEmbedding


OBS_SHAPE = (4, 84, 84)
NUM_ACTIONS = 6


class _MockAtariEnv:
    """NECAlgorithm.setup() only reads these three attributes."""

    def __init__(self) -> None:
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=OBS_SHAPE, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=NUM_ACTIONS)
        self.batch_size = torch.Size([])


def _make(eval_eps: float) -> NECAlgorithm:
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=NatureEmbedding,
        obs_key="pixels",
        embedding_dim=64,
        dnd_capacity=500,
        k=2,
        eval_eps=eval_eps,
    )
    alg.setup(_MockAtariEnv)
    return alg


def _actions_under_mode(alg: NECAlgorithm, n: int = 200) -> set[int]:
    """Sample the eval policy exactly as BaseTrainer.evaluate() drives it."""
    policy = alg.get_policy()
    obs = torch.randint(0, 256, OBS_SHAPE, dtype=torch.uint8)
    seen: set[int] = set()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for _ in range(n):
            td = policy(TensorDict({"pixels": obs}, batch_size=[]))
            seen.add(int(td["action"]))
    return seen


def test_eval_policy_ends_in_eval_egreedy_module():
    alg = _make(0.001)
    assert any(isinstance(m, EvalEGreedyModule) for m in alg.get_policy().module), (
        "get_policy() must end in EvalEGreedyModule. A bare QValueActor makes "
        "evaluation deterministic and eval/return_std identically 0."
    )


def test_eval_policy_is_stochastic_under_exploration_mode():
    """The regression that matters: a stock EGreedyModule would give 1 action."""
    seen = _actions_under_mode(_make(0.9))
    assert len(seen) > 1, (
        f"eval policy produced only {seen} across 200 calls under "
        "ExplorationType.MODE — epsilon is being ignored, which is what a stock "
        "EGreedyModule does. torchrl's exploration gate may have changed."
    )


def test_zero_eval_eps_restores_deterministic_argmax():
    """eval_eps=0.0 must be an exact no-op, so old runs stay reproducible."""
    assert len(_actions_under_mode(_make(0.0))) == 1


def test_eval_eps_is_constant_not_annealed():
    """The eval rate is a constant. If it were annealed by the training loop,
    evaluation would silently drift toward argmax over a long run."""
    alg = _make(0.05)
    before = float(alg.eval_greedy_module.eps)
    alg.greedy_module.step(100_000)          # anneal the *training* module hard
    assert float(alg.eval_greedy_module.eps) == before, (
        "eval epsilon moved when the training epsilon annealed — they must be "
        "separate modules."
    )
