"""Regression tests for MFEC's evaluation policy.

Bug being guarded against
-------------------------
``MFECAlgorithm.get_policy()`` used to return a bare argmax chain
(``_EmbedModule -> QValueActor``) with no exploration at all.  Combined with
``repeat_action_probability=0.0`` — which MFEC requires, since Eq. (1) is
max-over-returns (footnote 1) — the ALE is fully deterministic, and Ms. Pac-Man's
opening is insensitive to ``NoopResetEnv``.  So every evaluation episode replayed
the *same* trajectory:

    eval/return_std  == 0.0 exactly, for the whole run
    eval/return_min  == eval/return_mean == eval/return_max

i.e. ``num_eval_episodes: 5`` bought one sample at five times the cost, and the
eval curve could never show variance.  Measured on Ms. Pac-Man with one QEC:

    get_policy()         / MODE     mean=380.0  std=0.000   (5/5 identical)
    get_explore_policy() / RANDOM   mean=448.0  std=248.4
    get_explore_policy() / MODE     mean=380.0  std=0.000

The third row is the subtle part, and the reason ``_EvalEGreedyModule`` exists:
torchrl's ``EGreedyModule`` is gated on
``exploration_type() in (ExplorationType.RANDOM, None)``, and
``BaseTrainer.evaluate()`` runs under ``ExplorationType.MODE`` — so just adding a
stock ``EGreedyModule`` to the eval chain would still be a no-op.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Categorical
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule

from src.algorithms.mfec import _EvalEGreedyModule


NUM_ACTIONS = 9        # ALE/MsPacman-v5
BATCH = 4096


def _action_batch(action: int = 0) -> TensorDict:
    """A batch of identical greedy actions for an ε-greedy module to perturb."""
    return TensorDict(
        {"action": torch.full((BATCH,), action, dtype=torch.int64)},
        batch_size=[BATCH],
    )


def _module(cls, eps: float):
    return cls(
        spec=Categorical(NUM_ACTIONS),
        eps_init=eps,
        eps_end=eps,
        annealing_num_steps=1,
    )


# ---------------------------------------------------------------------------
# 1. The ExplorationType gate — the reason a stock EGreedyModule is not enough
# ---------------------------------------------------------------------------

def test_stock_egreedy_is_a_noop_under_mode():
    """Pins the torchrl behaviour this fix works around."""
    torch.manual_seed(0)
    module = _module(EGreedyModule, eps=0.5)

    with set_exploration_type(ExplorationType.MODE):
        out = module(_action_batch(action=3))

    assert (out["action"] == 3).all(), (
        "torchrl changed EGreedyModule's exploration gate — _EvalEGreedyModule "
        "may no longer be necessary; re-check MFECAlgorithm.get_policy()."
    )


def test_eval_egreedy_applies_epsilon_under_mode():
    torch.manual_seed(0)
    module = _module(_EvalEGreedyModule, eps=0.5)

    with set_exploration_type(ExplorationType.MODE):
        out = module(_action_batch(action=3))

    perturbed = (out["action"] != 3).float().mean().item()
    # eps=0.5, and a random draw coincides with the greedy action 1/9 of the
    # time, so the expected perturbed fraction is 0.5 * 8/9 ~= 0.444.
    assert perturbed == pytest.approx(0.444, abs=0.05), perturbed


def test_eval_egreedy_is_a_noop_at_zero_epsilon():
    """eval_eps=0.0 must restore exactly the old deterministic behaviour."""
    torch.manual_seed(0)
    module = _module(_EvalEGreedyModule, eps=0.0)

    with set_exploration_type(ExplorationType.MODE):
        out = module(_action_batch(action=3))

    assert (out["action"] == 3).all()


def test_eval_egreedy_matches_paper_epsilon_rate():
    """At eps=0.005 the eval policy is still ~greedy, just not deterministic."""
    torch.manual_seed(0)
    module = _module(_EvalEGreedyModule, eps=0.005)

    with set_exploration_type(ExplorationType.MODE):
        out = module(_action_batch(action=3))

    perturbed = (out["action"] != 3).float().mean().item()
    assert perturbed > 0.0, "eval policy is deterministic — the bug is back"
    assert perturbed < 0.02, perturbed


# ---------------------------------------------------------------------------
# 2. The algorithm wires it up
# ---------------------------------------------------------------------------

def _setup_algorithm(eval_eps: float):
    """A real MFECAlgorithm.setup() against a tiny stub env spec."""
    from torchrl.envs import EnvBase

    from src.algorithms.mfec import MFECAlgorithm

    class _ProofEnv(EnvBase):
        batch_locked = False

        def __init__(self):
            super().__init__(device="cpu", batch_size=torch.Size([]))
            from torchrl.data import Composite, Unbounded
            self.observation_spec = Composite(
                pixels=Unbounded(shape=torch.Size([1, 8, 8]), dtype=torch.float32)
            )
            self.action_spec = Categorical(NUM_ACTIONS)
            self.reward_spec = Unbounded(shape=torch.Size([1]))

        def _reset(self, tensordict=None, **kw):
            raise NotImplementedError

        def _step(self, tensordict):
            raise NotImplementedError

        def _set_seed(self, seed):
            return seed

    algorithm = MFECAlgorithm(
        device=torch.device("cpu"),
        state_dim=4,
        k=1,
        buffer_size=64,
        frames_per_batch=8,
        eval_eps=eval_eps,
    )
    algorithm.setup(lambda: _ProofEnv())
    return algorithm


def test_get_policy_chain_ends_in_the_eval_egreedy_module():
    algorithm = _setup_algorithm(eval_eps=0.005)

    tail = list(algorithm.get_policy().module)[-1]
    assert isinstance(tail, _EvalEGreedyModule), (
        f"get_policy() ends in {type(tail).__name__}; a bare argmax chain makes "
        f"eval/return_std identically 0 on a deterministic ALE."
    )
    assert float(algorithm.eval_greedy_module.eps) == pytest.approx(0.005)


def test_explore_policy_still_uses_the_annealed_training_module():
    """The training path must be untouched by this fix."""
    algorithm = _setup_algorithm(eval_eps=0.005)

    tail = list(algorithm.get_explore_policy().module)[-1]
    assert tail is algorithm.greedy_module
    assert not isinstance(tail, _EvalEGreedyModule)


def test_eval_module_is_not_annealed_by_training_steps():
    """step() anneals greedy_module; eval_eps must stay pinned."""
    algorithm = _setup_algorithm(eval_eps=0.005)

    before = float(algorithm.eval_greedy_module.eps)
    for _ in range(1000):
        algorithm.greedy_module.step(1)

    assert float(algorithm.eval_greedy_module.eps) == pytest.approx(before)
