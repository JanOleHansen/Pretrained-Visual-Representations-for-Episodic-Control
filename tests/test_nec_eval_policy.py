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


# ---------------------------------------------------------------------------
# The batched-proof-env case: what actually crashed a real run
# ---------------------------------------------------------------------------

class _MockParallelEnv:
    """Mimics ParallelEnv(E, fn): the specs carry the env batch dim.

    `_MockAtariEnv` above is unbatched, which is why it never caught this —
    `setup()` is handed the TRAINING env, and with num_envs>1 its action_spec
    has shape [E] while `BaseTrainer.evaluate()` builds a single env whose
    action is a scalar of shape [].
    """

    def __init__(self, num_envs: int = 8) -> None:
        self.num_envs = num_envs
        self.observation_spec = Composite(
            pixels=Bounded(
                low=0, high=255, shape=(num_envs, *OBS_SHAPE), dtype=torch.uint8
            ),
            shape=torch.Size([num_envs]),
        )
        self.action_spec = Categorical(n=NUM_ACTIONS, shape=torch.Size([num_envs]))
        self.batch_size = torch.Size([num_envs])


def _make_parallel(num_envs: int = 8) -> NECAlgorithm:
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=NatureEmbedding,
        obs_key="pixels",
        embedding_dim=64,
        dnd_capacity=500,
        k=2,
        eval_eps=0.001,
    )
    alg.setup(lambda: _MockParallelEnv(num_envs))
    return alg


def test_eval_policy_accepts_the_single_env_tensordict_evaluate_builds():
    """Regression: ValueError('Action spec shape does not match the action shape').

    setup() sees an [8]-batched training env; evaluate() feeds an unbatched
    tensordict. EGreedyModule only auto-expands an *unbatched* spec, so a
    batched one raises at the first evaluation — 4 minutes into a real run,
    after the first eval_every_n_steps boundary.
    """
    alg = _make_parallel(8)
    td = TensorDict(
        {"pixels": torch.randint(0, 256, OBS_SHAPE, dtype=torch.uint8)},
        batch_size=[],
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        out = alg.get_policy()(td)
    assert out["action"].shape == torch.Size([]), out["action"].shape


def test_explore_policy_still_accepts_the_batched_collector_tensordict():
    """The unbatched spec must remain correct for the OTHER caller too:
    the collector drives num_envs rows at a time."""
    E = 8
    alg = _make_parallel(E)
    td = TensorDict(
        {"pixels": torch.randint(0, 256, (E, *OBS_SHAPE), dtype=torch.uint8)},
        batch_size=[E],
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        out = alg.get_explore_policy()(td)
    assert out["action"].shape == torch.Size([E]), out["action"].shape


# ---------------------------------------------------------------------------
# eval_eps must be large enough to actually decorrelate episodes
# ---------------------------------------------------------------------------

def test_default_eval_eps_makes_a_repeated_episode_essentially_impossible():
    """A 2.25M-step run reported eval/return_min ~= eval/return_max because
    eval_eps was 0.001.

    ALE is deterministic here (repeat_action_probability=0.0) and NoopResetEnv
    does not perturb Ms. Pac-Man's opening, so an episode of length L replays
    pure argmax with probability (1 - eval_eps)^L. At 0.001 with L=600 that is
    55% — more than half of all eval episodes are the SAME trajectory, so
    num_eval_episodes=5 costs 5x for ~1 effective sample and the reported score
    is one fragile deterministic rollout.

    This asserts the *property* (episodes decorrelate) rather than the literal
    constant, so it keeps holding if the value is retuned.
    """
    import inspect

    default = inspect.signature(NECAlgorithm.__init__).parameters["eval_eps"].default
    typical_episode_len = 600            # measured on Ms. Pac-Man at ~250k steps
    p_identical = (1.0 - default) ** typical_episode_len
    assert p_identical < 0.01, (
        f"eval_eps={default} leaves a {p_identical:.1%} chance that an eval "
        f"episode of {typical_episode_len} steps is bit-identical to pure "
        "argmax. Evaluation then reports one deterministic rollout and "
        "eval/return_std collapses toward 0."
    )


def test_eval_deviates_from_greedy_at_about_the_configured_rate():
    """Empirical counterpart: over a realistic episode length the eval policy
    must actually depart from the argmax action, at roughly eval_eps."""
    eps = 0.05
    alg = _make(eps)
    greedy = _make(0.0)
    greedy.dnd = alg.dnd
    greedy.embedding_net = alg.embedding_net

    obs = torch.randint(0, 256, OBS_SHAPE, dtype=torch.uint8)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        base = int(greedy.get_policy()(TensorDict({"pixels": obs}, batch_size=[]))["action"])
        n, L = 0, 2000
        for _ in range(L):
            a = int(alg.get_policy()(TensorDict({"pixels": obs}, batch_size=[]))["action"])
            n += (a != base)

    expected = eps * (1.0 - 1.0 / NUM_ACTIONS)      # a random draw can re-pick greedy
    assert 0.5 * expected < n / L < 2.0 * expected, (
        f"deviation rate {n/L:.4f} over {L} calls is not ~{expected:.4f}; "
        "epsilon is not being applied at the configured rate"
    )
