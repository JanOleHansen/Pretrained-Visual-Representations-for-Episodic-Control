"""Regression tests for environment seeding.

Bug being guarded against
-------------------------
Nothing used to seed the environments. ``src/train.py`` calls
``seed_everything(cfg.trainer.seed)``, but that seeds the *parent* process
only; ``ParallelEnv`` spawns its workers, so each worker's ``torch`` /
``numpy`` / ``random`` RNGs were reseeded from OS entropy, and neither
``StepTrainer._create_collector`` nor ``BaseTrainer.setup`` ever called
``set_seed``.

Why that was fatal rather than cosmetic: with
``repeat_action_probability=0.0`` an ALE reset is fully deterministic (pinned
by ``test_bare_reset_is_deterministic`` below), so **the only** thing that
distinguishes one episode's start state from another's is ``NoopResetEnv``'s
``torch.randint`` draw — which happens inside the worker. Measured before the
fix: two runs after an identical ``seed_everything(42)`` produced different
per-worker reset states. So ``experiment=mfec/mspacman_5seed`` with
``seed: 42,43,44,45,46`` was not five controlled seeds; the environments and
rollouts differed run to run no matter what the sweep said.
"""
from __future__ import annotations

import pytest
import torch

from src.environments.factory import make_env
from src.utils.seeding import derive_seed, seed_everything


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """Leave the process RNGs exactly as found.

    This module reseeds them on purpose, and several tests elsewhere in the
    suite draw from the global streams without seeding first — so without this
    they pass or fail depending on which files ran before them.
    """
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)


ATARI = "ALE/Pong-v5"
GYM_KWARGS = {
    "frame_skip": 4,
    "repeat_action_probability": 0.0,
    "from_pixels": True,
    "pixels_only": True,
    "categorical_action_encoding": True,
}
TRANSFORMS = [
    {"_target_": "torchrl.envs.NoopResetEnv", "noops": 30, "random": True},
    {"_target_": "torchrl.envs.ToTensorImage"},
    {"_target_": "torchrl.envs.GrayScale"},
    {"_target_": "torchrl.envs.Resize", "w": 84, "h": 84},
]


def _atari_or_skip():
    pytest.importorskip("ale_py")
    try:
        env = make_env(name=ATARI, num_envs=1, gym_kwargs=GYM_KWARGS, seed=0)
    except Exception as exc:  # pragma: no cover - ROM/backend not available
        pytest.skip(f"Atari unavailable: {exc}")
    env.close()


def _reset_states(seed, num_envs=2):
    """Per-worker observations right after reset, for a given master seed."""
    seed_everything(1234)  # deliberately constant: the fix must not need it
    env = make_env(
        name=ATARI,
        num_envs=num_envs,
        transforms=TRANSFORMS,
        gym_kwargs=GYM_KWARGS,
        seed=seed,
    )
    try:
        return env.reset()["pixels"].clone()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# derive_seed
# ---------------------------------------------------------------------------

def test_derive_seed_is_stable_across_processes():
    """Must not use hash(): PYTHONHASHSEED is salted per interpreter, and the
    consumer of these seeds is a *spawned* worker."""
    assert derive_seed(42, 0) == derive_seed(42, 0)
    assert 0 <= derive_seed(42, 0) < 2**31 - 1


def test_derive_seed_streams_do_not_collide_like_base_plus_offset():
    """``base + i`` aliases across runs — (44, worker 1) and (45, worker 0)
    both give 45, so two 'independent' seeds replay each other."""
    seeds = {
        (base, worker): derive_seed(base, worker)
        for base in range(42, 47)
        for worker in range(4)
    }
    assert len(set(seeds.values())) == len(seeds)


# ---------------------------------------------------------------------------
# The property the fix exists to create
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_bare_reset_is_deterministic_so_noops_are_the_only_diversity():
    """The premise of the whole fix. If this ever fails, the argument that the
    worker-side ``NoopResetEnv`` draw must be seeded needs revisiting."""
    _atari_or_skip()
    env = make_env(name=ATARI, num_envs=1, gym_kwargs=GYM_KWARGS, seed=7)
    try:
        a = env.reset()["pixels"].clone()
        b = env.reset()["pixels"].clone()
    finally:
        env.close()
    assert torch.equal(a, b)


@pytest.mark.timeout(600)
def test_same_seed_reproduces_worker_start_states():
    _atari_or_skip()
    first = _reset_states(seed=99)
    second = _reset_states(seed=99)
    assert torch.equal(first, second), (
        "identical master seed produced different per-worker start states — "
        "the spawned workers are not being seeded"
    )


@pytest.mark.timeout(600)
def test_different_seed_changes_worker_start_states():
    """Guards the degenerate 'fix' of pinning every worker to one constant."""
    _atari_or_skip()
    assert not torch.equal(_reset_states(seed=99), _reset_states(seed=100))


@pytest.mark.timeout(600)
def test_workers_within_one_env_get_different_streams():
    """Four workers on one shared QEC must not replay the same episode; that
    would buy throughput and no information at all."""
    _atari_or_skip()
    px = _reset_states(seed=99, num_envs=4)
    dupes = [
        (i, j)
        for i in range(4)
        for j in range(i + 1, 4)
        if torch.equal(px[i], px[j])
    ]
    # Noop counts are drawn from [0, 30), so a collision is possible but two
    # independent pairs colliding is not.
    assert len(dupes) <= 1, f"workers share start states: {dupes}"


@pytest.mark.timeout(900)
def test_rollout_is_reproducible_end_to_end():
    """Start states matching is necessary but not sufficient — the emulator
    stream itself has to be seeded too."""
    _atari_or_skip()

    def rollout(seed):
        seed_everything(1234)
        env = make_env(
            name=ATARI,
            num_envs=2,
            transforms=TRANSFORMS,
            gym_kwargs=GYM_KWARGS,
            seed=seed,
        )
        try:
            torch.manual_seed(0)  # fix the *policy* draws, isolating the env
            td = env.rollout(60, break_when_any_done=False)
            return td["action"].clone(), td["next", "pixels"].clone()
        finally:
            env.close()

    a_act, a_px = rollout(5)
    b_act, b_px = rollout(5)
    assert torch.equal(a_act, b_act)
    assert torch.equal(a_px, b_px)


@pytest.mark.timeout(600)
def test_single_env_does_not_reseed_the_parent_process():
    """``BaseTrainer.evaluate()`` builds a fresh num_envs=1 env every eval
    interval. If that reseeded the global RNGs, the trainer's exploration
    stream would reset to the same state on every evaluation and training
    exploration would become periodic."""
    _atari_or_skip()
    seed_everything(1234)
    before = torch.rand(4)

    seed_everything(1234)
    env = make_env(name=ATARI, num_envs=1, gym_kwargs=GYM_KWARGS, seed=555)
    env.close()
    after = torch.rand(4)

    assert torch.equal(before, after)
