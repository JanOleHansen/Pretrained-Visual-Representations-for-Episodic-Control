"""Smoke test: one full training cycle of DQN on CartPole.

Loads the experiment config, applies minimal-frame overrides so the run
finishes in a few seconds, and asserts that ``_train()`` returns a non-empty
metrics dict without raising.

Run with:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


BASE_OVERRIDES = [
    "logger=[]",
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "checkpoint.save_dir=/tmp/hydra_smoke_tests/checkpoints",
    "checkpoint.save_last=false",
    "checkpoint.save_every_n_steps=999999999",
    "hydra.run.dir=/tmp/hydra_smoke_tests",
]


def _dqn_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
    ]


def test_smoke_dqn_cartpole():
    """DQN on CartPole-v1: discrete actions, MLP Q-network, replay buffer."""
    cfg = load_experiment_cfg("dqn/cartpole", _dqn_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _dqn_pong_overrides() -> list[str]:
    # Same shape as the cartpole overrides: 600 frames in 100-frame batches,
    # init_random_frames=100 so we hit the gradient path. Shrinks the 1M
    # replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_dqn_pong():
    """DQN on ALE/Pong-v5: pixel obs, NatureDQN CNN, eval-env split."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("dqn/pong", _dqn_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ddpg_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    # Shrink the 1M replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.replay_buffer.storage.max_size=500",
        "algorithm.exploration_noise.annealing_num_steps=600",
    ]


def test_smoke_ddpg_halfcheetah():
    """DDPG on HalfCheetah-v4: continuous actions, MLP actor/critic, OU noise."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("ddpg/halfcheetah", _ddpg_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _a2c_overrides() -> list[str]:
    # 600 frames in 120-frame rollouts: 5 collections, 6 mini-batches each
    # (mini_batch_size=20). On-policy: no replay buffer, no warm-up.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=120",
        "algorithm.mini_batch_size=20",
    ]


def test_smoke_a2c_halfcheetah():
    """A2C on HalfCheetah-v4: continuous actions, stochastic actor + GAE."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("a2c/halfcheetah", _a2c_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0



#
# MFEC
#

def _mfec_pong_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=500",
        "trainer.log_every_n_steps=100",
        "trainer.num_envs=1",
        "algorithm.frames_per_batch=100",
        "algorithm.annealing_frames=500",
        "algorithm.buffer_size=500",
        "algorithm.k=2",        # fewer neighbors so lookup works with tiny buffer
    ]


def test_smoke_mfec_pong():
    """MFEC on ALE/Pong-v5: random projection, QEC tables, MC returns."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("mfec/pong", _mfec_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/qec_size" in metrics


#
# NEC
#

def _nec_pong_overrides() -> list[str]:
    # 500 frames in 100-frame batches; tiny DND so memory stays bounded.
    # k=2 so kNN lookup works with dnd_capacity=200.
    # n_step=5 so N-step bootstrap exercises the DND lookup path early.
    # init_random_frames=100 so we hit the gradient path in this run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=500",
        "trainer.log_every_n_steps=100",
        "trainer.num_envs=1",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.dnd_capacity=200",
        "algorithm.k=2",
        "algorithm.n_step=5",
        "algorithm.annealing_frames=500",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_nec_pong():
    """NEC on ALE/Pong-v5: trainable CNN encoder, DND tables, N-step returns."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("nec/pong", _nec_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/dnd_size" in metrics