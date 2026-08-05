"""Regression tests for interval-averaged training metrics (StepTrainer).

Bug being guarded against
-------------------------
``_batch_metrics`` used to be called *only* inside the ``_should_log`` branch,
and it masked on ``("next", "done")`` within that single collector batch.  With
``frames_per_batch=1024`` and ``log_every_n_steps=10_000`` that threw away ~90%
of the episodes actually played, and the surviving batch held only ~1-2
finished Ms. Pac-Man episodes.  ``train/episode_reward`` was therefore a 1-2
sample estimate of a quantity with several hundred points of per-episode
spread, which is why the logged curve looked like pure noise.

The fix accumulates on *every* batch and emits the interval mean.  These tests
pin that: the logged value must equal the mean over all episodes completed
since the previous logging boundary, not the mean over the final batch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from src.trainers.StepTrainer import (
    _OPTIMISTIC_Q_THRESHOLD,
    _IntervalStats,
    StepTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    episode_rewards: list[float],
    *,
    batch_len: int = 8,
    num_actions: int = 4,
    q_value: float = 1.0,
) -> TensorDict:
    """One collector batch of ``batch_len`` transitions.

    ``episode_rewards`` gives the ``next.episode_reward`` of the transitions
    that are flagged done (episodes completing inside this batch); every other
    transition is not-done and carries a value that must never be averaged in.
    Episode length is set to ``100 + reward`` so the two metrics can be checked
    independently.
    """
    assert len(episode_rewards) <= batch_len

    done = torch.zeros(batch_len, 1, dtype=torch.bool)
    ep_reward = torch.full((batch_len, 1), -999.0)
    step_count = torch.zeros(batch_len, 1, dtype=torch.int64)

    for i, r in enumerate(episode_rewards):
        done[i, 0] = True
        ep_reward[i, 0] = r
        step_count[i, 0] = int(100 + r)

    return TensorDict(
        {
            "action": torch.zeros(batch_len, dtype=torch.int64),
            "action_value": torch.full((batch_len, num_actions), q_value),
            "next": TensorDict(
                {
                    "done": done,
                    "episode_reward": ep_reward,
                    "step_count": step_count,
                },
                batch_size=[batch_len],
            ),
        },
        batch_size=[batch_len],
    )


class _StubAlgorithm:
    """Minimal algorithm: reports nothing, so the trainer's metrics survive."""

    def step(self, batch: TensorDict) -> dict[str, float]:
        return {}


class _RecordingCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        self.calls.append((step, dict(metrics)))


def _make_trainer(batches: list[TensorDict], log_every: int):
    """A StepTrainer wired to a canned batch sequence, bypassing setup()."""
    from omegaconf import OmegaConf

    trainer = object.__new__(StepTrainer)
    trainer.trainer_cfg = OmegaConf.create({"log_every_n_steps": log_every})
    trainer.algorithm = _StubAlgorithm()
    trainer.collector = batches
    trainer._step = 0
    trainer._interval_stats = _IntervalStats()
    callback = _RecordingCallback()
    trainer.callbacks = [callback]
    return trainer, callback


# ---------------------------------------------------------------------------
# _IntervalStats unit behaviour
# ---------------------------------------------------------------------------

def test_interval_stats_averages_every_episode_in_the_interval():
    stats = _IntervalStats()

    all_rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats.update(_make_batch([10.0, 20.0]))
    stats.update(_make_batch([]))            # no episode finished in this batch
    stats.update(_make_batch([30.0]))
    stats.update(_make_batch([40.0, 50.0]))

    out = stats.flush()

    assert out["train/episode_reward"] == pytest.approx(float(np.mean(all_rewards)))
    assert out["train/episodes_completed"] == 5.0
    assert out["train/episode_length"] == pytest.approx(
        float(np.mean([100 + r for r in all_rewards]))
    )


def test_interval_stats_is_not_the_final_batch_mean():
    """The old bug: only the boundary-crossing batch was measured."""
    stats = _IntervalStats()
    stats.update(_make_batch([0.0, 0.0, 0.0, 0.0]))
    stats.update(_make_batch([100.0]))       # the batch the old code would see

    out = stats.flush()

    assert out["train/episode_reward"] == pytest.approx(20.0)   # 100 / 5
    assert out["train/episode_reward"] != pytest.approx(100.0)


def test_interval_stats_emits_nothing_when_no_episode_completed():
    stats = _IntervalStats()
    stats.update(_make_batch([]))
    stats.update(_make_batch([]))

    out = stats.flush()

    # No stale value, no fabricated zero — the key is simply absent.
    assert "train/episode_reward" not in out
    assert "train/episode_length" not in out
    assert "train/episodes_completed" not in out
    # Q-values are still observed on every transition, so they do survive.
    assert out["train/q_values"] == pytest.approx(1.0)


def test_flush_resets_so_intervals_do_not_bleed():
    stats = _IntervalStats()
    stats.update(_make_batch([10.0, 10.0]))
    first = stats.flush()
    stats.update(_make_batch([80.0]))
    second = stats.flush()

    assert first["train/episode_reward"] == pytest.approx(10.0)
    assert first["train/episodes_completed"] == 2.0
    assert second["train/episode_reward"] == pytest.approx(80.0)
    assert second["train/episodes_completed"] == 1.0


def test_optimistic_sentinels_are_excluded_from_q_values():
    """Warm-up 1e9 sentinels must not swamp train/q_values."""
    stats = _IntervalStats()
    stats.update(_make_batch([], q_value=1e9 + 5e5))   # optimistic init
    stats.update(_make_batch([], q_value=2.0))         # a real estimate

    out = stats.flush()

    assert out["train/q_values"] == pytest.approx(2.0)
    assert out["train/q_values"] < _OPTIMISTIC_Q_THRESHOLD


# ---------------------------------------------------------------------------
# End-to-end through _training_loop — guards the "called on every batch" wiring
# ---------------------------------------------------------------------------

def test_training_loop_logs_interval_mean_across_all_batches():
    # 8 transitions per batch, log every 32 frames -> a boundary every 4 batches.
    interval_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    interval_2 = [70.0, 80.0]

    batches = [
        _make_batch([1.0, 2.0]),
        _make_batch([3.0]),
        _make_batch([]),
        _make_batch([4.0, 5.0, 6.0]),     # crosses the first boundary
        _make_batch([70.0]),
        _make_batch([]),
        _make_batch([]),
        _make_batch([80.0]),              # crosses the second boundary
    ]

    trainer, callback = _make_trainer(batches, log_every=32)
    trainer._training_loop()

    assert len(callback.calls) == 2

    step_1, metrics_1 = callback.calls[0]
    assert step_1 == 32
    assert metrics_1["train/episode_reward"] == pytest.approx(
        float(np.mean(interval_1))
    )
    assert metrics_1["train/episodes_completed"] == float(len(interval_1))

    step_2, metrics_2 = callback.calls[1]
    assert step_2 == 64
    assert metrics_2["train/episode_reward"] == pytest.approx(
        float(np.mean(interval_2))
    )
    assert metrics_2["train/episodes_completed"] == float(len(interval_2))


def test_training_loop_lets_the_algorithm_win_on_shared_keys():
    """An algorithm-reported train/q_values must not be clobbered."""

    class _ReportingAlgorithm:
        def step(self, batch: TensorDict) -> dict[str, float]:
            return {"train/q_values": 42.0}

    batches = [_make_batch([1.0]), _make_batch([2.0])]
    trainer, callback = _make_trainer(batches, log_every=16)
    trainer.algorithm = _ReportingAlgorithm()

    trainer._training_loop()

    _, metrics = callback.calls[0]
    assert metrics["train/q_values"] == 42.0
    # ...while the trainer still supplies the episode metrics.
    assert metrics["train/episode_reward"] == pytest.approx(1.5)
