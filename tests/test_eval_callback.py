"""Unit tests for EvalCallback and its wiring into build_callbacks.

EvalCallback lets periodic greedy-policy evaluation (using the unclipped,
full-episode ``eval_environment``) land in the same metrics dict as
``train/*`` -- so ``eval/return_mean`` shows up in the normal training logs
(TensorBoard/W&B) instead of requiring a separate ``src/eval.py`` run.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from src.callbacks.eval import EvalCallback
from src.utils.instantiate import build_callbacks


class _MockTrainer:
    def __init__(self):
        self.eval_calls: list[int] = []

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        self.eval_calls.append(num_episodes)
        return {"eval/return_mean": 1.0, "eval/return_std": 0.0}


def test_eval_callback_fires_on_boundary_crossing():
    trainer = _MockTrainer()
    cb = EvalCallback(eval_every_n_steps=1_000, num_episodes=3)
    cb.set_trainer(trainer)

    metrics: dict[str, float] = {"train/episode_reward": -5.0}
    cb.on_step_end(metrics, step=500)   # below first boundary -> no eval
    assert trainer.eval_calls == []
    assert "eval/return_mean" not in metrics

    cb.on_step_end(metrics, step=1_200)  # crossed 1_000 -> eval fires
    assert trainer.eval_calls == [3]
    assert metrics["eval/return_mean"] == 1.0
    # train/* metrics from this step must not be clobbered.
    assert metrics["train/episode_reward"] == -5.0

    cb.on_step_end(metrics, step=1_800)  # still within the same boundary -> no eval
    assert trainer.eval_calls == [3]

    cb.on_step_end(metrics, step=2_100)  # crossed 2_000 -> eval fires again
    assert trainer.eval_calls == [3, 3]


def test_eval_callback_noop_without_trainer():
    cb = EvalCallback(eval_every_n_steps=1_000)
    metrics: dict[str, float] = {}
    cb.on_step_end(metrics, step=5_000)  # no trainer injected -> must not raise
    assert metrics == {}


def test_build_callbacks_includes_eval_callback_when_configured():
    trainer_cfg = OmegaConf.create({
        "total_frames": 10_000,
        "eval_every_n_steps": 2_000,
        "num_eval_episodes": 7,
    })
    checkpoint_cfg = OmegaConf.create({
        "save_dir": "/tmp/does-not-matter",
        "save_every_n_steps": 5_000,
        "save_last": False,
    })
    trainer = _MockTrainer()

    callbacks = build_callbacks(trainer_cfg, checkpoint_cfg, trainer, loggers=[])
    eval_cbs = [cb for cb in callbacks if isinstance(cb, EvalCallback)]
    assert len(eval_cbs) == 1
    assert eval_cbs[0].eval_every_n_steps == 2_000
    assert eval_cbs[0].num_episodes == 7

    # EvalCallback must come before loggers so it can enrich metrics first.
    sentinel_logger = object()
    callbacks_with_logger = build_callbacks(
        trainer_cfg, checkpoint_cfg, trainer, loggers=[sentinel_logger]
    )
    eval_idx = next(i for i, cb in enumerate(callbacks_with_logger) if isinstance(cb, EvalCallback))
    logger_idx = callbacks_with_logger.index(sentinel_logger)
    assert eval_idx < logger_idx


def test_build_callbacks_omits_eval_callback_when_not_configured():
    trainer_cfg = OmegaConf.create({
        "total_frames": 10_000,
        "eval_every_n_steps": None,
    })
    checkpoint_cfg = OmegaConf.create({
        "save_dir": "/tmp/does-not-matter",
        "save_every_n_steps": 5_000,
        "save_last": False,
    })
    trainer = _MockTrainer()

    callbacks = build_callbacks(trainer_cfg, checkpoint_cfg, trainer, loggers=[])
    assert not any(isinstance(cb, EvalCallback) for cb in callbacks)
