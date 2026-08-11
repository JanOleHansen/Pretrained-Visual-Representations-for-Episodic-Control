"""``BaseTrainer.evaluate`` must expose the sample size behind every eval point.

Motivation: the NEC and MFEC Ms. Pac-Man runs were compared directly even
though MFEC evaluated with ``num_eval_episodes=1`` (deliberately — see
``configs/algorithm/mfec_atari.yaml``) and NEC with 10. At n=1 the reported
``return_min``, ``return_mean`` and ``return_max`` are the same number by
construction and ``return_std`` was a fabricated ``0.0``, so the MFEC curves
looked like a tight, converged policy rather than a single sample. Nothing in
the logs distinguished the two cases.

So: ``eval/num_episodes`` is always emitted, and ``eval/return_std`` is
*omitted* at n=1 rather than reported as zero — the same "gap in the chart,
not a made-up point" rule ``_IntervalStats.flush`` and
``NECAlgorithm.eval_metrics`` already follow.
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


def _cartpole_trainer(num_eval_episodes: int):
    """Smallest real trainer that can run ``evaluate()``."""
    from hydra.utils import get_class, instantiate
    from omegaconf import OmegaConf

    from src.environments.environment import Environment

    cfg = load_experiment_cfg(
        "dqn/cartpole",
        [
            "logger=[]",
            "trainer.total_frames=200",
            "trainer.log_every_n_steps=100",
            "algorithm.frames_per_batch=100",
            "algorithm.init_random_frames=100",
            f"trainer.num_eval_episodes={num_eval_episodes}",
        ],
    )

    algorithm = instantiate(cfg.algorithm, device=None)
    env_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
        if k != "_target_"
    }
    environment = Environment(**env_kwargs)
    trainer = get_class(cfg.trainer._target_)(
        cfg=cfg, algorithm=algorithm, environment=environment
    )
    trainer.setup()
    return trainer


@pytest.mark.parametrize("n", [1, 3])
def test_num_episodes_is_always_reported(n):
    trainer = _cartpole_trainer(n)
    metrics = trainer.evaluate(num_episodes=n)

    assert metrics["eval/num_episodes"] == float(n), (
        "the sample size behind an eval point must always be recoverable"
    )


def test_std_omitted_at_one_episode_and_present_beyond():
    single = _cartpole_trainer(1).evaluate(num_episodes=1)
    assert "eval/return_std" not in single, (
        "a fabricated 0.0 std at n=1 reads as a perfectly consistent policy"
    )
    # At n=1 the three order statistics are necessarily the same number; that
    # is exactly why the sample size has to be logged alongside them.
    assert (
        single["eval/return_min"]
        == single["eval/return_mean"]
        == single["eval/return_max"]
    )

    multi = _cartpole_trainer(3).evaluate(num_episodes=3)
    assert "eval/return_std" in multi
    assert multi["eval/num_episodes"] == 3.0
