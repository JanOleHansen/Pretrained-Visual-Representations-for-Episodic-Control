from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.trainers import BaseTrainer


class EvalCallback:
    """Runs greedy-policy evaluation at fixed step intervals and merges the
    result into the same metrics dict the training loggers see.

    Uses ``trainer.eval_environment`` (unclipped, full-episode variant, e.g.
    ``pong_eval.yaml``) rather than the training environment, so
    ``eval/return_mean`` is directly comparable to paper-reported scores --
    see ``eval_environment`` in ``BaseTrainer``. Falls back to the training
    environment if no dedicated eval environment was configured.

    Must be added to the callback list *before* the logger callbacks (see
    ``build_callbacks``) so the loggers see the merged ``eval/*`` keys.

    Args:
        eval_every_n_steps: run an evaluation every this many environment steps
        num_episodes: number of greedy episodes to average per evaluation
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        num_episodes: int = 5,
    ) -> None:
        self.eval_every_n_steps = eval_every_n_steps
        self.num_episodes = num_episodes
        self._trainer: BaseTrainer | None = None
        self._last_eval_step: int = 0

    def set_trainer(self, trainer: BaseTrainer) -> None:
        """Inject the trainer instance (called by build_callbacks)."""
        self._trainer = trainer

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._trainer is None:
            return
        # Check if we've crossed an eval boundary since the last eval.
        if step // self.eval_every_n_steps > self._last_eval_step // self.eval_every_n_steps:
            metrics.update(self._trainer.evaluate(num_episodes=self.num_episodes))
            self._last_eval_step = step
