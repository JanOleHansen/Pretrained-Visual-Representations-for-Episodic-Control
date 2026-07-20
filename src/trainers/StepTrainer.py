"""Step-based trainer using ``SyncDataCollector``.

Each iteration:  collector yields one batch of transitions
                 ->  ``algorithm.step(batch)``
                 ->  fire callbacks if it's a logging step.

The trainer owns the loop, the collector and the callbacks; everything that
affects learning lives in the algorithm.

Per-iteration metrics emitted on logging boundaries mirror the torchrl SOTA
DQN reference (sota-implementations/dqn/dqn_cartpole.py):
  - ``train/episode_reward``, ``train/episode_length``: mean over episodes
    that completed inside the batch.
  - ``train/q_values``: mean Q-value of the actions actually executed.
  - ``time/collect``, ``time/step``, ``time/speed``: collector wait, in-step
    optimisation time, and frames/second for the iteration.
"""
from __future__ import annotations

import time

from tensordict import TensorDict

from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class StepTrainer(BaseTrainer):
    def setup(self) -> None:
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import Collector

        cc = self.algorithm.get_collector_config()
        self.collector = Collector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=cc.frames_per_batch,
            total_frames=int(self.trainer_cfg.total_frames),
            init_random_frames=cc.init_random_frames,
            max_frames_per_traj=cc.max_frames_per_traj,
            device=self.device,
            storing_device=self.device,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)
        metrics: dict[str, float] = {}

        collector_iter = iter(self.collector)
        while True:
            collect_start = time.perf_counter()
            try:
                batch = next(collector_iter)
            except StopIteration:
                break
            collect_time = time.perf_counter() - collect_start

            batch_frames = batch.numel()
            self._step += batch_frames

            step_start = time.perf_counter()
            metrics = self.algorithm.step(batch)
            step_time = time.perf_counter() - step_start

            if self._should_log(log_every, batch_frames):
                # Fill in only metrics the algorithm didn't already report.
                # Some algorithms (e.g. NEC/MFEC) use +inf-valued optimistic
                # Q-estimates for under-explored state-actions in their
                # exploration policy; `_batch_metrics`'s "train/q_values" reads
                # that raw (sentinel-substituted) action_value straight from
                # the collected batch, so it must never clobber an algorithm's
                # own, correctly-bounded "train/q_values" from step().
                for key, value in _batch_metrics(batch).items():
                    metrics.setdefault(key, value)
                total_time = collect_time + step_time
                metrics["time/collect"] = collect_time
                metrics["time/step"] = step_time
                metrics["time/speed"] = (
                    batch_frames / total_time if total_time > 0 else 0.0
                )
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics


def _batch_metrics(batch: TensorDict) -> dict[str, float]:
    """Per-batch training metrics that mirror the torchrl SOTA DQN reference.

    Each metric is emitted only when the underlying TensorDict key is present:
    ``RewardSum`` for ``episode_reward``, ``StepCounter`` for ``step_count``,
    and a ``QValueActor``-style policy for ``action_value`` / ``action``.
    """
    flat = batch.reshape(-1)
    out: dict[str, float] = {}

    done = flat.get(("next", "done"), default=None)
    if done is not None and done.bool().any():
        mask = done.bool()
        episode_rewards = flat.get(("next", "episode_reward"), default=None)
        if episode_rewards is not None:
            out["train/episode_reward"] = (
                episode_rewards[mask].float().mean().item()
            )
        episode_lengths = flat.get(("next", "step_count"), default=None)
        if episode_lengths is not None:
            out["train/episode_length"] = (
                episode_lengths[mask].float().mean().item()
            )

    # Q-value of the action actually executed.
    # Handles both one-hot encoding (action shape [B, A]) and categorical
    # encoding (action shape [B], integer indices).
    action_value = flat.get("action_value", default=None)
    action = flat.get("action", default=None)
    if action_value is not None and action is not None:
        if action.dim() == action_value.dim():
            out["train/q_values"] = (
                (action_value * action).sum().item() / flat.numel()
            )
        else:
            out["train/q_values"] = (
                action_value.gather(-1, action.long().unsqueeze(-1))
                .mean()
                .item()
            )

    return out
