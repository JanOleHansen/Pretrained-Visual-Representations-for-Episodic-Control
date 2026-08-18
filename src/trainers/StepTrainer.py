"""Step-based trainer using ``SyncDataCollector``.

Each iteration:  collector yields one batch of transitions
                 ->  ``algorithm.step(batch)``
                 ->  accumulate episode statistics
                 ->  fire callbacks if it's a logging step.

The trainer owns the loop, the collector and the callbacks; everything that
affects learning lives in the algorithm.

Metrics emitted on logging boundaries:
  - ``train/episode_reward``, ``train/episode_length``: mean over **every**
    episode that finished since the previous logging boundary — not just those
    inside the one batch that happened to land on the boundary.
  - ``train/episodes_completed``: how many episodes that mean averages over,
    so the sample size behind each point is visible.
  - ``train/q_values``: mean Q-value of the actions actually executed over the
    same interval, excluding optimistic-initialisation sentinels.
  - ``time/collect``, ``time/step``, ``time/speed``: collector wait, in-step
    optimisation time, and frames/second for the iteration that crossed the
    boundary.

Interval accumulation (rather than per-batch sampling) matters because the
batch is far smaller than the logging period: with ``frames_per_batch=1024``
and ``log_every_n_steps=10_000`` a single batch holds only ~1-2 finished
Ms. Pac-Man episodes, so a per-batch mean is a 1-2 sample estimate of a
quantity whose per-episode spread is several hundred points, and it silently
discards ~90% of the episodes actually played.  See ``_IntervalStats``.
"""
from __future__ import annotations

import time

import torch
from tensordict import TensorDict

from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


#: Executed-action Q-values at least this large are treated as optimistic
#: initialisation sentinels rather than real returns, and are left out of
#: ``train/q_values``.
#:
#: Episodic-control policies (``MFECAlgorithm``'s ``QECPolicy``, ``NEC``'s
#: equivalent) substitute ~1e9 for the ``+inf`` estimate of a state-action pair
#: their memory cannot yet evaluate.  Averaging those in makes the metric read
#: ~1e9 during warm-up and buries the real values.  The threshold sits three
#: orders of magnitude below the sentinel and far above any Atari return, so it
#: separates the two cleanly without the trainer needing to import an
#: algorithm-specific constant.
_OPTIMISTIC_Q_THRESHOLD = 1e8


class _IntervalStats:
    """Episode and Q-value statistics accumulated between logging boundaries.

    ``update()`` is called on *every* collector batch; ``flush()`` is called
    only on a logging step, returns the mean over the whole interval, and
    resets.

    A metric is omitted from ``flush()`` entirely when the interval produced no
    samples for it — an interval in which no episode finished emits no
    ``train/episode_reward`` rather than a stale or zero one, leaving a genuine
    gap in the chart instead of a fabricated point.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._reward_sum = 0.0
        self._reward_count = 0
        self._length_sum = 0.0
        self._length_count = 0
        self._episodes = 0
        self._q_sum = 0.0
        self._q_count = 0

    def update(self, batch: TensorDict) -> None:
        """Fold one collector batch into the running totals.

        Each quantity is accumulated only when the producing transform is in
        the env stack: ``RewardSum`` for ``episode_reward``, ``StepCounter``
        for ``step_count``, and a ``QValueActor``-style policy for
        ``action_value`` / ``action``.
        """
        flat = batch.reshape(-1)

        done = flat.get(("next", "done"), default=None)
        if done is not None:
            mask = done.bool()
            if mask.any():
                self._episodes += int(mask.sum().item())

                episode_rewards = flat.get(("next", "episode_reward"), default=None)
                if episode_rewards is not None:
                    finished = episode_rewards[mask].float()
                    self._reward_sum += finished.sum().item()
                    self._reward_count += finished.numel()

                episode_lengths = flat.get(("next", "step_count"), default=None)
                if episode_lengths is not None:
                    finished = episode_lengths[mask].float()
                    self._length_sum += finished.sum().item()
                    self._length_count += finished.numel()

        # Q-value of the action actually executed.  Handles both one-hot
        # encoding (action shape [B, A]) and categorical encoding (action
        # shape [B], integer indices).
        action_value = flat.get("action_value", default=None)
        action = flat.get("action", default=None)
        if action_value is not None and action is not None:
            if action.dim() == action_value.dim():
                chosen = (action_value * action).sum(-1)
            else:
                chosen = action_value.gather(
                    -1, action.long().unsqueeze(-1)
                ).squeeze(-1)
            chosen = chosen.float().reshape(-1)
            real = chosen[
                torch.isfinite(chosen) & (chosen.abs() < _OPTIMISTIC_Q_THRESHOLD)
            ]
            self._q_sum += real.sum().item()
            self._q_count += real.numel()

    def flush(self) -> dict[str, float]:
        """Return interval means and reset. Keys without samples are omitted."""
        out: dict[str, float] = {}
        if self._reward_count:
            out["train/episode_reward"] = self._reward_sum / self._reward_count
        if self._length_count:
            out["train/episode_length"] = self._length_sum / self._length_count
        if self._episodes:
            out["train/episodes_completed"] = float(self._episodes)
        if self._q_count:
            out["train/q_values"] = self._q_sum / self._q_count
        self.reset()
        return out


class StepTrainer(BaseTrainer):
    def setup(self) -> None:
        super().setup()
        self._create_collector()
        self._interval_stats = _IntervalStats()

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

        # Wall-clock start of *this* process's training. On CUDA, reset the peak
        # memory counter so ``sys/gpu_mem_peak_gb`` reports this run's footprint
        # rather than a leftover high watermark from setup or an earlier phase.
        # ``getattr`` because tests drive ``_training_loop`` directly on a stub
        # that bypasses ``setup()`` and never assigns ``self.device``.
        loop_start = time.perf_counter()
        device = getattr(self, "device", None)
        cuda_device = device if getattr(device, "type", None) == "cuda" else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)

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

            # Every batch contributes, so the logged mean covers all episodes
            # played in the interval rather than only those in this batch.
            self._interval_stats.update(batch)

            step_start = time.perf_counter()
            metrics = self.algorithm.step(batch)
            step_time = time.perf_counter() - step_start

            if self._should_log(log_every, batch_frames):
                # Fill in only metrics the algorithm didn't already report.
                # Some algorithms (e.g. NEC/MFEC) use +inf-valued optimistic
                # Q-estimates for under-explored state-actions in their
                # exploration policy and compute a correctly-bounded
                # "train/q_values" themselves in step(); that must win over the
                # value derived here from the collected batch.
                for key, value in self._interval_stats.flush().items():
                    metrics.setdefault(key, value)
                total_time = collect_time + step_time
                metrics["time/collect"] = collect_time
                metrics["time/step"] = step_time
                metrics["time/speed"] = (
                    batch_frames / total_time if total_time > 0 else 0.0
                )
                # Compute-cost metrics for the encoder comparison: cumulative
                # wall-clock (its last value is the run's total training time)
                # and the peak GPU allocation (monotonic; its last value is the
                # run's peak). Both surface directly as W&B run-summary numbers,
                # so the cost table needs no post-processing.
                metrics["time/elapsed_min"] = (time.perf_counter() - loop_start) / 60.0
                if cuda_device is not None:
                    metrics["sys/gpu_mem_peak_gb"] = (
                        torch.cuda.max_memory_allocated(cuda_device) / 1e9
                    )
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics
