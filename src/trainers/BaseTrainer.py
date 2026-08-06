"""Base trainer and callback infrastructure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms.base import BaseAlgorithm
from src.environments.environment import Environment
from src.environments.factory import env_worker_device
from src.utils.device import resolve_device


class TrainerEvent(Enum):
    ON_TRAIN_START = auto()
    ON_STEP_END = auto()
    ON_TRAIN_END = auto()
    ON_EVAL_START = auto()
    ON_EVAL_END = auto()


@runtime_checkable
class Callback(Protocol):
    def on_train_start(self, state: dict[str, Any]) -> None: ...
    def on_step_end(self, metrics: dict[str, float], step: int) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


def fire_callbacks(
    event: TrainerEvent,
    callbacks: list,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a training event to all callbacks that implement the matching method."""
    method_name = event.name.lower()
    for cb in callbacks:
        method = getattr(cb, method_name, None)
        if callable(method):
            method(*args, **kwargs)


class BaseTrainer(ABC):
    """Base class for all trainers.

    Owns: device resolution, environment creation, eval loop, callbacks,
    and checkpoint orchestration.

    Args:
        cfg: full Hydra config
        algorithm: algorithm instance (already ``__init__``'d, not yet set up)
        environment: environment config wrapper used for training
        eval_environment: optional separate environment used by ``evaluate()``;
            falls back to ``environment`` when ``None``
        callbacks: list of callback objects
    """

    def __init__(
        self,
        cfg: DictConfig,
        algorithm: BaseAlgorithm,
        environment: Environment,
        eval_environment: Environment | None = None,
        callbacks: list | None = None,
    ) -> None:
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.algorithm = algorithm
        self.environment = environment
        self.eval_environment = eval_environment or environment
        self.callbacks = callbacks or []

        self.device = resolve_device(
            self.trainer_cfg.accelerator,
            list(self.trainer_cfg.devices),
        )
        self.algorithm.device = self.device

        #: Device the *environment* (and therefore its transform stack) runs
        #: on.  Differs from ``self.device`` whenever ``num_envs > 1`` puts the
        #: envs in ``ParallelEnv`` workers, which are CPU-only.  ``setup()``
        #: recomputes it from the real ``num_envs``; the accelerator is the
        #: right answer for the single-env case.
        self._env_device: str = str(self.device)

        self._step: int = 0

    def setup(self) -> None:
        """Create environment and set up the algorithm."""
        num_envs = int(self.trainer_cfg.get("num_envs", 1))
        self._env_device = env_worker_device(num_envs, str(self.device))

        def make_env():
            return self.environment.make_env(
                num_envs=num_envs,
                device=str(self.device),
            )

        self.train_env = make_env()
        self.algorithm.setup(make_env)

    def fit(self) -> dict[str, float]:
        """Run the full training loop.

        ``state`` carries the current step so loggers can tell a fresh run from a
        resume (``load_checkpoint`` runs before ``fit``; see ``src/train.py``).

        Returns:
            dict of final training metrics
        """
        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            self.callbacks,
            state={"cfg": self.cfg, "step": self._step},
        )

        metrics = self._training_loop()

        fire_callbacks(
            TrainerEvent.ON_TRAIN_END,
            self.callbacks,
            state={"cfg": self.cfg},
        )
        return metrics

    @abstractmethod
    def _training_loop(self) -> dict[str, float]:
        """Subclass-specific training loop."""

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        """Run evaluation episodes using the greedy policy.

        Creates a fresh single-env for eval (separate from the train env).

        The eval env is built on ``self._env_device`` — the device the
        *training* env runs on — not on the accelerator.  With ``num_envs > 1``
        those differ: the training observations are produced by CPU-side
        transforms inside ``ParallelEnv`` workers, and ``GrayScale`` /
        bilinear-antialias ``Resize`` do not return bit-identical floats on CPU
        and CUDA.  Building the eval env on the accelerator therefore feeds the
        policy observations that differ from the training ones in the last few
        bits — invisible to a DQN network, fatal to MFEC/NEC, whose episodic
        memory is a hash of the embedding (see ``env_worker_device``).  The
        tensordict is moved onto ``self.device`` for the policy call and back
        for the env step, exactly as the collector does.
        """
        eval_env = self.eval_environment.make_env(
            num_envs=1,
            device=self._env_device,
        )
        policy = self.algorithm.get_policy()
        self.algorithm.reset_eval_metrics()

        returns: list[float] = []
        lengths: list[int] = []
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = eval_env.reset()
                episode_return = 0.0
                episode_length = 0
                done = False
                while not done:
                    td = policy(td.to(self.device)).to(self._env_device)
                    td = eval_env.step(td)
                    episode_return += td["next", "reward"].sum().item()
                    episode_length += 1
                    done = (
                        td["next", "done"].any().item()
                        or td["next", "terminated"].any().item()
                    )
                    td = td["next"]
                returns.append(episode_return)
                lengths.append(episode_length)

        eval_env.close()
        t = torch.tensor(returns, dtype=torch.float32)
        metrics = {
            "eval/return_mean": t.mean().item(),
            # Unbiased std needs n >= 2; with num_eval_episodes=1 torch returns
            # NaN, which loggers happily plot as a hole in the chart.
            "eval/return_std": t.std().item() if t.numel() > 1 else 0.0,
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
            # Separates "the policy plays badly" from "the episode ended early":
            # a return that collapses while the length holds is a scoring
            # problem, both collapsing together is a dying-agent problem.
            "eval/episode_length": float(sum(lengths)) / len(lengths),
        }
        metrics.update(self.algorithm.eval_metrics())
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save algorithm state + trainer step."""
        self.algorithm.save_checkpoint(path, step=self._step)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore algorithm state + trainer step."""
        self._step = self.algorithm.load_checkpoint(path)

    def _should_log(self, log_every: int, batch_frames: int) -> bool:
        """Check if we crossed a ``log_every`` boundary this iteration."""
        prev = self._step - batch_frames
        return prev // log_every < self._step // log_every