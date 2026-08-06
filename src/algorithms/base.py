"""Base algorithm interface.

Every RL algorithm subclasses :class:`BaseAlgorithm`.  The trainer drives the
loop and calls these methods — the algorithm never owns the loop.

Lifecycle:
    1. ``__init__(device)``           — store hyperparameters
    2. ``setup(make_env)``            — build networks, loss, optimiser
    3. Loop: ``step(batch) -> dict``  — called repeatedly by the trainer
    4. ``save_checkpoint`` / ``load_checkpoint``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase


@dataclass
class TrainingState:
    """Snapshot of algorithm state for checkpointing and resuming."""

    step: int
    policy_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    extra: dict[str, Any] | None = field(default=None)


@dataclass
class CollectorConfig:
    """Parameters the algorithm hands to :class:`StepTrainer` to build ``SyncDataCollector``."""

    frames_per_batch: int
    init_random_frames: int = 0
    max_frames_per_traj: int = -1  # -1 = no limit


class BaseAlgorithm(ABC):
    """Abstract base class for all RL algorithms.

    Hyperparameters and design decisions (network, replay buffer, loss,
    optimiser, exploration) live entirely on the algorithm.  The trainer
    only handles the loop, device placement, logging, and callbacks.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device

    @abstractmethod
    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        """Build networks, loss module and optimiser.

        ``make_env`` returns a fresh environment; the algorithm calls it
        if it needs to read ``observation_spec``/``action_spec``.
        """

    @abstractmethod
    def get_policy(self) -> TensorDictModule:
        """Greedy / deterministic policy used for evaluation."""

    @abstractmethod
    def get_explore_policy(self) -> TensorDictModule:
        """Exploration policy used for data collection."""

    @abstractmethod
    def get_collector_config(self) -> CollectorConfig:
        """Tell :class:`StepTrainer` how to configure ``SyncDataCollector``."""

    @abstractmethod
    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one collected batch and return scalar metrics."""

    # ------------------------------------------------------------------
    # Evaluation diagnostics (optional)
    # ------------------------------------------------------------------

    def reset_eval_metrics(self) -> None:
        """Clear whatever :meth:`eval_metrics` accumulates.

        ``BaseTrainer.evaluate`` calls this immediately before the rollout.
        Gradient-based algorithms have nothing to report and inherit the no-op.
        """

    def eval_metrics(self) -> dict[str, float]:
        """Algorithm-side ``eval/*`` diagnostics for the rollout just finished.

        Merged into the trainer's ``eval/*`` dict. Episodic-control algorithms
        use this to report how often the evaluation policy actually found the
        query state in memory — the one number that distinguishes "the memory
        holds a bad policy" from "the memory is never being hit at all".
        """
        return {}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, step: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._get_training_state()
        state.step = step
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> int:
        state: TrainingState = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self._load_training_state(state)
        return state.step

    @abstractmethod
    def _get_training_state(self) -> TrainingState: ...

    @abstractmethod
    def _load_training_state(self, state: TrainingState) -> None: ...
