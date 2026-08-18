"""Base trainer and callback infrastructure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import math

import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms.base import BaseAlgorithm
from src.environments.environment import Environment
from src.environments.factory import env_worker_device
from src.utils.atari_scores import human_random, resolve_game
from src.utils.device import resolve_device
from src.utils.seeding import derive_seed

#: Sub-stream ids for :func:`derive_seed`. Fixed constants — changing one
#: changes which trajectories a given ``trainer.seed`` produces.
_SEED_STREAM_TRAIN = 0
_SEED_STREAM_EVAL = 1


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

        #: Master seed for every RNG stream this run owns. ``seed_everything``
        #: in ``src/train.py`` covers *this* process only; the env workers are
        #: spawned interpreters and the eval env is rebuilt per evaluation, so
        #: both get their own stream derived from this (see ``derive_seed``).
        #: ``.get`` rather than attribute access because trainer configs built
        #: by hand in tests omit it; ``configs/train.yaml`` always sets it.
        self._seed: int = int(self.trainer_cfg.get("seed", 0))

        self._step: int = 0

        #: ``(random, human)`` baseline raw scores for this run's game, or
        #: ``None`` when the game is not one of the 26 Atari 100k titles (or the
        #: config names no game at all, as in unit tests). When set,
        #: ``evaluate()`` logs ``eval/hns`` — the human-normalised score — next
        #: to the raw return, so cross-game aggregation needs no post-hoc
        #: rescaling. Resolved from several config fields because the game token
        #: lives in different places for the per-game (``run.game``) and
        #: game-generic (top-level ``game`` / the ``ALE/<game>-v5`` env name)
        #: experiment layouts. See ``src/utils/atari_scores.py``.
        # ``throw_on_resolution_failure=False``: ``run.game`` is often the Hydra
        # interpolation ``${hydra:runtime.choices.environment}``, which raises
        # outside a live Hydra run (e.g. in tests). Fall through to the next
        # candidate instead of crashing the trainer over a diagnostic metric.
        def _select(path: str):
            return OmegaConf.select(
                cfg, path, default=None, throw_on_resolution_failure=False
            )

        game = resolve_game(
            _select("run.game"),
            _select("game"),
            _select("environment.name"),
            _select("eval_environment.name"),
        )
        self._game: str | None = game
        self._hns_ref: tuple[float, float] | None = (
            human_random(game) if game is not None else None
        )

    def setup(self) -> None:
        """Create environment and set up the algorithm."""
        num_envs = int(self.trainer_cfg.get("num_envs", 1))
        self._env_device = env_worker_device(num_envs, str(self.device))

        def make_env():
            return self.environment.make_env(
                num_envs=num_envs,
                device=str(self.device),
                seed=derive_seed(self._seed, _SEED_STREAM_TRAIN),
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
        # Seeded from the step count, not a constant: the ALE reset is
        # deterministic, so a fixed seed would make every evaluation replay the
        # same episodes and turn eval/return_mean into a single sample. Keyed
        # on a separate stream from the training workers so the eval env never
        # lands on a start state a training worker is already replaying.
        eval_env = self.eval_environment.make_env(
            num_envs=1,
            device=self._env_device,
            seed=derive_seed(self._seed, _SEED_STREAM_EVAL, self._step),
        )
        policy = self.algorithm.get_policy()
        self.algorithm.reset_eval_metrics()

        # Discount for the Monte-Carlo return-to-go used by the retrieval-quality
        # metric below; the algorithm's own gamma (1.0 for MFEC on Atari, 0.99
        # for NEC) is the value its memory approximates. Stub algorithms in
        # tests have none, so fall back to the undiscounted sum.
        gamma = float(getattr(self.algorithm, "gamma", 1.0))

        returns: list[float] = []
        lengths: list[int] = []
        #: Retrieval-quality samples pooled over every eval step of every
        #: episode: the value the memory assigned to the action actually taken,
        #: paired with the realised discounted return-to-go from that state.
        pred_values: list[float] = []
        realised_returns: list[float] = []
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = eval_env.reset()
                episode_return = 0.0
                episode_length = 0
                # Full per-step reward sequence and the memory's value for the
                # taken action (or None when the policy exposes no action-value,
                # e.g. a bare test policy). Kept whole so the return-to-go is
                # computed over every reward even if a value is missing.
                ep_rewards: list[float] = []
                ep_values: list[float | None] = []
                done = False
                while not done:
                    td = policy(td.to(self.device))
                    value = self._taken_action_value(td)
                    td = eval_env.step(td.to(self._env_device))
                    reward = td["next", "reward"].sum().item()
                    episode_return += reward
                    episode_length += 1
                    ep_rewards.append(reward)
                    ep_values.append(value)
                    done = (
                        td["next", "done"].any().item()
                        or td["next", "terminated"].any().item()
                    )
                    td = td["next"]
                returns.append(episode_return)
                lengths.append(episode_length)

                # Discounted return-to-go G_t = r_t + gamma*G_{t+1}, matched to
                # the value the memory reported for the action taken at t.
                g = 0.0
                for reward, value in zip(reversed(ep_rewards), reversed(ep_values)):
                    g = reward + gamma * g
                    if value is not None:
                        pred_values.append(value)
                        realised_returns.append(g)

        eval_env.close()
        t = torch.tensor(returns, dtype=torch.float32)
        metrics = {
            "eval/return_mean": t.mean().item(),
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
            # Separates "the policy plays badly" from "the episode ended early":
            # a return that collapses while the length holds is a scoring
            # problem, both collapsing together is a dying-agent problem.
            "eval/episode_length": float(sum(lengths)) / len(lengths),
            # Sample size behind this point, so return_{mean,min,max} can be read
            # with the right uncertainty — at num_eval_episodes=1 the three
            # order statistics are necessarily one number.
            "eval/num_episodes": float(len(returns)),
        }
        # Unbiased std needs n >= 2. At n=1 it is undefined; omit it rather than
        # log a fabricated 0.0, which would read as a perfectly consistent
        # policy (and leaves a genuine gap in the chart instead).
        if t.numel() > 1:
            metrics["eval/return_std"] = t.std().item()

        # Human-normalised score: the primary cross-game metric. Logged here so
        # every game lands on one axis and the aggregate needs no post-hoc
        # rescaling. Absent when the game is not one of the 26 Atari 100k titles.
        if self._hns_ref is not None:
            random_score, human_score = self._hns_ref
            metrics["eval/hns"] = (
                (metrics["eval/return_mean"] - random_score)
                / (human_score - random_score)
            )

        # kNN retrieval quality: how well the value the memory retrieves for the
        # taken action tracks the return that action actually earned. This is
        # the dependent variable of the encoder comparison — a representation
        # whose neighbourhoods group states of similar value scores higher.
        corr = self._pearson(pred_values, realised_returns)
        if corr is not None:
            metrics["eval/value_return_corr"] = corr
            metrics["eval/value_return_n"] = float(len(pred_values))

        metrics.update(self.algorithm.eval_metrics())
        return metrics

    @staticmethod
    def _taken_action_value(td) -> float | None:
        """Value the policy assigned to the action it actually took, or ``None``.

        Reads the ``QValueActor`` outputs (``action_value`` = per-action values,
        ``action`` = the chosen action) and returns the value of the taken
        action — using the *taken* action rather than the greedy one so an
        epsilon-random eval step still yields a truthful ``(value, return)``
        pair. Handles both categorical (integer index) and one-hot action
        encodings, and returns ``None`` when the policy exposes no action-value
        (a plain test policy) so the caller simply skips the sample.
        """
        action_value = td.get("action_value", default=None)
        action = td.get("action", default=None)
        if action_value is None or action is None:
            return None
        av = action_value.detach().reshape(-1, action_value.shape[-1]).float()
        n_rows, n_actions = av.shape
        act = action.detach()
        if act.numel() == n_rows:                 # categorical index
            idx = act.reshape(n_rows, 1).long()
            chosen = av.gather(-1, idx).squeeze(-1)
        elif act.numel() == n_rows * n_actions:   # one-hot
            chosen = (av * act.reshape(n_rows, n_actions).float()).sum(-1)
        else:
            return None
        return float(chosen.mean().item())

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        """Pearson correlation of paired samples, or ``None`` if undefined.

        Drops non-finite pairs and the optimistic-initialisation sentinels that
        episodic-control memories report (~1e9) for state-actions they cannot
        yet evaluate, then requires at least two points with non-zero variance
        in both series. Pure Python so it carries no NumPy dependency into the
        trainer.
        """
        pairs = [
            (x, y)
            for x, y in zip(xs, ys)
            if math.isfinite(x) and math.isfinite(y) and abs(x) < 1e8
        ]
        n = len(pairs)
        if n < 2:
            return None
        mean_x = sum(p[0] for p in pairs) / n
        mean_y = sum(p[1] for p in pairs) / n
        sxx = sum((p[0] - mean_x) ** 2 for p in pairs)
        syy = sum((p[1] - mean_y) ** 2 for p in pairs)
        sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
        if sxx <= 0.0 or syy <= 0.0:
            return None
        return sxy / math.sqrt(sxx * syy)

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