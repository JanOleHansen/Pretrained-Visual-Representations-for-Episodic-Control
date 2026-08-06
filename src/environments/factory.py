"""Environment factory for gymnasium-backed TorchRL envs.

Builds a (possibly vectorised) ``TransformedEnv`` from a small parameter
set and an explicit list of transform descriptors.

Each transform descriptor is a dict with a ``_target_`` key (a dotted path
to a ``torchrl.envs.transforms`` class) plus its constructor kwargs.
Transforms are instantiated fresh per ``make_env()`` call so each env has
independent transform state.
"""
from __future__ import annotations

import importlib
from contextlib import nullcontext
from functools import partial
from typing import Sequence


def env_worker_device(num_envs: int, device: str) -> str:
    """Device the env — and therefore its transform stack — actually runs on.

    A CUDA context cannot survive the fork/spawn into a ``ParallelEnv`` worker,
    so with ``num_envs > 1`` every per-worker transform (``ToTensorImage``,
    ``GrayScale``, ``Resize``, ...) executes on **CPU** no matter what
    ``device`` says; the collector only moves the finished observation to the
    accelerator afterwards.

    Anything that has to reproduce the *same observation values* as the
    collector — chiefly ``BaseTrainer.evaluate``, which builds its own
    ``num_envs=1`` env — must build that env on this device rather than on the
    accelerator.  Bilinear-antialias ``Resize`` and ``GrayScale`` do not return
    bit-identical floats on CPU and CUDA, and for MFEC/NEC that difference is
    not cosmetic: the episodic memory is keyed on the embedding, so a few ULP
    of drift turns every lookup into a miss.  See "Eval must preprocess
    observations on the training env's device" in AGENTS.md.
    """
    return "cpu" if num_envs > 1 else device


def make_env(
    name: str,
    num_envs: int = 1,
    device: str = "cpu",
    transforms: list | None = None,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
    **_: object,
):
    """Build a (possibly vectorised) ``TransformedEnv`` for a gymnasium env.

    Args:
        name: gymnasium env name (e.g. ``"CartPole-v1"``).
        num_envs: number of parallel envs (>1 -> ``ParallelEnv``).
        device: target device string. ``ParallelEnv`` workers always run on
            CPU because CUDA contexts cannot survive ``fork``; the collector
            moves data to ``device`` after collection.
        transforms: list of ``_target_``-keyed dicts to apply on top of the
            base env. ``None`` or empty -> bare base env.
        gym_kwargs: extra kwargs passed straight to ``GymEnv`` (e.g.
            ``{"frame_skip": 4, "from_pixels": True}``).
        gym_backend: optional gym backend name for ``set_gym_backend``
            (e.g. ``"gymnasium"``); if ``None`` torchrl picks the default.
    """
    worker_device = env_worker_device(num_envs, device)

    env_fn = partial(
        _make_gymnasium_env,
        name=name,
        transforms=transforms,
        device=worker_device,
        gym_kwargs=gym_kwargs,
        gym_backend=gym_backend,
    )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv

        return ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    return env_fn()


def _instantiate_transform(cfg: dict):
    """Instantiate a transform from a ``_target_``-keyed dict (no Hydra runtime)."""
    cfg = dict(cfg)  # copy — don't mutate the caller
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _make_gymnasium_env(
    name: str,
    transforms: list | None,
    device: str,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
):
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import Compose

    backend_ctx = nullcontext()
    if gym_backend is not None:
        from torchrl.envs import set_gym_backend
        backend_ctx = set_gym_backend(gym_backend)

    with backend_ctx:
        base_env = GymEnv(name, device=device, **(gym_kwargs or {}))

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))
