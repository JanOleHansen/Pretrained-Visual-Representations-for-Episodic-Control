"""Environment: thin config wrapper around the gymnasium env factory.

Holds construction kwargs and produces fresh ``TransformedEnv`` instances on
demand.  Never holds a live env itself — the trainer controls env lifecycle
by calling ``make_env()`` when it needs one.
"""
from __future__ import annotations

from torchrl.envs import EnvBase

from src.environments.factory import make_env


class Environment:
    """Wraps environment parameters and produces TorchRL envs.

    Args:
        name: gymnasium env name (e.g. ``"CartPole-v1"``).
        transforms: list of ``_target_``-keyed dicts; each is instantiated as
            a ``torchrl.envs.transforms`` object and composed on top of the
            base env. ``None`` or empty leaves the env un-transformed.
        gym_kwargs: optional extra kwargs forwarded to ``GymEnv`` (e.g.
            ``{"frame_skip": 4, "from_pixels": True}``).
        gym_backend: optional gym backend name (e.g. ``"gymnasium"``).
    """

    def __init__(
        self,
        name: str,
        transforms: list | None = None,
        gym_kwargs: dict | None = None,
        gym_backend: str | None = None,
        **_: object,
    ) -> None:
        self._factory_kwargs: dict = {
            "name": name,
            "transforms": transforms,
            "gym_kwargs": gym_kwargs,
            "gym_backend": gym_backend,
        }

    def make_env(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int | None = None,
    ) -> EnvBase:
        """Build an env.

        Args:
            seed: master seed for this env group; each worker gets its own
                derived stream. ``None`` leaves the env entropy-seeded.
        """
        return make_env(
            **self._factory_kwargs, num_envs=num_envs, device=device, seed=seed
        )
