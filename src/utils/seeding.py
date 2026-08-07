"""Seeding helpers.

``seed_everything`` seeds the *current* process.  That is not enough on its
own: ``ParallelEnv`` uses ``mp_start_method="spawn"``, so every env worker is a
fresh interpreter whose RNGs are seeded from OS entropy unless something seeds
them explicitly.  ``derive_seed`` exists to hand each of those workers (and the
eval env) its own reproducible stream from the single ``trainer.seed``.
"""
import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set the random seed across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base: int, *stream: int) -> int:
    """A stable, independent seed for a named sub-stream of ``base``.

    Every RNG stream in a run (training env worker *i*, the eval env at step
    *s*, ...) is addressed by an integer tuple and hashed to a seed.  Hashing
    rather than the obvious ``base + i`` matters because those offsets collide
    across streams: with five seeds ``42..46`` and four workers, ``base + i``
    hands seed 45 to (44, worker 1) and to (45, worker 0), so two runs that are
    supposed to be independent replay each other's trajectories.

    Deterministic across processes and Python versions — ``hash()`` is not,
    being salted per interpreter, which is exactly the failure this must avoid
    when the consumer is a spawned worker.

    Args:
        base: the run's master seed (``trainer.seed``).
        stream: integer coordinates identifying the sub-stream.

    Returns:
        A seed in ``[0, 2**31 - 1)``, the range gymnasium/ALE accept.
    """
    key = repr((int(base), *(int(s) for s in stream))).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
