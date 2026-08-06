"""Regression tests: the MFEC eval policy chain must live on the run's device.

Bug being guarded against
-------------------------
``EGreedyModule`` takes its device from the ``device=`` **kwarg**, not from
``spec``: ``__init__`` registers ``eps`` via
``torch.as_tensor(eps_init, device=device)`` and ``forward`` raises

    RuntimeError: Expected action and e-greedy module to be on the same
    device, but got action.device=cuda:0 and e-greedy device=cpu

when ``action.device != self.eps.device``.  ``MFECAlgorithm.setup()`` passed
``spec`` but not ``device``, so both e-greedy modules got ``eps`` on CPU even
for an ``accelerator=gpu`` run.

Training never noticed: ``StepTrainer`` builds ``Collector(device=self.device)``
with ``get_explore_policy()``, and the collector's ``.to(device)`` moved
``greedy_module.eps`` as a side effect.  ``_policy`` — the *eval* chain holding
``_EvalEGreedyModule`` — is never handed to the collector, so its ``eps`` stayed
on CPU while ``BaseTrainer.evaluate()`` built its env with
``device=str(self.device)``.  The run died at the first eval, ~100k frames in,
after the QEC had already been filled.

Two tiers, because the failure is device-dependent:

  * ``test_greedy_modules_receive_explicit_device`` runs anywhere.  It asserts
    the ``device`` kwarg actually reaches both constructors, which is the
    contract that broke; on a CPU box the buffer-device assertions below are
    tautological and would not catch a regression.
  * The cuda-gated tests reproduce the crash end-to-end.
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms import mfec as mfec_mod
from src.algorithms.mfec import MFECAlgorithm


OBS_SHAPE = (4, 84, 84)
NUM_ACTIONS = 4
STATE_DIM = 8


class _MockAtariEnv:
    """Duck-typed EnvBase: setup() only reads these three attributes.

    Unlike the mock in test_mfec_encoder_refactor.py this one places its specs
    on a chosen device, because that is exactly what BaseTrainer.setup() does
    (``make_env(..., device=str(self.device))``).
    """

    def __init__(self, device: torch.device):
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=OBS_SHAPE,
                           dtype=torch.uint8, device=device),
            device=device,
        )
        self.action_spec = Categorical(n=NUM_ACTIONS, device=device)
        self.batch_size = torch.Size([])


def _make_algorithm(device: torch.device) -> MFECAlgorithm:
    alg = MFECAlgorithm(
        device=device,
        encoder_name="random_projection",
        seed=0,
        obs_key="pixels",
        buffer_size=200,
        k=1,
        state_dim=STATE_DIM,
        gamma=0.9,
        frames_per_batch=10,
    )
    # BaseTrainer assigns `algorithm.device` before calling setup(); the
    # constructor already took it here, so setup() is all that's left.
    alg.setup(lambda: _MockAtariEnv(device))
    return alg


# ---------------------------------------------------------------------------
# 1. Device-agnostic: the kwarg must be threaded through at all
# ---------------------------------------------------------------------------

def test_greedy_modules_receive_explicit_device(monkeypatch):
    """Both e-greedy modules must be constructed with an explicit `device=`.

    Deliberately white-box.  On a CPU-only box every `.device` assertion in
    this file is trivially satisfied, so without this test the regression
    would sail through CI and only reappear on the cluster.
    """
    seen: dict[str, object] = {}

    for name in ("EGreedyModule", "_EvalEGreedyModule"):
        real = getattr(mfec_mod, name)

        def make_spy(real=real, name=name):
            def spy(*args, **kwargs):
                seen[name] = kwargs.get("device", "MISSING")
                return real(*args, **kwargs)
            return spy

        monkeypatch.setattr(mfec_mod, name, make_spy())

    dev = torch.device("cpu")
    alg = _make_algorithm(dev)

    assert seen["EGreedyModule"] == alg._buffer_device
    assert seen["_EvalEGreedyModule"] == alg._buffer_device, (
        "the eval chain is never passed to the Collector, so nothing else "
        "will ever move it onto the run's device"
    )


def test_both_eps_buffers_are_on_the_algorithm_device():
    alg = _make_algorithm(torch.device("cpu"))
    assert alg.greedy_module.eps.device == alg._buffer_device
    assert alg.eval_greedy_module.eps.device == alg._buffer_device


# ---------------------------------------------------------------------------
# 2. cuda-gated: the actual crash, end to end
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reproduces a GPU-only device mismatch"
)


@requires_cuda
def test_eval_policy_runs_on_cuda_without_touching_the_collector():
    """What BaseTrainer.evaluate() does: fresh cuda env, get_policy(), MODE.

    No Collector is created, so nothing has called `.to(cuda)` on any policy —
    which is precisely why the eval chain has to be built on-device.
    """
    dev = torch.device("cuda:0")
    alg = _make_algorithm(dev)

    td = TensorDict(
        {"pixels": torch.zeros(OBS_SHAPE, dtype=torch.uint8, device=dev)},
        batch_size=[],
        device=dev,
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        out = alg.get_policy()(td)

    assert out["action"].device.type == "cuda"


@requires_cuda
def test_explore_policy_runs_on_cuda_without_collector_to():
    """The train chain must not depend on Collector.to() for correctness."""
    dev = torch.device("cuda:0")
    alg = _make_algorithm(dev)

    td = TensorDict(
        {"pixels": torch.zeros(OBS_SHAPE, dtype=torch.uint8, device=dev)},
        batch_size=[],
        device=dev,
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        out = alg.get_explore_policy()(td)

    assert out["action"].device.type == "cuda"
