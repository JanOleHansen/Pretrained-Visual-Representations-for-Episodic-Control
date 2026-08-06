"""Regression tests for MFEC's evaluation-time episodic-memory lookup.

Background — the bug these pin
------------------------------
``BaseTrainer.evaluate()`` used to build its eval env on the *accelerator*
(``str(self.device)``), while training with ``num_envs > 1`` runs every env —
and therefore its whole ``ToTensorImage``/``GrayScale``/``Resize`` stack —
inside CPU-only ``ParallelEnv`` workers.  So on a GPU box the same ALE frame
was preprocessed by CPU kernels at training time and by CUDA kernels at
evaluation time.  Bilinear-antialias resize is not bit-identical across the
two, and MFEC's memory is keyed on ``round(embedding * key_scale)``: a drift
of ~1e-6 per pixel changes *every* hash key.  Evaluation then answered every
Q query with a k-neighbour mean, which is near-constant across actions, so the
argmax became noise and ``eval/return_mean`` collapsed to random play while
``train/episode_reward`` kept climbing.

Two independent guards, one per half of the fix:

1. ``env_worker_device`` + ``evaluate()`` — the eval env is built on the device
   the *training* env runs on, so the observation bytes match.
2. ``QEC._NEAR_EXACT_RTOL`` — a query whose nearest stored key is numerically
   indistinguishable still resolves to that key's stored value instead of
   silently degrading to the mean.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.algorithms.mfec import MFECAlgorithm, QEC
from src.environments.factory import env_worker_device


# ---------------------------------------------------------------------------
# 1. Observation preprocessing must happen on the same device at train and eval
# ---------------------------------------------------------------------------

def test_env_worker_device_reports_parallel_env_placement():
    # ParallelEnv workers cannot hold a CUDA context -> transforms run on CPU,
    # whatever the accelerator is.
    assert env_worker_device(4, "cuda:0") == "cpu"
    assert env_worker_device(2, "cuda:1") == "cpu"
    # A single env is built directly on the accelerator.
    assert env_worker_device(1, "cuda:0") == "cuda:0"
    assert env_worker_device(1, "cpu") == "cpu"


class _RecordingEnvironment:
    """Stands in for ``Environment``; records how the trainer asks for envs.

    Always hands back a plain CPU CartPole so the test needs no GPU and no
    ``ParallelEnv``; only the *requested* ``(num_envs, device)`` is asserted on.
    """

    def __init__(self) -> None:
        from src.environments.environment import Environment

        self._env = Environment(
            name="CartPole-v1",
            gym_kwargs={"categorical_action_encoding": True},
        )
        self.requests: list[tuple[int, str]] = []

    def make_env(self, num_envs: int = 1, device: str = "cpu"):
        self.requests.append((num_envs, str(device)))
        return self._env.make_env(num_envs=1, device="cpu")


def _constant_policy():
    """Always-action-0 policy; the eval loop only needs *some* valid action."""
    import torch.nn as nn
    from tensordict.nn import TensorDictModule

    class _Zero(nn.Module):
        def forward(self, obs):
            return torch.zeros(obs.shape[:-1], dtype=torch.int64, device=obs.device)

    return TensorDictModule(_Zero(), in_keys=["observation"], out_keys=["action"])


class _StubAlgorithm:
    """Minimal object satisfying the slice of BaseAlgorithm ``evaluate()`` uses."""

    device = None

    def __init__(self) -> None:
        self._policy = _constant_policy()
        self.reset_calls = 0

    def setup(self, make_env) -> None:
        make_env()      # mirrors real algorithms reading specs off a proof env

    def get_policy(self):
        return self._policy

    def reset_eval_metrics(self) -> None:
        self.reset_calls += 1

    def eval_metrics(self) -> dict[str, float]:
        return {"eval/stub_metric": 1.0}


def _make_trainer(env: _RecordingEnvironment, num_envs: int):
    from src.trainers.BaseTrainer import BaseTrainer

    class _Trainer(BaseTrainer):
        def _training_loop(self):    # never run here
            return {}

    cfg = OmegaConf.create({
        "trainer": {
            "accelerator": "cpu",
            "devices": [0],
            "num_envs": num_envs,
            "total_frames": 10,
            "log_every_n_steps": 10,
        }
    })
    return _Trainer(cfg=cfg, algorithm=_StubAlgorithm(), environment=env)


def test_setup_records_the_device_the_env_actually_runs_on():
    env = _RecordingEnvironment()
    trainer = _make_trainer(env, num_envs=4)
    trainer.setup()
    assert trainer._env_device == env_worker_device(4, str(trainer.device))


def test_evaluate_builds_eval_env_on_the_env_device_not_the_accelerator():
    env = _RecordingEnvironment()
    trainer = _make_trainer(env, num_envs=1)
    trainer.setup()

    # Force the two devices apart while keeping both real and CPU-only, so the
    # assertion below distinguishes "eval env follows the accelerator" (the
    # bug) from "eval env follows the training env" (the fix) without a GPU.
    trainer.device = torch.device("cpu", 0)     # str() -> "cpu:0"
    trainer._env_device = "cpu"

    metrics = trainer.evaluate(num_episodes=1)

    eval_num_envs, eval_device = env.requests[-1]
    assert eval_num_envs == 1
    assert eval_device == "cpu", (
        "evaluate() must build the eval env on the training env's device; "
        f"got {eval_device!r}"
    )
    # Episode length is reported so a collapsed return can be told apart from
    # a truncated episode, and algorithm-side eval metrics are merged in.
    assert "eval/episode_length" in metrics
    assert metrics["eval/stub_metric"] == 1.0
    assert trainer.algorithm.reset_calls == 1


# ---------------------------------------------------------------------------
# 2. A numerically indistinguishable query must not degrade to a kNN mean
# ---------------------------------------------------------------------------

def _filled_qec(k: int = 3, n: int = 40, d: int = 8) -> tuple[QEC, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    qec = QEC(num_actions=1, capacity=100, k=k, device=torch.device("cpu"))
    states = torch.randn(n, d)
    # Values are far apart so a k-neighbour mean can never coincide with the
    # stored value by luck.
    values = torch.arange(n, dtype=torch.float64) * 100.0
    qec.add_batch(0, states, values)
    return qec, states, values


def test_exact_query_hits_the_hash_path():
    qec, states, values = _filled_qec()
    q = states[7:8]
    est = qec.estimate_all(q)
    assert est[0, 0].item() == values[7].item()

    queries, exact, near = qec.lookup_stats()
    assert (queries, exact, near) == (1, 1, 0)


def test_near_exact_query_survives_preprocessing_drift():
    """A drift far too small to be a different state must not lose the value.

    This is the CPU-vs-CUDA preprocessing case: the hash key is gone, but the
    stored state is still the nearest neighbour at a distance ~1e-6, and the
    correct answer is its value — not the mean of it and k-1 unrelated states.

    Every stored state is checked, not one, because the rescue has to hold for
    all of them: ``torch.cdist`` takes the ``x²+y²-2xy`` shortcut and its
    reported distance for a near-identical pair is dominated by cancellation
    error (~4e-4 for a true 2.8e-6 here), so a test that trusts ``cdist``
    passes on some indices and fails on others.
    """
    qec, states, values = _filled_qec()

    keys_intact = 0
    for i in range(len(states)):
        drifted = states[i:i + 1] + 1e-6
        if qec._make_keys(drifted) == qec._make_keys(states[i:i + 1]):
            keys_intact += 1
        est = qec.estimate_all(drifted)[0, 0].item()
        assert est == values[i].item(), f"state {i} degraded to a kNN mean ({est})"

    # `round(state * key_scale)` only flips when a coordinate crosses a .5
    # boundary, so a fraction of keys survive any given drift.  Assert most do
    # not, otherwise this test would pass without exercising the rescue path.
    assert keys_intact < len(states) // 2

    queries, exact, near = qec.lookup_stats()
    assert queries == len(states)
    assert exact == keys_intact
    assert exact + near == len(states), "some drifted state fell through to kNN"


def test_genuinely_different_state_still_uses_the_knn_mean():
    """The relative tolerance must not swallow real neighbours."""
    qec, states, values = _filled_qec(k=3)
    novel = torch.randn(1, states.shape[1]) * 5.0

    est = qec.estimate_all(novel).item()
    assert est not in set(values.tolist()), "novel state was mistaken for a stored one"

    queries, exact, near = qec.lookup_stats()
    assert (queries, exact, near) == (1, 0, 0)


# ---------------------------------------------------------------------------
# 3. The hit rate is reported, so a silent lookup failure is visible
# ---------------------------------------------------------------------------

def test_eval_metrics_report_the_memory_hit_rate():
    alg = MFECAlgorithm(device=torch.device("cpu"), k=3, state_dim=8, buffer_size=100)
    alg.qec, states, _ = _filled_qec(k=3)

    alg.reset_eval_metrics()
    assert alg.eval_metrics() == {}, "no queries yet -> no fabricated 0.0"

    alg.qec.estimate_all(states[3:4])            # exact hit
    alg.qec.estimate_all(states[4:5] + 1e-6)     # near-exact hit
    alg.qec.estimate_all(torch.randn(1, 8) * 5)  # genuine miss

    metrics = alg.eval_metrics()
    assert metrics["eval/exact_hit_rate"] == 1 / 3
    assert metrics["eval/memory_hit_rate"] == 2 / 3

    alg.reset_eval_metrics()
    assert alg.eval_metrics() == {}
