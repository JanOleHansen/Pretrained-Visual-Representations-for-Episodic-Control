"""Smoke test: one full training cycle of DQN on CartPole.

Loads the experiment config, applies minimal-frame overrides so the run
finishes in a few seconds, and asserts that ``_train()`` returns a non-empty
metrics dict without raising.

Run with:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


BASE_OVERRIDES = [
    "logger=[]",
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "checkpoint.save_dir=/tmp/hydra_smoke_tests/checkpoints",
    "checkpoint.save_last=false",
    "checkpoint.save_every_n_steps=999999999",
    "hydra.run.dir=/tmp/hydra_smoke_tests",
]


def _dqn_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
    ]


def test_smoke_dqn_cartpole():
    """DQN on CartPole-v1: discrete actions, MLP Q-network, replay buffer."""
    cfg = load_experiment_cfg("dqn/cartpole", _dqn_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _dqn_pong_overrides() -> list[str]:
    # Same shape as the cartpole overrides: 600 frames in 100-frame batches,
    # init_random_frames=100 so we hit the gradient path. Shrinks the 1M
    # replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_dqn_pong():
    """DQN on ALE/Pong-v5: pixel obs, NatureDQN CNN, eval-env split."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("dqn/pong", _dqn_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ddpg_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    # Shrink the 1M replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.replay_buffer.storage.max_size=500",
        "algorithm.exploration_noise.annealing_num_steps=600",
    ]


def test_smoke_ddpg_halfcheetah():
    """DDPG on HalfCheetah-v4: continuous actions, MLP actor/critic, OU noise."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("ddpg/halfcheetah", _ddpg_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _a2c_overrides() -> list[str]:
    # 600 frames in 120-frame rollouts: 5 collections, 6 mini-batches each
    # (mini_batch_size=20). On-policy: no replay buffer, no warm-up.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=120",
        "algorithm.mini_batch_size=20",
    ]


def test_smoke_a2c_halfcheetah():
    """A2C on HalfCheetah-v4: continuous actions, stochastic actor + GAE."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("a2c/halfcheetah", _a2c_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0



#
# MFEC
#

def _mfec_pong_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=500",
        "trainer.log_every_n_steps=100",
        "trainer.num_envs=1",
        "algorithm.frames_per_batch=100",
        "algorithm.annealing_frames=500",
        "algorithm.buffer_size=500",
        "algorithm.k=2",        # fewer neighbors so lookup works with tiny buffer
    ]


def test_smoke_mfec_pong():
    """MFEC on ALE/Pong-v5: random projection, QEC tables, MC returns."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("mfec/pong", _mfec_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/qec_size" in metrics


def test_smoke_mfec_pong_with_periodic_eval():
    """EvalCallback: eval/return_mean (unclipped, full-episode) lands in the
    same metrics dict as train/* when trainer.eval_every_n_steps is set --
    no separate ``src/eval.py`` run required.

    eval_every_n_steps=500 is chosen to land exactly on the final (and only)
    logging boundary of this run (500 frames, 100-frame batches, log every
    100), so the eval fires once and its keys survive into the dict
    ``_train()`` returns.
    """
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "mfec/pong",
        [
            *_mfec_pong_overrides(),
            "trainer.eval_every_n_steps=500",
            "trainer.num_eval_episodes=1",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert "eval/return_mean" in metrics


#
# NEC
#

def _nec_pong_overrides() -> list[str]:
    # 500 frames in 100-frame batches; tiny DND so memory stays bounded.
    # k=2 so kNN lookup works with dnd_capacity=200.
    # n_step=5 so N-step bootstrap exercises the DND lookup path early.
    # init_random_frames=100 so we hit the gradient path in this run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=500",
        "trainer.log_every_n_steps=100",
        "trainer.num_envs=1",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.dnd_capacity=200",
        "algorithm.k=2",
        "algorithm.n_step=5",
        "algorithm.annealing_frames=500",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_nec_pong():
    """NEC on ALE/Pong-v5: trainable CNN encoder, DND tables, N-step returns."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("nec/pong", _nec_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/dnd_size" in metrics


def test_smoke_nec_pong_dinov2_finetune(monkeypatch):
    """NEC with the DINOv2 embedding network, driven from the CLI override.

    `algorithm/embedding_network=dinov2_finetune` is the documented way to run
    this arm, so the smoke test exercises it exactly that way rather than
    constructing the network by hand -- the Hydra group swap, the 4-channel
    Atari observation reaching the ViT's channel adapter, and the two-group
    optimizer all have to survive the real `_train()` path.

    The backbone is a STUB (torch.hub.load monkeypatched, `weights_path=null`),
    same tiering as tests/test_dinov2_encoder.py: the real ViT-S/14 is 22M
    parameters and needs the facebookresearch/dinov2 architecture code, which
    CI has no way to fetch. The real backbone is covered end-to-end by
    tests/test_nec_dinov2_finetune.py's NEC_DINOV2_REAL tier.
    """
    pytest.importorskip("ale_py")
    import torch
    import torch.nn as nn

    class _StubViT(nn.Module):
        embed_dim = 32

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x.mean(dim=(-1, -2)))

    monkeypatch.setattr("torch.hub.load", lambda *a, **k: _StubViT())

    cfg = load_experiment_cfg(
        "nec/pong",
        [
            *_nec_pong_overrides(),
            "algorithm/embedding_network=dinov2_finetune",
            # `???` in the YAML; null is the "keep the hub init" path, which is
            # what the stub wants -- there is no pretrained .pth to point at.
            "algorithm.embedding_network.weights_path=null",
            "algorithm.embedding_network.image_size=28",
            "run.encoder=dinov2",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/dnd_size" in metrics

def test_smoke_nec_pong_clip_finetune(monkeypatch):
    """NEC with the CLIP embedding network, driven from the CLI override.

    `algorithm/embedding_network=clip_finetune` is the documented way to run
    this arm, so the smoke test exercises it exactly that way: the Hydra group
    swap, the 4-channel Atari observation reaching the vision tower's channel
    adapter, and the two-group optimizer all have to survive `_train()`.

    `open_clip_torch` is an OPTIONAL dependency (`uv sync --extra clip`), so a
    stub `open_clip` module is injected rather than importorskip-ing: that way
    this still runs on a machine without the extra, and it pins the lazy-import
    property that keeps the dependency optional. The real ViT-B-32 is covered
    by tests/test_nec_clip_finetune.py's NEC_CLIP_REAL tier.
    """
    pytest.importorskip("ale_py")
    import sys
    import types

    import torch
    import torch.nn as nn

    class _StubVisual(nn.Module):
        def __init__(self):
            super().__init__()
            # kernel == stride, as every open_clip ViT patch-embed conv has.
            self.conv1 = nn.Conv2d(3, 8, kernel_size=32, stride=32, bias=False)
            self.proj = nn.Linear(8, 32)
            self.image_size = (224, 224)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(self.conv1(x).mean(dim=(-1, -2)))

    class _StubCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = _StubVisual()
            self.transformer = nn.Linear(8, 8)   # text tower, must be dropped

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = lambda name, **kw: (_StubCLIP(), None, None)
    monkeypatch.setitem(sys.modules, "open_clip", module)

    cfg = load_experiment_cfg(
        "nec/pong",
        [
            *_nec_pong_overrides(),
            "algorithm/embedding_network=clip_finetune",
            # 64 = 2 x the stub's patch size, so the patch-grid guard passes.
            "algorithm.embedding_network.image_size=64",
            "run.encoder=clip",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/dnd_size" in metrics


def test_smoke_nec_pong_mae_finetune(monkeypatch):
    """NEC with the MAE embedding network, driven from the CLI override.

    `algorithm/embedding_network=mae_finetune` is the documented way to run
    this arm, so the smoke test exercises it exactly that way: the Hydra group
    swap, the 4-channel Atari observation reaching the ViT's channel adapter,
    the token pooling, and the two-group optimizer all have to survive
    `_train()`.

    `timm` is an OPTIONAL dependency (`uv sync --extra mae`), so a stub `timm`
    module is injected rather than importorskip-ing: that way this still runs
    on a machine without the extra, and it pins the lazy-import property that
    keeps the dependency optional. The real ViT-B/16 is covered by
    tests/test_nec_mae_finetune.py's NEC_MAE_REAL tier.
    """
    pytest.importorskip("ale_py")
    import sys
    import types

    import torch
    import torch.nn as nn

    class _StubViT(nn.Module):
        """forward_features -> (B, 1 + P, 32) tokens, as timm's ViT gives."""

        def __init__(self):
            super().__init__()
            self.embed_dim = 32
            self.num_prefix_tokens = 1
            self.patch_embed = nn.Module()
            # kernel == stride, as every timm ViT patch-embed conv has.
            self.patch_embed.proj = nn.Conv2d(3, 32, kernel_size=16, stride=16)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
            return torch.cat([t.mean(dim=1, keepdim=True), t], dim=1)

    module = types.ModuleType("timm")
    module.create_model = lambda name, **kw: _StubViT()
    monkeypatch.setitem(sys.modules, "timm", module)

    cfg = load_experiment_cfg(
        "nec/pong",
        [
            *_nec_pong_overrides(),
            "algorithm/embedding_network=mae_finetune",
            # 32 = 2 x the stub's patch size, so the patch-grid guard passes
            # and the stub stays cheap.
            "algorithm.embedding_network.image_size=32",
            "run.encoder=mae",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/epsilon" in metrics
    assert "train/dnd_size" in metrics
