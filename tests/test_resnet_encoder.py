"""Tests for the ResNet PVM encoder (src/encoders/resnet_encoder.py).

Two tiers, mirroring tests/test_dinov2_encoder.py:

  * Stub tests: monkeypatch torchvision's get_model to return a tiny backbone
    with the same (B,3,H,W) -> (B,d) contract plus a `fc` attribute, so the
    resize / ImageNet-normalise / shape / determinism / device / state
    round-trip logic runs with no download.

  * Real-architecture tests: build a genuine torchvision resnet18 with
    *random* weights (weights=None, no download) to exercise the thing a stub
    cannot — BatchNorm.  This is the failure mode that separates ResNet from
    DINOv2: in train() mode BN normalises with batch statistics, so the same
    frame embeds differently depending on what else is in the batch and the
    QEC exact-hit path never fires.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from src.encoders.resnet_encoder import ResNetEncoder

EMBED_DIM = 512          # resnet18 / resnet34
IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Stub backbone: same I/O contract as a real ResNet, ~zero cost.
# ---------------------------------------------------------------------------

class _StubResNet(nn.Module):
    """(B, 3, H, W) -> (B, 512), deterministic. Stands in for a real ResNet."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, EMBED_DIM)
        # ResNetEncoder reads fc.in_features, then replaces fc with Identity.
        self.fc = nn.Linear(EMBED_DIM, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.proj(x.mean(dim=(-1, -2))))


def _save_stub_ckpt(path, seed: int) -> str:
    """Write a stub state_dict to `path`; different seeds -> different weights."""
    torch.manual_seed(seed)
    torch.save(_StubResNet().state_dict(), path)
    return str(path)


@pytest.fixture
def stub_tv(monkeypatch):
    """Make torchvision.models.get_model return a fresh stub (no download)."""
    def fake_get_model(*_args, **_kwargs):
        return _StubResNet()
    monkeypatch.setattr("torchvision.models.get_model", fake_get_model)


@pytest.fixture
def encoder(stub_tv, tmp_path):
    ckpt = _save_stub_ckpt(tmp_path / "stub.pth", seed=0)
    return ResNetEncoder(model_name="resnet18", weights_path=ckpt,
                         image_size=IMAGE_SIZE, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# 1. state_dim is fixed by the backbone, not by config
# ---------------------------------------------------------------------------

def test_state_dim_matches_backbone(encoder):
    assert encoder.state_dim == EMBED_DIM


def test_fc_is_stripped(encoder):
    assert isinstance(encoder.model.fc, nn.Identity), (
        "the 1000-way ImageNet head must be removed or embed() returns logits"
    )


# ---------------------------------------------------------------------------
# 2. embed(): shape (leading dims flattened) + on-device output
# ---------------------------------------------------------------------------

def test_embed_shape_flattens_leading_dims(encoder):
    E, T = 2, 3
    obs = torch.rand(E, T, 3, 84, 84)          # RGB, not yet 224 -> resize path
    out = encoder.embed(obs)
    assert out.shape == (E * T, EMBED_DIM)
    assert out.dtype == torch.float32


def test_embed_output_on_obs_device(encoder):
    obs = torch.rand(4, 3, 84, 84)
    assert encoder.embed(obs).device == obs.device


def test_embed_accepts_already_224(encoder):
    obs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)   # no resize needed
    assert encoder.embed(obs).shape == (2, EMBED_DIM)


# ---------------------------------------------------------------------------
# 3. Determinism -- the QEC exact-hit hash relies on it
# ---------------------------------------------------------------------------

def test_embed_is_deterministic(encoder):
    obs = torch.rand(5, 3, 84, 84)
    a = encoder.embed(obs)
    b = encoder.embed(obs)
    assert torch.equal(a, b), (
        "embed() must be bit-identical for identical pixels or the QEC "
        "exact-hit path never fires."
    )


# ---------------------------------------------------------------------------
# 4. Checkpoint round-trip: state() -> load_state() restores embeddings
# ---------------------------------------------------------------------------

def test_state_roundtrip_restores_embeddings(stub_tv, tmp_path):
    obs = torch.rand(4, 3, 84, 84)

    src_ckpt = _save_stub_ckpt(tmp_path / "src.pth", seed=1)
    src = ResNetEncoder(weights_path=src_ckpt, image_size=IMAGE_SIZE)
    before = src.embed(obs)

    # A different checkpoint -> different embeddings pre-load (sanity).
    dst_ckpt = _save_stub_ckpt(tmp_path / "dst.pth", seed=2)
    dst = ResNetEncoder(weights_path=dst_ckpt, image_size=IMAGE_SIZE)
    assert not torch.equal(dst.embed(obs), before)

    dst.load_state(src.state())
    assert torch.equal(dst.embed(obs), before)


# ---------------------------------------------------------------------------
# 5. BatchNorm -- the ResNet-specific failure mode a stub cannot catch
# ---------------------------------------------------------------------------

@pytest.fixture
def real_arch(monkeypatch):
    """Real resnet18 architecture, random weights, no download.

    The original ``get_model`` must be captured *before* the patch is applied:
    calling ``tvm.get_model`` from inside the replacement re-enters the patched
    function and recurses until the stack blows (which is what this fixture did
    until it was caught — every test in this section errored out).
    """
    import torchvision.models as tvm

    original_get_model = tvm.get_model

    def fake_get_model(name, *_args, **_kwargs):
        return original_get_model(name, weights=None)
    monkeypatch.setattr("torchvision.models.get_model", fake_get_model)


@pytest.fixture
def real_encoder(real_arch, tmp_path):
    import torchvision.models as tvm
    torch.manual_seed(0)
    ckpt = tmp_path / "r18.pth"
    torch.save(tvm.resnet18(weights=None).state_dict(), ckpt)
    return ResNetEncoder(model_name="resnet18", weights_path=str(ckpt),
                         image_size=IMAGE_SIZE, device=torch.device("cpu"))


def test_backbone_is_in_eval_mode(real_encoder):
    assert not real_encoder.model.training, (
        "BatchNorm in train() mode normalises with *batch* statistics, so the "
        "same frame embeds differently depending on batch composition."
    )


def test_load_state_reasserts_eval_mode(real_encoder):
    real_encoder.model.train()                  # simulate a caller flipping it
    real_encoder.load_state(real_encoder.state())
    assert not real_encoder.model.training


def test_embed_is_batch_composition_invariant(real_encoder):
    """The property MFEC actually needs: training embeds num_envs rows at a
    time, BaseTrainer.evaluate embeds 1. Both must produce the same key."""
    obs = torch.rand(4, 3, 84, 84)
    batched = real_encoder.embed(obs)
    single = torch.cat([real_encoder.embed(obs[i:i + 1]) for i in range(4)])
    assert torch.allclose(batched, single, atol=1e-5)


def test_running_stats_do_not_drift(real_encoder):
    """BN running_mean/var must not update during embed(), or φ stops being a
    fixed function and every stored QEC key goes stale."""
    bn = real_encoder.model.bn1
    before = bn.running_mean.clone()
    real_encoder.embed(torch.rand(8, 3, 84, 84))
    assert torch.equal(bn.running_mean, before)


# ---------------------------------------------------------------------------
# 6. Real ImageNet weights -- run locally, skipped otherwise
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RESNET_PRETRAINED"),
    reason="set RESNET_PRETRAINED=1 to download/load real ImageNet weights",
)
def test_real_pretrained_weights_load_and_embed():
    enc = ResNetEncoder(model_name="resnet18", weights_path=None,
                        image_size=IMAGE_SIZE)
    obs = torch.rand(2, 3, 84, 84)
    out = enc.embed(obs)
    assert out.shape == (2, EMBED_DIM)
    assert torch.equal(enc.embed(obs), out)           # deterministic
