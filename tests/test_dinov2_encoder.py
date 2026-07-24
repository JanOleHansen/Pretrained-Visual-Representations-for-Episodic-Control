"""Tests for the DINOv2 PVM encoder (src/encoders/dinov2_encoder.py).

The real DINOv2 ViT-S/14 needs both the pretrained .pth and the
facebookresearch/dinov2 architecture code (torch.hub / a local clone), so it
can't run in CI. These tests fall into two tiers:

  * Unit tests (default): monkeypatch torch.hub.load to return a tiny STUB
    backbone with the same (B,3,H,W) -> (B,384) contract. This exercises the
    code we actually wrote -- resize, ImageNet-normalise, embed() shape,
    determinism, device, state()/load_state() round-trip -- with no weights
    and no network.

  * Real-weights test: skipped unless DINOV2_WEIGHTS points at a real
    dinov2_vits14_pretrain.pth. Run locally to confirm the actual checkpoint
    loads and yields (B, 384). Set DINOV2_REPO_DIR too if the box is offline.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from src.encoders.dino_v2_encoder import DINOv2Encoder

EMBED_DIM = 384          # dinov2_vits14
IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Stub backbone: same I/O contract as a real DINOv2 ViT, ~zero cost.
# ---------------------------------------------------------------------------

class _StubDINOv2(nn.Module):
    """(B, 3, H, W) -> (B, 384), deterministic. Stands in for the real ViT."""

    embed_dim = EMBED_DIM

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-1, -2)))   # spatial-mean -> (B, 3) -> (B, 384)


def _save_stub_ckpt(path, seed: int) -> str:
    """Write a stub state_dict to `path`; different seeds -> different weights."""
    torch.manual_seed(seed)
    torch.save(_StubDINOv2().state_dict(), path)
    return str(path)


@pytest.fixture
def stub_hub(monkeypatch):
    """Make torch.hub.load return a fresh stub (architecture only, no download)."""
    def fake_load(*_args, **_kwargs):
        return _StubDINOv2()
    monkeypatch.setattr("torch.hub.load", fake_load)


@pytest.fixture
def encoder(stub_hub, tmp_path):
    ckpt = _save_stub_ckpt(tmp_path / "stub.pth", seed=0)
    return DINOv2Encoder(weights_path=ckpt, model_name="dinov2_vits14",
                         image_size=IMAGE_SIZE, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# 1. state_dim is fixed by the backbone, not by config
# ---------------------------------------------------------------------------

def test_state_dim_matches_backbone(encoder):
    assert encoder.state_dim == EMBED_DIM


def test_image_size_must_be_multiple_of_14(stub_hub, tmp_path):
    ckpt = _save_stub_ckpt(tmp_path / "stub.pth", seed=0)
    with pytest.raises(AssertionError):
        DINOv2Encoder(weights_path=ckpt, image_size=100)   # 100 % 14 != 0


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

def test_state_roundtrip_restores_embeddings(stub_hub, tmp_path):
    obs = torch.rand(4, 3, 84, 84)

    src_ckpt = _save_stub_ckpt(tmp_path / "src.pth", seed=1)
    src = DINOv2Encoder(weights_path=src_ckpt, image_size=IMAGE_SIZE)
    before = src.embed(obs)

    # A different checkpoint -> different embeddings pre-load (sanity).
    dst_ckpt = _save_stub_ckpt(tmp_path / "dst.pth", seed=2)
    dst = DINOv2Encoder(weights_path=dst_ckpt, image_size=IMAGE_SIZE)
    assert not torch.equal(dst.embed(obs), before)

    dst.load_state(src.state())
    assert torch.equal(dst.embed(obs), before)


# ---------------------------------------------------------------------------
# 5. Real weights -- run locally with the actual .pth, skipped otherwise
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("DINOV2_WEIGHTS"),
    reason="set DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth to run",
)
def test_real_dinov2_weights_load_and_embed():
    enc = DINOv2Encoder(
        weights_path=os.environ["DINOV2_WEIGHTS"],
        model_name="dinov2_vits14",
        repo_dir=os.environ.get("DINOV2_REPO_DIR"),   # None -> torch.hub
        image_size=IMAGE_SIZE,
    )
    obs = torch.rand(2, 3, 84, 84)
    out = enc.embed(obs)
    assert out.shape == (2, EMBED_DIM)
    assert torch.equal(enc.embed(obs), out)           # deterministic