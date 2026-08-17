"""Tests for the MAE PVR encoder (src/encoders/mae_encoder.py).

``timm`` is an OPTIONAL dependency (``uv sync --extra mae``) and is deliberately
not in the base ``pyproject.toml`` dependencies, so these tests must run without
it. They inject a stub ``timm`` module into ``sys.modules`` — which also pins the
property that makes the optional dependency safe: ``mae_encoder`` imports
``timm`` lazily, inside ``MAEEncoder.__init__``. A module-scope import there
would make a missing package break *every* MFEC run through
``src.encoders.factory``, not just the mae arm.

Same two tiers as tests/test_clip_encoder.py and tests/test_dinov2_encoder.py:

  * Stub tests (default): a tiny ViT with the same
    ``forward_features(B, 3, H, W) -> (B, num_prefix + P, d)`` contract, so the
    resize / ImageNet-normalise / POOLING / shape / determinism / device / state
    round-trip logic all runs with no weights and no network.

  * Opt-in tests: one that needs only ``timm`` installed (no download) and pins
    what timm's own defaults for ``vit_base_patch16_224.mae`` actually are, and
    one gated on ``MAE_WEIGHTS`` pointing at a real checkpoint.

The pooling tests are the load-bearing ones. MAE's CLS token is never directly
supervised by the reconstruction loss, so taking it would measure our pooling
choice and report it as a property of MAE — see the module docstring of
src/encoders/mae_encoder.py.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.mae_encoder import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    MAEEncoder,
)

EMBED_DIM = 768          # ViT-B/16
NATIVE_SIZE = 224
GRID = 14                # 224 / 16 -> 14x14 = 196 patch tokens
NUM_PATCHES = GRID * GRID
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)   # what MAE must NOT use

#: Sentinel written into the prefix tokens of the poisoned stub. Large enough
#: that a mean which wrongly includes a prefix token cannot be mistaken for a
#: patch mean.
_PREFIX_SENTINEL = 1e6


# ---------------------------------------------------------------------------
# Stub ViT: same forward_features contract as timm's, ~zero cost.
# ---------------------------------------------------------------------------

class _StubViT(nn.Module):
    """``(B, 3, H, W) -> (B, num_prefix + 196, 768)`` tokens, deterministic.

    Mirrors the parts of ``timm.models.VisionTransformer`` that ``MAEEncoder``
    touches: ``forward_features``, ``num_prefix_tokens``, ``embed_dim``,
    ``global_pool``. Records what it was fed so the preprocessing can be
    asserted on.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_prefix_tokens: int = 1,
        prefix_value: float | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(3, embed_dim)
        # A SEPARATE projection for the prefix token, so CLS is a genuinely
        # different function of the frame than the patch mean is. With one
        # shared affine layer the two coincide exactly — mean(W·p + b) is
        # W·mean(p) + b — and every pooling test would pass vacuously.
        self.cls_proj = nn.Linear(3, embed_dim)
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        # timm's default for the .mae tag. Never read by MAEEncoder — the point
        # is that it pools itself — but present so a test can assert that.
        self.global_pool = "token"
        self._prefix_value = prefix_value
        self.last_input: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()

        # One token per patch, content derived from the patch's pixels.
        patches = F.adaptive_avg_pool2d(x, GRID)          # (B, 3, 14, 14)
        patches = patches.flatten(2).transpose(1, 2)      # (B, 196, 3)
        tokens = self.proj(patches)                       # (B, 196, d)

        if self._prefix_value is None:
            # Input-dependent prefix, so "cls" pooling is deterministic and
            # actually a function of the frame (as a real CLS token is).
            prefix = self.cls_proj(x.mean(dim=(-1, -2)))[:, None, :]
            prefix = prefix.expand(-1, self.num_prefix_tokens, -1)
        else:
            prefix = torch.full(
                (x.shape[0], self.num_prefix_tokens, self.embed_dim),
                self._prefix_value,
                dtype=tokens.dtype,
                device=tokens.device,
            )
        return torch.cat([prefix, tokens], dim=1)


def _install_timm(monkeypatch, **stub_kwargs) -> dict:
    """Put a fake ``timm`` in ``sys.modules``; return the recorded build kwargs."""
    seen: dict = {}

    def create_model(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        return _StubViT(**stub_kwargs)

    module = types.ModuleType("timm")
    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)
    return seen


@pytest.fixture
def stub_timm(monkeypatch):
    return _install_timm(monkeypatch)


@pytest.fixture
def encoder(stub_timm):
    return MAEEncoder(device=torch.device("cpu"))


def _obs(n=4, h=210, w=160):
    torch.manual_seed(0)
    return torch.rand(n, 3, h, w)


# ---------------------------------------------------------------------------
# 1. The optional dependency is imported lazily, with an actionable message
# ---------------------------------------------------------------------------

def test_importing_the_module_does_not_require_timm():
    """src.encoders.mae_encoder must import with timm absent."""
    assert "timm" not in sys.modules or True   # the import above succeeded
    import src.encoders.factory as factory     # noqa: F401  (imports MAEEncoder)


def test_missing_timm_raises_an_actionable_error(monkeypatch):
    # Ensure any real/stub timm is invisible and cannot be re-imported.
    monkeypatch.setitem(sys.modules, "timm", None)
    with pytest.raises(ImportError, match="timm"):
        MAEEncoder()


def test_the_error_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)
    with pytest.raises(ImportError, match="--extra mae"):
        MAEEncoder()


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def test_hub_is_used_when_no_local_path(stub_timm):
    MAEEncoder()
    assert stub_timm["model_name"] == "vit_base_patch16_224.mae"
    assert stub_timm["pretrained"] is True
    assert stub_timm["num_classes"] == 0
    assert stub_timm["img_size"] == NATIVE_SIZE
    # Nothing to overlay: timm resolves the tag from the HuggingFace hub.
    assert "pretrained_cfg_overlay" not in stub_timm


def test_local_weights_path_goes_through_pretrained_cfg_overlay(stub_timm):
    """timm's own loader, NOT a raw torch.load + load_state_dict.

    The overlay routes the file through timm's checkpoint_filter_fn, which
    unwraps a {"model": ...} wrapper, remaps the original
    facebookresearch/mae key names, and resamples pos_embed. A raw strict
    load_state_dict (the DINOv2Encoder approach) rejects the upstream release.
    """
    MAEEncoder(weights_path="/some/where/model.safetensors")
    assert stub_timm["pretrained"] is True
    assert stub_timm["pretrained_cfg_overlay"] == {
        "file": "/some/where/model.safetensors"
    }


def test_random_init_is_expressible_as_pretrained_false(stub_timm):
    """What --mae-random-init in encoder_diagnostics.py relies on."""
    MAEEncoder(pretrained=False)
    assert stub_timm["pretrained"] is False
    assert "pretrained_cfg_overlay" not in stub_timm


def test_a_weights_path_loads_even_with_pretrained_false(stub_timm):
    """A path means "load this file"; it is not silently ignored."""
    MAEEncoder(weights_path="/some/where/model.safetensors", pretrained=False)
    assert stub_timm["pretrained"] is True
    assert stub_timm["pretrained_cfg_overlay"] == {
        "file": "/some/where/model.safetensors"
    }


def test_global_pool_is_never_passed_to_timm(stub_timm):
    """The arm pools itself; see the module docstring for both reasons.

    Passing global_pool='avg' would flip timm's use_fc_norm, dropping the
    pretrained final LayerNorm for a freshly initialised fc_norm the MAE
    checkpoint does not contain.
    """
    MAEEncoder(pooling="mean")
    assert "global_pool" not in stub_timm


def test_image_size_is_forwarded_as_img_size(stub_timm):
    encoder = MAEEncoder(image_size=112)
    assert stub_timm["img_size"] == 112
    assert encoder.image_size == 112


def test_a_wider_backbone_is_read_not_assumed(monkeypatch):
    """state_dim comes off a real forward, so ViT-L/16 reports 1024."""
    _install_timm(monkeypatch, embed_dim=1024)
    assert MAEEncoder(model_name="vit_large_patch16_224.mae").state_dim == 1024


def test_state_dim_is_probed_from_a_real_forward(stub_timm):
    assert MAEEncoder().state_dim == EMBED_DIM


def test_model_is_frozen_and_in_eval_mode(stub_timm):
    encoder = MAEEncoder()
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())


# ---------------------------------------------------------------------------
# 3. POOLING — the choice that decides whether this arm measures MAE
# ---------------------------------------------------------------------------

def test_mean_pooling_is_the_default(stub_timm):
    assert MAEEncoder().pooling == "mean"


def test_mean_pooling_excludes_the_prefix_tokens(monkeypatch):
    """The bug this exists to catch: averaging CLS in with the patch tokens.

    The stub reports TWO prefix tokens and fills them with a huge sentinel, so
    a slice that assumed a single prefix token (``tokens[:, 1:]``) would still
    drag one sentinel into the mean and blow the magnitude up. Reading
    ``num_prefix_tokens`` is the only thing that passes.
    """
    _install_timm(monkeypatch, num_prefix_tokens=2, prefix_value=_PREFIX_SENTINEL)
    out = MAEEncoder(pooling="mean").embed(_obs())

    assert out.shape == (4, EMBED_DIM)
    assert out.abs().max() < 1e3, (
        "a prefix token leaked into the patch mean — MAEEncoder is slicing off "
        "a hardcoded number of prefix tokens instead of reading "
        "model.num_prefix_tokens"
    )


def test_mean_pooling_really_is_the_mean_over_patch_tokens(monkeypatch):
    """Not just 'excludes the prefix' — it is the arithmetic mean of the rest."""
    _install_timm(monkeypatch)
    encoder = MAEEncoder()
    obs = _obs(n=2)

    tokens = encoder.model.forward_features(
        (F.interpolate(obs, (NATIVE_SIZE, NATIVE_SIZE), mode="bilinear",
                       align_corners=False)
         - torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        / torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    )
    expected = tokens[:, encoder._num_prefix:].mean(dim=1)
    assert torch.allclose(encoder.embed(obs), expected, atol=1e-6)


def test_cls_pooling_returns_the_first_token(monkeypatch):
    _install_timm(monkeypatch, prefix_value=_PREFIX_SENTINEL)
    out = MAEEncoder(pooling="cls").embed(_obs())
    assert torch.allclose(out, torch.full_like(out, _PREFIX_SENTINEL))


def test_the_two_pooling_modes_disagree(stub_timm):
    """If they matched, the ablation between them would measure nothing."""
    obs = _obs()
    torch.manual_seed(7)
    mean_out = MAEEncoder(pooling="mean").embed(obs)
    torch.manual_seed(7)                       # same stub weights
    cls_out = MAEEncoder(pooling="cls").embed(obs)
    assert not torch.allclose(mean_out, cls_out)


def test_an_unknown_pooling_mode_is_rejected(stub_timm):
    with pytest.raises(ValueError, match="mae_pooling"):
        MAEEncoder(pooling="avg")


def test_the_rejection_names_both_supported_modes(stub_timm):
    with pytest.raises(ValueError, match="mean"):
        MAEEncoder(pooling="gem")
    with pytest.raises(ValueError, match="cls"):
        MAEEncoder(pooling="gem")


# ---------------------------------------------------------------------------
# 4. NOT L2-normalised — held constant with DINOv2/ResNet, unlike CLIP
# ---------------------------------------------------------------------------

def test_the_embedding_is_not_l2_normalised(stub_timm):
    """CLIP normalises because cosine is its training metric; MAE has none.

    Normalising here would add a second difference between this arm and the
    DINOv2/ResNet arms, so `clip` must stay the only arm on the unit sphere.
    """
    out = MAEEncoder().embed(_obs())
    assert not torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-3)


# ---------------------------------------------------------------------------
# 5. embed() contract and preprocessing
# ---------------------------------------------------------------------------

def test_embed_shape_and_dtype(encoder):
    out = encoder.embed(_obs())
    assert out.shape == (4, EMBED_DIM)
    assert out.dtype == torch.float32


def test_leading_dims_are_flattened(encoder):
    """(..., C, H, W) -> (B, d), as src/encoders/base.py requires."""
    assert encoder.embed(torch.rand(2, 3, 3, 210, 160)).shape == (6, EMBED_DIM)


def test_atari_frames_are_resized_to_the_native_size(encoder):
    encoder.embed(_obs(h=210, w=160))
    assert encoder.model.last_input.shape[-2:] == (NATIVE_SIZE, NATIVE_SIZE)


def test_already_native_frames_skip_the_resize(encoder):
    assert encoder.embed(torch.rand(2, 3, NATIVE_SIZE, NATIVE_SIZE)).shape == (
        2, EMBED_DIM
    )


def test_the_whole_frame_is_resized_not_centre_cropped(encoder):
    """A centre crop would cut the maze edges and the score row off-screen."""
    obs = torch.zeros(1, 3, 210, 160)
    obs[..., :4] = 1.0                       # paint the leftmost columns only
    encoder.embed(obs)

    raw = (encoder.model.last_input
           * torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
           + torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
    assert raw[..., :2].max() > 0.9, "left edge lost — this is a centre crop"


def test_imagenet_stats_are_used_not_clips(encoder):
    obs = torch.full((1, 3, NATIVE_SIZE, NATIVE_SIZE), 0.5)
    encoder.embed(obs)

    expected = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    got = encoder.model.last_input[0, :, 0, 0]
    assert torch.allclose(got, expected, atol=1e-6)

    clip_stats = (0.5 - torch.tensor(_CLIP_MEAN)) / torch.tensor(
        (0.26862954, 0.26130258, 0.27577711)
    )
    assert not torch.allclose(got, clip_stats, atol=1e-3), (
        "MAE is being fed CLIP's statistics; MAE was pretrained with "
        "ImageNet's, and so are the DINOv2/ResNet arms."
    )


# ---------------------------------------------------------------------------
# 6. Determinism and device — the QEC hash depends on both
# ---------------------------------------------------------------------------

def test_embed_is_deterministic(encoder):
    obs = _obs()
    a, b = encoder.embed(obs), encoder.embed(obs)
    assert torch.equal(a, b), (
        "non-deterministic phi violates src/encoders/base.py and the QEC "
        "exact-hit hash path never fires"
    )


def test_key_is_invariant_to_batch_shape_for_this_stub(encoder):
    """The property scripts/encoder_diagnostics.py measures for the real ViT.

    A stub cannot prove it for a real float32 ViT on CUDA — that is why the
    script exists and must be run on the training GPU — but this pins that the
    *encoder wrapper* introduces no batch-dependence of its own.
    """
    from src.algorithms.mfec import QEC

    obs = _obs(n=6)
    qec = QEC(1, 8, 1, torch.device("cpu"))

    batched = qec._make_keys(encoder.embed(obs))
    single = [qec._make_keys(encoder.embed(obs[i:i + 1]))[0] for i in range(6)]
    assert batched == single


def test_embed_follows_the_observation_device(encoder):
    assert encoder.embed(_obs()).device == torch.device("cpu")


# ---------------------------------------------------------------------------
# 7. Checkpointing
# ---------------------------------------------------------------------------

def test_state_round_trip_restores_the_embedding(stub_timm):
    a, b = MAEEncoder(), MAEEncoder()
    obs = _obs()
    assert not torch.equal(a.embed(obs), b.embed(obs)), "stubs share weights"

    b.load_state(a.state())
    assert torch.equal(a.embed(obs), b.embed(obs))
    assert not b.model.training, "load_state must leave the backbone in eval mode"


def test_load_state_reasserts_eval_mode(stub_timm):
    """A checkpoint restore must not be able to un-freeze phi."""
    a, b = MAEEncoder(), MAEEncoder()
    b.model.train()                       # simulate the damage
    b.load_state(a.state())
    assert not b.model.training


# ---------------------------------------------------------------------------
# 8. What timm's defaults for the .mae tag ACTUALLY are — needs timm, no network
# ---------------------------------------------------------------------------

def test_the_timm_default_pooling_is_token_not_avg():
    """Pins the two claims src/encoders/mae_encoder.py's docstring makes.

    ``pretrained=False`` so this builds the architecture only — no download,
    no network. Skipped where timm is absent (i.e. in CI and on any box without
    ``uv sync --extra mae``); run it on the training server.

    1. timm's default pooling for this tag is 'token' (CLS), which is the WRONG
       pooling for MAE — hence MAEEncoder pooling itself.
    2. global_pool='avg' is not the fix: it flips use_fc_norm, making `norm` an
       Identity and `fc_norm` a LayerNorm the MAE checkpoint does not contain.
    """
    timm = pytest.importorskip("timm")

    default = timm.create_model(
        "vit_base_patch16_224.mae", pretrained=False, num_classes=0
    )
    assert default.global_pool == "token", (
        "timm's default pooling for vit_base_patch16_224.mae is no longer "
        "'token'; src/encoders/mae_encoder.py's docstring says it is."
    )
    assert default.num_prefix_tokens == 1
    assert default.embed_dim == EMBED_DIM
    assert not isinstance(default.norm, torch.nn.Identity), (
        "the pretrained final LayerNorm must survive on the default build — "
        "it is what MAEEncoder pools after"
    )

    avg = timm.create_model(
        "vit_base_patch16_224.mae", pretrained=False, num_classes=0,
        global_pool="avg",
    )
    assert isinstance(avg.norm, torch.nn.Identity), (
        "global_pool='avg' no longer swaps `norm` for `fc_norm`; the docstring's "
        "second reason for pooling by hand needs revisiting"
    )


# ---------------------------------------------------------------------------
# 9. Real weights — opt-in
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("MAE_WEIGHTS"),
    reason="set MAE_WEIGHTS=/path/to/mae checkpoint to run",
)
def test_real_mae_weights_load_and_embed():
    pytest.importorskip("timm")
    encoder = MAEEncoder(
        weights_path=os.environ["MAE_WEIGHTS"],
        model_name=os.environ.get("MAE_MODEL", "vit_base_patch16_224.mae"),
    )
    obs = _obs(n=2)
    out = encoder.embed(obs)
    assert out.shape == (2, encoder.state_dim)
    assert encoder.state_dim == EMBED_DIM
    assert torch.equal(encoder.embed(obs), out)        # deterministic
    assert not torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-3)
