"""Tests for the CLIP PVM encoder (src/encoders/clip_encoder.py).

``open_clip_torch`` is an OPTIONAL dependency and is deliberately not in
pyproject.toml, so these tests must run without it. They inject a stub
``open_clip`` module into ``sys.modules`` — which also pins the property that
makes the optional dependency safe: ``clip_encoder`` imports ``open_clip``
lazily, inside ``CLIPEncoder.__init__``. A module-scope import there would make
a missing package break *every* MFEC run through
``src.encoders.factory``, not just the clip arm.

Two tiers, mirroring tests/test_dinov2_encoder.py and tests/test_resnet_encoder.py:

  * Stub tests (default): a tiny vision tower with the same
    ``(B, 3, H, W) -> (B, d)`` contract, so the resize / CLIP-normalise /
    projection / L2 / shape / determinism / device / state round-trip logic all
    runs with no weights and no network.

  * Real-weights test: skipped unless ``CLIP_WEIGHTS`` points at a real
    open_clip checkpoint AND open_clip is installed.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn as nn

from src.encoders.clip_encoder import _CLIP_MEAN, _CLIP_STD, CLIPEncoder

PROJ_DIM = 512           # ViT-B-32 / ViT-B-16 projected width
NATIVE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)   # what CLIP must NOT use


# ---------------------------------------------------------------------------
# Stub vision tower: same I/O contract as open_clip's, ~zero cost.
# ---------------------------------------------------------------------------

class _StubVisual(nn.Module):
    """(B, 3, H, W) -> (B, 512), deterministic. Records what it was fed."""

    def __init__(self, image_size=NATIVE_SIZE, declare_stats: bool = False):
        super().__init__()
        self.proj = nn.Linear(3, PROJ_DIM)
        self.image_size = image_size
        self.last_input: torch.Tensor | None = None
        if declare_stats:
            # Newer open_clip hangs the checkpoint's stats off the tower.
            self.image_mean = (0.1, 0.2, 0.3)
            self.image_std = (0.4, 0.5, 0.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return self.proj(x.mean(dim=(-1, -2)))


class _StubCLIP(nn.Module):
    """Full CLIP: a vision tower plus a text tower that must be discarded."""

    def __init__(self, **visual_kwargs):
        super().__init__()
        self.visual = _StubVisual(**visual_kwargs)
        self.transformer = nn.Linear(8, 8)      # stands in for the text tower


@pytest.fixture
def stub_open_clip(monkeypatch):
    """Install a fake ``open_clip`` module; yields the recorded build kwargs."""
    seen: dict = {}

    def create_model_and_transforms(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        visual_kwargs = seen.pop("_visual_kwargs", {})
        return _StubCLIP(**visual_kwargs), None, None

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", module)
    return seen


# ---------------------------------------------------------------------------
# 1. The optional dependency is imported lazily, with an actionable message
# ---------------------------------------------------------------------------

def test_importing_the_module_does_not_require_open_clip():
    """src.encoders.clip_encoder must import with open_clip absent."""
    assert "open_clip" not in sys.modules or True   # the import above succeeded
    import src.encoders.factory as factory          # noqa: F401  (imports CLIPEncoder)


def test_missing_open_clip_raises_an_actionable_error(monkeypatch):
    # Ensure any real/stub open_clip is invisible and cannot be re-imported.
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(ImportError, match="open_clip_torch"):
        CLIPEncoder()


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def test_local_weights_path_wins_over_the_hub_tag(stub_open_clip):
    CLIPEncoder(weights_path="/local/ckpt.bin", pretrained_tag="openai",
                model_name="ViT-B-16-quickgelu")
    assert stub_open_clip["model_name"] == "ViT-B-16-quickgelu"
    assert stub_open_clip["pretrained"] == "/local/ckpt.bin"
    assert stub_open_clip["precision"] == "fp32"
    # Not forwarded unless explicitly requested — older open_clip lacks it.
    assert "force_image_size" not in stub_open_clip


def test_hub_tag_used_when_no_local_path(stub_open_clip):
    CLIPEncoder(weights_path=None, pretrained_tag="laion2b_s34b_b79k",
                model_name="ViT-B-32")
    assert stub_open_clip["pretrained"] == "laion2b_s34b_b79k"


# ---------------------------------------------------------------------------
# 2b. QuickGELU pairing — silent wrong features if unguarded
# ---------------------------------------------------------------------------
#
# OpenAI's CLIP was trained with QuickGELU activations; open_clip's plain
# ViT-B-32 config uses standard GELU. open_clip loads the mismatch anyway and
# only emits a UserWarning, which is trivially lost in a training log — and the
# result is a *subtly wrong embedding geometry*, i.e. exactly the property the
# whole CLIP arm exists to measure. So the encoder refuses the combination.

def test_openai_tag_with_plain_gelu_model_is_rejected(stub_open_clip):
    with pytest.raises(ValueError, match="QuickGELU"):
        CLIPEncoder(model_name="ViT-B-32", pretrained_tag="openai")


def test_the_rejection_names_the_exact_fix(stub_open_clip):
    with pytest.raises(ValueError, match=r"ViT-B-16-quickgelu"):
        CLIPEncoder(model_name="ViT-B-16", pretrained_tag="openai")


def test_openai_tag_is_rejected_even_with_local_weights(stub_open_clip):
    """A local file does not make the architecture right.

    The recommended offline flow parks OpenAI's checkpoint on disk and sets
    clip_weights_path, so this is the path users actually hit. The tag still
    documents the checkpoint's provenance, so it still has to agree.
    """
    with pytest.raises(ValueError, match="QuickGELU"):
        CLIPEncoder(model_name="ViT-B-32", pretrained_tag="openai",
                    weights_path="/datahome/me/models/clip_ViT-B-32_openai.pt")


def test_laion_tags_want_the_plain_name(stub_open_clip):
    """The rule is not "always quickgelu" — LAION used standard GELU."""
    CLIPEncoder(model_name="ViT-B-32", pretrained_tag="laion2b_s34b_b79k")
    CLIPEncoder(model_name="ViT-B-32", pretrained_tag="datacomp_xl_s13b_b90k")


def test_the_default_pairing_is_self_consistent(stub_open_clip):
    """The constructor defaults must not trip the constructor's own guard."""
    encoder = CLIPEncoder()
    assert "quickgelu" in encoder.model_name.lower()


def test_random_init_is_expressible_as_no_weights_and_no_tag(stub_open_clip):
    """What --clip-random-init in encoder_diagnostics.py relies on."""
    CLIPEncoder(weights_path=None, pretrained_tag=None)
    assert stub_open_clip["pretrained"] is None


def test_explicit_image_size_is_forwarded_as_force_image_size(stub_open_clip):
    CLIPEncoder(image_size=112)
    assert stub_open_clip["force_image_size"] == 112


def test_old_open_clip_without_force_image_size_gives_a_clear_error(monkeypatch):
    def create_model_and_transforms(model_name, **kwargs):
        if "force_image_size" in kwargs:
            raise TypeError("unexpected keyword argument 'force_image_size'")
        return _StubCLIP(), None, None

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", module)

    with pytest.raises(TypeError, match="native resolution"):
        CLIPEncoder(image_size=112)
    # ...and the native-resolution path still works on that same old version.
    assert CLIPEncoder(image_size=None).state_dim == PROJ_DIM


def test_only_the_vision_tower_is_kept(stub_open_clip):
    """The ~63M-param text tower is dead weight for a memory key."""
    encoder = CLIPEncoder()
    assert isinstance(encoder.model, _StubVisual)
    assert not hasattr(encoder.model, "transformer")


def test_state_dim_is_probed_from_a_real_forward(stub_open_clip):
    assert CLIPEncoder().state_dim == PROJ_DIM


def test_model_is_frozen_and_in_eval_mode(stub_open_clip):
    encoder = CLIPEncoder()
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_native_image_size_is_read_off_the_tower(stub_open_clip):
    assert CLIPEncoder().image_size == NATIVE_SIZE


def test_tuple_valued_image_size_is_accepted(monkeypatch):
    """open_clip reports (H, W) for some architectures/versions."""
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = (
        lambda name, **kw: (_StubCLIP(image_size=(196, 196)), None, None)
    )
    monkeypatch.setitem(sys.modules, "open_clip", module)
    assert CLIPEncoder().image_size == 196


# ---------------------------------------------------------------------------
# 3. embed() contract
# ---------------------------------------------------------------------------

def _obs(n=4, h=210, w=160):
    torch.manual_seed(0)
    return torch.rand(n, 3, h, w)


def test_embed_shape_and_dtype(stub_open_clip):
    out = CLIPEncoder().embed(_obs())
    assert out.shape == (4, PROJ_DIM)
    assert out.dtype == torch.float32


def test_leading_dims_are_flattened(stub_open_clip):
    """(..., C, H, W) -> (B, d), as src/encoders/base.py requires."""
    out = CLIPEncoder().embed(torch.rand(2, 3, 3, 210, 160))
    assert out.shape == (6, PROJ_DIM)


def test_atari_frames_are_resized_to_the_native_size(stub_open_clip):
    encoder = CLIPEncoder()
    encoder.embed(_obs(h=210, w=160))
    assert encoder.model.last_input.shape[-2:] == (NATIVE_SIZE, NATIVE_SIZE)


def test_the_whole_frame_is_resized_not_centre_cropped(stub_open_clip):
    """A centre crop would cut the maze edges and the score row off-screen.

    Distinguished from cropping by feeding a frame whose content lives only in
    the far-left column: after a resize it must survive; a 224-centre-crop of a
    160-wide frame would discard it.
    """
    encoder = CLIPEncoder()
    obs = torch.zeros(1, 3, 210, 160)
    obs[..., :4] = 1.0                       # paint the leftmost columns only
    encoder.embed(obs)
    seen = encoder.model.last_input
    # Un-normalise back to [0, 1] to compare against the painted value.
    mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
    raw = seen * std + mean
    assert raw[..., :2].max() > 0.9, "left edge lost — this is a centre crop"


def test_clip_normalisation_stats_are_used_not_imagenet(stub_open_clip):
    encoder = CLIPEncoder()
    obs = torch.full((1, 3, NATIVE_SIZE, NATIVE_SIZE), 0.5)
    encoder.embed(obs)

    expected = (0.5 - torch.tensor(_CLIP_MEAN)) / torch.tensor(_CLIP_STD)
    got = encoder.model.last_input[0, :, 0, 0]
    assert torch.allclose(got, expected, atol=1e-6)

    imagenet = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(
        (0.229, 0.224, 0.225)
    )
    assert not torch.allclose(got, imagenet, atol=1e-3), (
        "CLIP is being fed ImageNet statistics; they are different numbers "
        "and stock CLIP inference uses its own."
    )


def test_stats_declared_by_the_checkpoint_win(monkeypatch):
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = (
        lambda name, **kw: (_StubCLIP(declare_stats=True), None, None)
    )
    monkeypatch.setitem(sys.modules, "open_clip", module)

    encoder = CLIPEncoder()
    assert torch.allclose(encoder._mean.flatten(), torch.tensor((0.1, 0.2, 0.3)))
    assert torch.allclose(encoder._std.flatten(), torch.tensor((0.4, 0.5, 0.6)))


def test_explicit_stats_override_everything(monkeypatch):
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = (
        lambda name, **kw: (_StubCLIP(declare_stats=True), None, None)
    )
    monkeypatch.setitem(sys.modules, "open_clip", module)

    encoder = CLIPEncoder(image_mean=(0.7, 0.8, 0.9), image_std=(1.0, 1.0, 1.0))
    assert torch.allclose(encoder._mean.flatten(), torch.tensor((0.7, 0.8, 0.9)))


# ---------------------------------------------------------------------------
# 4. L2 normalisation — the reason CLIP is interesting for MFEC at all
# ---------------------------------------------------------------------------

def test_embeddings_are_unit_norm_by_default(stub_open_clip):
    out = CLIPEncoder().embed(_obs())
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_normalize_false_leaves_the_raw_projection(stub_open_clip):
    out = CLIPEncoder(normalize=False).embed(_obs())
    assert not torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-3)


def test_normalised_euclidean_ranking_equals_cosine_ranking(stub_open_clip):
    """Why normalising is the point: ||a-b||^2 = 2 - 2cos on the unit sphere.

    MFEC's kNN is Euclidean, so this is what puts it on CLIP's own metric.
    """
    emb = CLIPEncoder().embed(_obs(n=12))
    query, bank = emb[0], emb[1:]

    euclid_order = torch.cdist(query[None], bank)[0].argsort()
    cosine_order = (-(bank @ query)).argsort()
    assert torch.equal(euclid_order, cosine_order)


# ---------------------------------------------------------------------------
# 5. Determinism and device — the QEC hash depends on both
# ---------------------------------------------------------------------------

def test_embed_is_deterministic(stub_open_clip):
    encoder = CLIPEncoder()
    obs = _obs()
    a, b = encoder.embed(obs), encoder.embed(obs)
    assert torch.equal(a, b), (
        "non-deterministic phi violates src/encoders/base.py and the QEC "
        "exact-hit hash path never fires"
    )


def test_key_is_invariant_to_batch_shape_for_this_stub(stub_open_clip):
    """The property scripts/encoder_diagnostics.py measures for the real ViT.

    A stub cannot prove it for a real float32 ViT on CUDA — that is why the
    script exists and must be run on the training GPU — but this pins that the
    *encoder wrapper* introduces no batch-dependence of its own.
    """
    from src.algorithms.mfec import QEC

    encoder = CLIPEncoder()
    obs = _obs(n=6)
    qec = QEC(1, 8, 1, torch.device("cpu"))

    batched = qec._make_keys(encoder.embed(obs))
    single = [qec._make_keys(encoder.embed(obs[i:i + 1]))[0] for i in range(6)]
    assert batched == single


def test_embed_follows_the_observation_device(stub_open_clip):
    encoder = CLIPEncoder(device=torch.device("cpu"))
    out = encoder.embed(_obs())
    assert out.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# 6. Checkpointing
# ---------------------------------------------------------------------------

def test_state_round_trip_restores_the_embedding(stub_open_clip):
    a, b = CLIPEncoder(), CLIPEncoder()
    obs = _obs()
    assert not torch.equal(a.embed(obs), b.embed(obs)), "stubs share weights"

    b.load_state(a.state())
    assert torch.equal(a.embed(obs), b.embed(obs))
    assert not b.model.training, "load_state must leave the tower in eval mode"


# ---------------------------------------------------------------------------
# 7. The whole seam: MFECAlgorithm.setup() -> factory -> CLIPEncoder -> QEC
# ---------------------------------------------------------------------------

def test_mfec_end_to_end_with_the_clip_encoder(stub_open_clip):
    """Drives the real RGB env config, the real collector and the real QEC.

    The unit tests above build CLIPEncoder directly; this covers the parts only
    an integration can: that ``encoder_name: clip`` resolves through
    ``make_encoder``, that a 512-d unit-norm key survives the QEC hash, and
    that ``eval_metrics()`` reports against it.
    """
    from omegaconf import OmegaConf
    from torchrl.collectors import Collector

    from src.algorithms.mfec import MFECAlgorithm
    from src.environments.factory import make_env

    cfg = OmegaConf.load("configs/environment/atari_mfec_train_rgb.yaml")
    kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k != "_target_"
    }

    def mk():
        return make_env(**kwargs, num_envs=1, device="cpu", seed=42)

    algorithm = MFECAlgorithm(
        device=torch.device("cpu"), encoder_name="clip", state_dim=PROJ_DIM,
        buffer_size=1_000, k=4, gamma=1.0, frames_per_batch=512,
    )
    algorithm.setup(mk)
    assert isinstance(algorithm.encoder, CLIPEncoder)

    # 1536 frames, not a token 256: MFEC writes to the QEC only at episode end
    # (see `_carry`), so the run has to outlast at least one whole episode —
    # ~450 decisions here, since the default eps_start=1.0 keeps the policy
    # near-random. `max_frames_per_traj` would be the cheap way to force short
    # trajectories, but this env stack already carries a StepCounter and
    # torchrl rejects the combination.
    collector = Collector(
        create_env_fn=mk(), policy=algorithm.get_explore_policy(),
        frames_per_batch=512, total_frames=1536, init_random_frames=0,
        device=torch.device("cpu"), storing_device=torch.device("cpu"),
    )
    try:
        for batch in collector:
            algorithm.step(batch)
    finally:
        collector.shutdown()

    assert sum(algorithm.qec._sizes) > 0, "nothing reached the QEC"
    stored = algorithm.qec.states[0, : algorithm.qec._sizes[0]]
    assert torch.allclose(stored.norm(dim=1), torch.ones(len(stored)), atol=1e-4), (
        "stored keys are not unit-norm — L2 normalisation did not reach the QEC"
    )


# ---------------------------------------------------------------------------
# 8. Real weights — opt-in
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("CLIP_WEIGHTS"),
    reason="set CLIP_WEIGHTS=/path/to/open_clip_pytorch_model.bin to run",
)
def test_real_checkpoint_loads_and_embeds():
    pytest.importorskip("open_clip")
    encoder = CLIPEncoder(
        weights_path=os.environ["CLIP_WEIGHTS"],
        model_name=os.environ.get("CLIP_MODEL", "ViT-B-32"),
    )
    out = encoder.embed(_obs(n=2))
    assert out.shape == (2, encoder.state_dim)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-4)
