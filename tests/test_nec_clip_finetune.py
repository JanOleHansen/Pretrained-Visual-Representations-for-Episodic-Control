"""Tests for NEC's finetunable CLIP embedding network.

`src/networks.py::CLIPEmbedding`, selected by
`algorithm/embedding_network=clip_finetune`. The trainable counterpart to
MFEC's frozen `src/encoders/clip_encoder.py::CLIPEncoder` (covered by
tests/test_clip_encoder.py).

``open_clip_torch`` is an OPTIONAL dependency, so the default tier injects a
stub ``open_clip`` module into ``sys.modules`` — which also pins the property
that makes the dependency optional at all: ``CLIPEmbedding`` imports
``open_clip`` lazily, inside ``__init__``. A module-scope import in
``src/networks.py`` would break *every* algorithm in the repo, since
``src/algorithms/nec.py`` and the DQN/DDPG/A2C configs all import that module.

Two tiers, mirroring tests/test_nec_dinov2_finetune.py:

  * Stub tests (default): a tiny vision tower with the same
    ``(B, 3, H, W) -> (B, 512)`` contract. Covers the channel adapter, the
    resize / CLIP-normalise pipeline, the QuickGELU guard, param groups,
    finetuning through NEC's real ``step()``, and checkpoint round-trip.

  * Real-architecture tests: skipped unless NEC_CLIP_REAL=1 *and* open_clip is
    importable. These build the genuine ``ViT-B-32-quickgelu`` (87.8M-param
    vision tower) with ``pretrained=None``, so they need no network and no
    checkpoint. They pin what a stub structurally cannot: that the text tower
    is really dropped, that the patch-grid guard matches open_clip's actual
    behaviour, that CLIP's stats come off the real checkpoint metadata, and
    that gradients reach the tower's own transformer blocks.

Deliberately NOT covered: whether NEC scores better with CLIP than with
`nature` or `dinov2_finetune`. That is the experiment.
"""
from __future__ import annotations

import functools
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.nec import NECAlgorithm
from src.networks import _CLIP_MEAN, _CLIP_STD, CLIPEmbedding


OBS_SHAPE = (4, 84, 84)          # Atari: 4 stacked grayscale frames
EMBEDDING_DIM = 16
PROJ_DIM = 512                   # ViT-B-32 / ViT-B-16 projected width
NATIVE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)    # what CLIP must NOT use
_IMAGENET_STD = (0.229, 0.224, 0.225)

REAL_MODEL = "ViT-B-32-quickgelu"

real_only = pytest.mark.skipif(
    os.environ.get("NEC_CLIP_REAL") != "1",
    reason="set NEC_CLIP_REAL=1 (and have open_clip_torch installed) to build "
           "the genuine CLIP ViT-B-32",
)


# ---------------------------------------------------------------------------
# Stub open_clip
# ---------------------------------------------------------------------------

class _StubVisual(nn.Module):
    """(B, 3, H, W) -> (B, 512). Records what it was fed.

    ``conv1`` is a real patch-embedding conv and is genuinely used by
    ``forward``, not just hung on the module for the patch-grid guard to
    read. A decorative conv would be the only parameter in the tower that
    never receives a gradient, which would make the "gradients reach the
    backbone" test fail for a reason that exists nowhere in open_clip.
    """

    def __init__(self, image_size=NATIVE_SIZE, with_conv1: bool = True,
                 patch: int = 32, batchnorm: bool = False,
                 declared_stats: tuple | None = "clip"):
        super().__init__()
        self.proj = nn.Linear(8 if with_conv1 else 3, PROJ_DIM)
        # open_clip 3.3.0 reports a (H, W) tuple even for square inputs.
        self.image_size = (image_size, image_size)
        self.last_input: torch.Tensor | None = None
        if with_conv1:
            # Patch-embedding conv: kernel == stride == patch, as every
            # open_clip ViT tower has. The guard reads `.stride` off this.
            self.conv1 = nn.Conv2d(3, 8, kernel_size=patch, stride=patch,
                                   bias=False)
        if batchnorm:
            self.bn = nn.BatchNorm2d(3)
        # open_clip >= ~2.24 hangs the checkpoint's own stats off the tower.
        if declared_stats == "clip":
            self.image_mean, self.image_std = _CLIP_MEAN, _CLIP_STD
        elif declared_stats is not None:
            self.image_mean, self.image_std = declared_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        if hasattr(self, "bn"):
            x = self.bn(x)
        feat = self.conv1(x) if hasattr(self, "conv1") else x
        return self.proj(feat.mean(dim=(-1, -2)))


class _StubCLIP(nn.Module):
    """Full CLIP: a vision tower plus a text tower that must be discarded."""

    def __init__(self, **visual_kwargs):
        super().__init__()
        self.visual = _StubVisual(**visual_kwargs)
        self.transformer = nn.Linear(64, 64)      # stands in for the text tower


@pytest.fixture
def stub_open_clip(monkeypatch):
    """Install a fake ``open_clip``; yields the recorded build kwargs."""
    seen: dict = {}
    visual_kwargs: dict = {}

    def create_model_and_transforms(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        return _StubCLIP(**visual_kwargs), None, None

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", module)
    seen["_visual_kwargs"] = visual_kwargs        # test-writable
    return seen


def _factory(**kwargs):
    """`CLIPEmbedding` pre-bound like a Hydra `_partial_` would."""
    return functools.partial(CLIPEmbedding, **kwargs)


def _real(**kwargs) -> CLIPEmbedding:
    kwargs.setdefault("model_name", REAL_MODEL)
    kwargs.setdefault("pretrained_tag", None)     # random init, no network
    kwargs.setdefault("weights_path", None)
    return CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, **kwargs)


class _MockAtariEnv:
    """Duck-typed EnvBase: setup() only reads specs, never resets or steps."""

    def __init__(self, obs_shape=OBS_SHAPE, num_actions=2):
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=obs_shape, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=num_actions)
        self.batch_size = torch.Size([])


def _make_nec(embedding_network, *, num_actions=2, T=8, lr=1e-2):
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=embedding_network,
        obs_key="pixels",
        embedding_dim=EMBEDDING_DIM,
        dnd_capacity=64,
        k=2,
        n_step=2,
        lr=lr,
        batch_size=4,
        frames_per_batch=T,
        init_random_frames=0,
        num_updates=2,
        annealing_frames=100,
    )
    alg.setup(lambda: _MockAtariEnv(OBS_SHAPE, num_actions))
    return alg


def _episode_batch(num_actions=2, T=8):
    """One env, two complete episodes. Round-robin actions so both DND tables
    clear the k=2 sparsity gate and no assertion below is vacuous."""
    dones = torch.zeros(1, T, dtype=torch.bool)
    dones[0, T // 2 - 1] = True
    dones[0, T - 1] = True
    return TensorDict(
        {
            "pixels": torch.rand(1, T, *OBS_SHAPE),
            "action": (torch.arange(T) % num_actions).reshape(1, T),
            "next": TensorDict(
                {
                    "pixels":     torch.rand(1, T, *OBS_SHAPE),
                    "reward":     torch.ones(1, T, 1),
                    "done":       dones.unsqueeze(-1),
                    "terminated": dones.unsqueeze(-1),
                },
                batch_size=[1, T],
            ),
        },
        batch_size=[1, T],
    )


# ---------------------------------------------------------------------------
# 1. The optional dependency stays optional
# ---------------------------------------------------------------------------

def test_open_clip_is_not_a_module_level_import_of_networks():
    """src.networks must import with open_clip absent.

    It is imported by src/algorithms/nec.py and by every DQN/DDPG/A2C config,
    so a module-scope `import open_clip` there would take the whole repo down
    on a machine without the optional extra — not just the clip arm. Checked
    by parsing the AST rather than by string matching, so the lazy import
    inside `CLIPEmbedding.__init__` does not produce a false positive.
    """
    import ast

    import src.networks as networks           # noqa: F401
    import src.algorithms.nec as nec          # noqa: F401  (imports networks)

    tree = ast.parse(open(networks.__file__).read())
    top_level = [
        alias.name.split(".")[0]
        for node in tree.body                  # module scope only
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", [])
    ] + [
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert "open_clip" not in top_level, (
        "open_clip must be imported lazily inside CLIPEmbedding.__init__"
    )


def test_missing_open_clip_raises_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(ImportError, match="open_clip_torch"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 2. QuickGELU pairing — silent wrong features if unguarded
# ---------------------------------------------------------------------------

def test_openai_tag_with_plain_gelu_model_is_rejected(stub_open_clip):
    with pytest.raises(ValueError, match="QuickGELU"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, model_name="ViT-B-32",
                      pretrained_tag="openai")


def test_the_rejection_names_the_exact_fix(stub_open_clip):
    with pytest.raises(ValueError, match=r"ViT-B-16-quickgelu"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, model_name="ViT-B-16",
                      pretrained_tag="openai")


def test_openai_tag_is_rejected_even_with_local_weights(stub_open_clip):
    """A local file does not make the architecture right."""
    with pytest.raises(ValueError, match="QuickGELU"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, model_name="ViT-B-32",
                      pretrained_tag="openai",
                      weights_path="/models/clip_ViT-B-32_openai.pt")


def test_laion_tags_want_the_plain_name(stub_open_clip):
    """The rule is not "always quickgelu" — LAION used standard GELU."""
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, model_name="ViT-B-32",
                  pretrained_tag="laion2b_s34b_b79k")
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, model_name="ViT-B-32",
                  pretrained_tag="datacomp_xl_s13b_b90k")


def test_the_default_pairing_is_self_consistent(stub_open_clip):
    """The constructor defaults must not trip the constructor's own guard."""
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert "quickgelu" in net.model_name.lower()


# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

def test_local_weights_path_wins_over_the_hub_tag(stub_open_clip):
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, weights_path="/local/ckpt.bin",
                  pretrained_tag="openai", model_name="ViT-B-16-quickgelu")
    assert stub_open_clip["model_name"] == "ViT-B-16-quickgelu"
    assert stub_open_clip["pretrained"] == "/local/ckpt.bin"
    assert stub_open_clip["precision"] == "fp32"
    # Not forwarded unless requested — older open_clip lacks the kwarg.
    assert "force_image_size" not in stub_open_clip


def test_hub_tag_used_when_no_local_path(stub_open_clip):
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, weights_path=None,
                  pretrained_tag="laion2b_s34b_b79k", model_name="ViT-B-32")
    assert stub_open_clip["pretrained"] == "laion2b_s34b_b79k"


def test_only_the_vision_tower_is_kept(stub_open_clip):
    """The ~63M-param text tower must not reach RMSProp or the checkpoint."""
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert isinstance(net.backbone, _StubVisual)
    assert not hasattr(net.backbone, "transformer")
    assert not any("transformer" in n for n in net.state_dict())


def test_proj_dim_is_probed_from_a_real_forward(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert net.proj_dim == PROJ_DIM
    assert net.head.in_features == PROJ_DIM
    assert net.head.out_features == EMBEDDING_DIM


def test_tuple_valued_image_size_is_accepted(stub_open_clip):
    """open_clip reports (H, W); the module needs a scalar."""
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert net.image_size == NATIVE_SIZE


def test_explicit_image_size_is_forwarded_as_force_image_size(stub_open_clip):
    stub_open_clip["_visual_kwargs"]["image_size"] = 96
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=96)
    assert stub_open_clip["force_image_size"] == 96


def test_old_open_clip_without_force_image_size_gives_a_clear_error(monkeypatch):
    def create_model_and_transforms(model_name, **kwargs):
        if "force_image_size" in kwargs:
            raise TypeError("unexpected keyword argument 'force_image_size'")
        return _StubCLIP(), None, None

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", module)

    with pytest.raises(TypeError, match="native resolution"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=96)
    # ...and the native-resolution path still works on that same old version.
    assert CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM).proj_dim == PROJ_DIM


# ---------------------------------------------------------------------------
# 3b. Patch-grid guard — open_clip drops pixels silently without it
# ---------------------------------------------------------------------------

def test_image_size_not_divisible_by_the_patch_size_is_rejected(stub_open_clip):
    """force_image_size=112 on a patch-32 tower discards 16 px per axis.

    open_clip builds and runs it without complaint (verified against 3.3.0:
    the patch conv yields a 3x3 grid covering 96 of 112 pixels). On Ms.
    Pac-Man the dropped strip is the score/lives row.
    """
    stub_open_clip["_visual_kwargs"]["image_size"] = 112
    with pytest.raises(ValueError, match="not divisible"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=112)


def test_the_patch_guard_names_valid_sizes(stub_open_clip):
    stub_open_clip["_visual_kwargs"]["image_size"] = 100
    with pytest.raises(ValueError, match=r"Valid sizes: \[32, 64, 96"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100)


def test_a_divisible_image_size_is_accepted(stub_open_clip):
    stub_open_clip["_visual_kwargs"]["image_size"] = 96
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=96)
    assert net.image_size == 96


def test_native_resolution_is_never_rejected(stub_open_clip):
    """image_size=None must skip the guard even for an odd native size."""
    stub_open_clip["_visual_kwargs"]["image_size"] = 100
    assert CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=None).image_size == 100


def test_towers_without_a_patch_conv_skip_the_guard(stub_open_clip):
    """Non-ViT towers have no patch grid; the arithmetic does not apply."""
    stub_open_clip["_visual_kwargs"]["with_conv1"] = False
    stub_open_clip["_visual_kwargs"]["image_size"] = 112
    assert CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=112).image_size == 112


# ---------------------------------------------------------------------------
# 3c. BatchNorm warning — NEC trains in train() mode
# ---------------------------------------------------------------------------

def test_batchnorm_tower_warns(stub_open_clip):
    """CLIP's RN* towers carry BatchNorm; NEC never calls .eval().

    Batch statistics would make the same frame embed differently during
    collection (batch=num_envs), gradient steps (batch=batch_size) and
    evaluation (batch=1) — which destabilises every key in the DND.
    """
    stub_open_clip["_visual_kwargs"]["batchnorm"] = True
    with pytest.warns(UserWarning, match="BatchNorm"):
        CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)


def test_vit_tower_does_not_warn(stub_open_clip, recwarn):
    CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert not [w for w in recwarn if "BatchNorm" in str(w.message)]


# ---------------------------------------------------------------------------
# 4. Preprocessing — CLIP's constants, whole-frame resize
# ---------------------------------------------------------------------------

def test_clip_normalisation_stats_are_used_not_imagenet(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    # A constant 4-channel frame -> adapter mean -> constant 0.5 RGB.
    net(torch.full((1, *OBS_SHAPE), 0.5))

    got = net.backbone.last_input[0, :, 0, 0]
    expected = (0.5 - torch.tensor(_CLIP_MEAN)) / torch.tensor(_CLIP_STD)
    assert torch.allclose(got, expected, atol=1e-6)

    imagenet = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    assert not torch.allclose(got, imagenet, atol=1e-3), (
        "CLIP is being fed ImageNet statistics; they are different numbers "
        "and stock CLIP inference uses its own."
    )


def test_stats_declared_by_the_checkpoint_win(stub_open_clip):
    """Newer open_clip hangs the checkpoint's own stats off the tower."""
    stub_open_clip["_visual_kwargs"]["declared_stats"] = (
        (0.1, 0.2, 0.3), (0.4, 0.5, 0.6)
    )
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert torch.allclose(net._mean.flatten(), torch.tensor((0.1, 0.2, 0.3)))
    assert torch.allclose(net._std.flatten(), torch.tensor((0.4, 0.5, 0.6)))


def test_explicit_stats_override_the_checkpoint(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM,
                        image_mean=(0.7, 0.8, 0.9), image_std=(0.1, 0.1, 0.1))
    assert torch.allclose(net._mean.flatten(), torch.tensor((0.7, 0.8, 0.9)))


def test_clip_constants_are_the_fallback_on_older_open_clip(stub_open_clip):
    """A tower that declares nothing must still get CLIP's stats, not None."""
    stub_open_clip["_visual_kwargs"]["declared_stats"] = None
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert torch.allclose(net._mean.flatten(), torch.tensor(_CLIP_MEAN))
    assert torch.allclose(net._std.flatten(), torch.tensor(_CLIP_STD))


def test_frames_are_resized_to_the_tower_resolution(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    net(torch.rand(2, *OBS_SHAPE))
    assert net.backbone.last_input.shape == (2, 3, NATIVE_SIZE, NATIVE_SIZE)


def test_the_whole_frame_is_resized_not_centre_cropped(stub_open_clip):
    """A centre crop would cut the maze edges and the score row off-screen."""
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    obs = torch.zeros(1, *OBS_SHAPE)
    obs[..., :4] = 1.0                        # paint the leftmost columns only
    net(obs)

    raw = net.backbone.last_input * net._std + net._mean
    assert raw[..., :2].max() > 0.9, "left edge lost — this is a centre crop"


def test_interpolation_mode_is_configurable(stub_open_clip):
    assert CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM).interpolation == "bicubic"
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, interpolation="bilinear")
    net(torch.rand(1, *OBS_SHAPE))            # must actually run
    assert net.interpolation == "bilinear"


# ---------------------------------------------------------------------------
# 5. Channel adapter and the NECEmbeddingNetwork contract
# ---------------------------------------------------------------------------

def test_channel_adapter_starts_as_grayscale_to_rgb(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    x = torch.rand(3, *OBS_SHAPE)
    with torch.no_grad():
        out = net.channel_adapter(x)

    expected = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_rgb_observations_skip_the_adapter(stub_open_clip):
    net = CLIPEmbedding((3, 84, 84), EMBEDDING_DIM)
    assert isinstance(net.channel_adapter, nn.Identity)
    assert net(torch.rand(2, 3, 84, 84)).shape == (2, EMBEDDING_DIM)


def test_forward_shape_and_dtype(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    out = net(torch.rand(5, *OBS_SHAPE))
    assert out.shape == (5, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_output_is_not_prenormalised(stub_open_clip):
    """Contract clause 3: NEC normalises downstream; the module must not.

    This must hold even with normalize_features=True, which normalises the
    CLIP embedding *before* the head, not the module's output.
    """
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, normalize_features=True)
    norms = net(torch.rand(8, *OBS_SHAPE)).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    assert (norms > 0).all()


def test_normalize_features_puts_the_head_input_on_the_unit_sphere(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, normalize_features=True)
    obs = torch.rand(4, *OBS_SHAPE)

    captured = {}
    net.head.register_forward_pre_hook(
        lambda _m, args: captured.setdefault("x", args[0].detach())
    )
    net(obs)
    assert torch.allclose(captured["x"].norm(dim=-1), torch.ones(4), atol=1e-5)


def test_normalize_features_false_leaves_the_head_input_unscaled(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, normalize_features=False)
    captured = {}
    net.head.register_forward_pre_hook(
        lambda _m, args: captured.setdefault("x", args[0].detach())
    )
    net(torch.rand(4, *OBS_SHAPE))
    assert not torch.allclose(
        captured["x"].norm(dim=-1), torch.ones(4), atol=1e-3
    )


def test_all_parameters_are_trainable_by_default(stub_open_clip):
    """Contract clause 2 — the difference from MFEC's frozen CLIPEncoder."""
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    assert all(p.requires_grad for p in net.parameters())
    assert all(p.requires_grad for p in net.backbone.parameters()), (
        "src/encoders/clip_encoder.py freezes the tower for MFEC; the NEC "
        "variant is finetunable and must not"
    )


def test_freeze_backbone_leaves_the_head_trainable(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, freeze_backbone=True)
    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.head.parameters())
    assert any(p.requires_grad for p in net.parameters())


# ---------------------------------------------------------------------------
# 6. Param groups
# ---------------------------------------------------------------------------

def test_param_groups_split_backbone_from_head(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, backbone_lr_scale=0.1)
    groups = net.param_groups(1e-3)

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-4)
    assert groups[1]["lr"] == pytest.approx(1e-3)
    assert {id(p) for g in groups for p in g["params"]} == {
        id(p) for p in net.parameters() if p.requires_grad
    }


def test_param_groups_drop_the_backbone_when_frozen(stub_open_clip):
    net = CLIPEmbedding(OBS_SHAPE, EMBEDDING_DIM, freeze_backbone=True)
    groups = net.param_groups(1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)


def test_setup_builds_the_optimizer_from_param_groups(stub_open_clip):
    alg = _make_nec(_factory(backbone_lr_scale=0.1), lr=1e-3)
    assert len(alg.optimizer.param_groups) == 2
    assert [g["lr"] for g in alg.optimizer.param_groups] == [
        pytest.approx(1e-4), pytest.approx(1e-3)
    ]


# ---------------------------------------------------------------------------
# 7. NEC actually finetunes it
# ---------------------------------------------------------------------------

def test_step_trains_the_vision_tower_end_to_end(stub_open_clip):
    torch.manual_seed(0)
    alg = _make_nec(_factory(backbone_lr_scale=0.1), lr=1e-2)

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    metrics = alg.step(_episode_batch())

    assert sum(alg.dnd._sizes) > 0, "step() stored nothing -- assertions vacuous"
    assert metrics["train/updates"] > 0, "no gradient update ran"

    moved = {
        n for n, p in alg.embedding_net.named_parameters()
        if not torch.equal(p.detach(), before[n])
    }
    assert any(n.startswith("backbone.") for n in moved), (
        "the CLIP vision tower was not updated -- this is the frozen-MFEC "
        "behaviour, and the NEC variant exists specifically to avoid it"
    )
    assert any(n.startswith("head.") for n in moved)
    assert any(n.startswith("channel_adapter.") for n in moved)


def test_gradient_reaches_the_tower_through_the_dnd_kernel(stub_open_clip):
    torch.manual_seed(0)
    alg = _make_nec(_factory())
    alg.step(_episode_batch())

    alg.optimizer.zero_grad()
    assert alg._gradient_step() is not None, "gradient step skipped"

    grads = [p.grad for p in alg.embedding_net.backbone.parameters()
             if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_checkpoint_roundtrip_preserves_weights_and_param_groups(stub_open_clip):
    torch.manual_seed(0)
    src = _make_nec(_factory(backbone_lr_scale=0.1), lr=1e-2)
    src.step(_episode_batch())
    state = src._get_training_state()

    torch.manual_seed(1)
    dst = _make_nec(_factory(backbone_lr_scale=0.1), lr=1e-2)
    obs = torch.rand(3, *OBS_SHAPE)
    assert not torch.allclose(dst.embedding_net(obs), src.embedding_net(obs)), (
        "the two networks start identical -- the restore assertion is vacuous"
    )

    dst._load_training_state(state)

    assert torch.allclose(dst.embedding_net(obs), src.embedding_net(obs), atol=1e-6)
    assert len(dst.optimizer.param_groups) == 2
    assert [g["lr"] for g in dst.optimizer.param_groups] == [
        pytest.approx(1e-3), pytest.approx(1e-2)
    ]


# ---------------------------------------------------------------------------
# 8. Hydra composition
# ---------------------------------------------------------------------------

def test_experiment_config_selects_the_finetunable_clip():
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg("nec/mspacman_clip", ["logger=[]"])

    assert cfg.algorithm.embedding_network._target_ == "src.networks.CLIPEmbedding"
    assert cfg.algorithm.embedding_network.freeze_backbone is False
    assert cfg.algorithm.embedding_network.model_name == "ViT-B-32-quickgelu"
    assert cfg.algorithm.embedding_network.pretrained_tag == "openai"
    assert cfg.run.encoder == "clip"
    assert cfg.run.name == "nec_mspacman_clip_seed42"
    # Same env as nec/mspacman.yaml -> the encoder is the only variable.
    assert cfg.environment.name == "ALE/MsPacman-v5"
    assert any("CatFrames" in t["_target_"] for t in cfg.environment.transforms)

    alg = instantiate(cfg.algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)


def test_experiment_config_builds_the_network(stub_open_clip):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg("nec/mspacman_clip", ["logger=[]"])
    alg = instantiate(cfg.algorithm, device=None)
    net = alg._make_embedding_network(OBS_SHAPE, cfg.algorithm.embedding_dim)

    assert isinstance(net, CLIPEmbedding)
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, cfg.algorithm.embedding_dim)
    assert all(p.requires_grad for p in net.parameters())
    assert net.backbone_lr_scale == pytest.approx(0.1)


def test_cli_group_override_works_on_any_nec_experiment(stub_open_clip):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(
        "nec/pong",
        ["logger=[]", "algorithm/embedding_network=clip_finetune",
         "run.encoder=clip"],
    )
    assert cfg.algorithm.embedding_network._target_ == "src.networks.CLIPEmbedding"
    assert cfg.run.encoder == "clip"


# ---------------------------------------------------------------------------
# 9. Real CLIP architecture -- opt-in (NEC_CLIP_REAL=1)
# ---------------------------------------------------------------------------

@real_only
def test_real_tower_forward_shape_and_text_tower_dropped():
    pytest.importorskip("open_clip")
    net = _real()

    out = net(torch.rand(2, *OBS_SHAPE))
    assert out.shape == (2, EMBEDDING_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert net.proj_dim == PROJ_DIM

    # ViT-B-32: 87.8M in the vision tower, 63.4M in the text tower. The text
    # half must not reach RMSProp or the checkpoint.
    n_params = sum(p.numel() for p in net.backbone.parameters())
    assert 80e6 < n_params < 95e6, f"unexpected tower size {n_params/1e6:.1f}M"
    assert not any("token_embedding" in k for k in net.state_dict())


@real_only
def test_real_tower_uses_quickgelu_for_the_openai_pairing():
    pytest.importorskip("open_clip")
    quick = _real(model_name="ViT-B-32-quickgelu")
    plain = _real(model_name="ViT-B-32")

    def act(net):
        return type(net.backbone.transformer.resblocks[0].mlp.gelu).__name__

    assert act(quick) == "QuickGELU"
    assert act(plain) == "GELU", (
        "open_clip's plain ViT-B-32 no longer uses standard GELU; the "
        "QuickGELU pairing rule in CLIPEmbedding needs re-checking"
    )


@real_only
def test_real_tower_declares_clip_stats_not_imagenet():
    pytest.importorskip("open_clip")
    net = _real()
    assert torch.allclose(net._mean.flatten(), torch.tensor(_CLIP_MEAN), atol=1e-6)
    assert torch.allclose(net._std.flatten(), torch.tensor(_CLIP_STD), atol=1e-6)


@real_only
@pytest.mark.parametrize("image_size", [96, 224])
def test_real_tower_accepts_patch_aligned_sizes(image_size):
    pytest.importorskip("open_clip")
    net = _real(image_size=None if image_size == 224 else image_size)
    assert net.image_size == image_size
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, EMBEDDING_DIM)


@real_only
def test_real_tower_rejects_the_pixel_dropping_size():
    """112 on a patch-32 tower: open_clip allows it, we must not.

    Verified against open_clip 3.3.0 -- the patch conv yields a 3x3 grid
    covering 96 of 112 pixels, discarding 16 per axis with no warning.
    """
    pytest.importorskip("open_clip")
    with pytest.raises(ValueError, match="not divisible"):
        _real(image_size=112)


@real_only
def test_real_tower_transformer_blocks_receive_gradients():
    pytest.importorskip("open_clip")
    net = _real(image_size=96)
    net(torch.rand(2, *OBS_SHAPE)).square().mean().backward()

    named = dict(net.backbone.named_parameters())
    for key in ("conv1.weight", "positional_embedding", "class_embedding",
                "transformer.resblocks.0.attn.in_proj_weight",
                "transformer.resblocks.0.mlp.c_fc.weight", "proj"):
        grad = named[key].grad
        assert grad is not None, f"no gradient reached backbone.{key}"
        assert grad.abs().sum() > 0, f"gradient at backbone.{key} is all zero"


@real_only
def test_real_tower_runs_end_to_end_through_nec():
    pytest.importorskip("open_clip")
    torch.manual_seed(0)
    alg = _make_nec(
        functools.partial(
            CLIPEmbedding, model_name=REAL_MODEL, pretrained_tag=None,
            image_size=96, backbone_lr_scale=0.1,
        ),
        lr=1e-3,
    )
    assert len(alg.optimizer.param_groups) == 2

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    metrics = alg.step(_episode_batch())

    assert sum(alg.dnd._sizes) > 0
    assert metrics["train/updates"] > 0
    assert any(
        n.startswith("backbone.") and not torch.equal(p.detach(), before[n])
        for n, p in alg.embedding_net.named_parameters()
    ), "the real CLIP tower was not updated by NEC's loop"

    q = alg._dnd_policy(torch.rand(2, *OBS_SHAPE))
    assert q.shape == (2, 2)


@real_only
def test_real_vit_tower_has_no_batchnorm(recwarn):
    """Every ViT-* tower is batch-independent; only the RN* ones are not."""
    pytest.importorskip("open_clip")
    net = _real()

    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        for m in net.backbone.modules()
    )
    assert not [w for w in recwarn if "BatchNorm" in str(w.message)]


@real_only
@pytest.mark.skipif(
    not os.environ.get("CLIP_WEIGHTS"),
    reason="set CLIP_WEIGHTS=/path/to/open_clip_pytorch_model.bin to run",
)
def test_pretrained_checkpoint_loads():
    """The actual released weights, not a random init."""
    pytest.importorskip("open_clip")
    net = CLIPEmbedding(
        OBS_SHAPE, EMBEDDING_DIM,
        weights_path=os.environ["CLIP_WEIGHTS"],
        model_name=os.environ.get("CLIP_MODEL", REAL_MODEL),
        pretrained_tag="openai",
    )
    out = net(torch.rand(2, *OBS_SHAPE))
    assert out.shape == (2, EMBEDDING_DIM)
    assert torch.isfinite(out).all()
