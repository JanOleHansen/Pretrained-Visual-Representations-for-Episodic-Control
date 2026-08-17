"""Tests for NEC's finetunable MAE embedding network.

`src/networks.py::MAEEmbedding`, selected by
`algorithm/embedding_network=mae_finetune`. The trainable counterpart to MFEC's
frozen `src/encoders/mae_encoder.py::MAEEncoder` (covered by
tests/test_mae_encoder.py), and the reconstruction arm of the NEC encoder
ablation against `nature`, `dinov2_finetune` and `clip_finetune`.

``timm`` is an OPTIONAL dependency (``uv sync --extra mae``), so the default
tier injects a stub ``timm`` module into ``sys.modules`` — which also pins the
property that makes the dependency optional at all: ``MAEEmbedding`` imports
``timm`` lazily, inside ``__init__``. A module-scope import in
``src/networks.py`` would break *every* algorithm in the repo, since
``src/algorithms/nec.py`` and the DQN/DDPG/A2C configs all import that module.

Two tiers, mirroring tests/test_nec_dinov2_finetune.py and
tests/test_nec_clip_finetune.py:

  * Stub tests (default): a tiny ViT with the same
    ``forward_features(B, 3, H, W) -> (B, num_prefix + P, d)`` contract. Covers
    the channel adapter, the resize / ImageNet-normalise pipeline, POOLING,
    the patch-grid guard, param groups, finetuning through NEC's real
    ``step()``, and checkpoint round-trip.

  * Real-architecture tests: skipped unless NEC_MAE_REAL=1 *and* timm is
    importable. These build the genuine ``vit_base_patch16_224.mae``
    architecture with ``pretrained=False``, so they need no network and no
    checkpoint. They pin what a stub structurally cannot: the real patch grid,
    that timm resamples ``pos_embed`` for a non-native ``image_size``, and that
    gradients reach the ViT's own transformer blocks. ``MAE_WEIGHTS`` (or
    NEC_MAE_DOWNLOAD=1) additionally exercises the pretrained checkpoint.

The pooling tests are load-bearing. MAE's CLS token is never directly
supervised by the reconstruction loss, so taking it would measure our pooling
choice and report it as a property of MAE — see the module docstring of
src/encoders/mae_encoder.py.

Deliberately NOT covered: whether NEC scores better with MAE than with
`nature`, `dinov2_finetune` or `clip_finetune`. That is the experiment.
"""
from __future__ import annotations

import functools
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.nec import NECAlgorithm
from src.networks import _IMAGENET_MEAN, _IMAGENET_STD, MAEEmbedding


OBS_SHAPE = (4, 84, 84)          # Atari: 4 stacked grayscale frames
EMBEDDING_DIM = 16
STUB_EMBED_DIM = 32
STUB_GRID = 7                    # 112 / 16 -> 7x7 = 49 patch tokens
STUB_IMAGE_SIZE = 112
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)   # what MAE must NOT use

REAL_MODEL = "vit_base_patch16_224.mae"
REAL_EMBED_DIM = 768

#: Sentinel written into the prefix tokens of the poisoned stub. Large enough
#: that a mean which wrongly includes a prefix token cannot be mistaken for a
#: patch mean.
_PREFIX_SENTINEL = 1e6

real_only = pytest.mark.skipif(
    os.environ.get("NEC_MAE_REAL") != "1",
    reason="set NEC_MAE_REAL=1 (and have timm installed) to build the genuine "
           "MAE ViT-B/16 architecture",
)


# ---------------------------------------------------------------------------
# Stub timm
# ---------------------------------------------------------------------------

class _StubViT(nn.Module):
    """``(B, 3, H, W) -> (B, num_prefix + 49, d)`` tokens. Records its input.

    Mirrors the parts of ``timm.models.VisionTransformer`` that
    ``MAEEmbedding`` touches: ``forward_features``, ``num_prefix_tokens``,
    ``embed_dim``, and a ``patch_embed.proj`` conv the patch-grid guard reads
    its stride off. Same contract as tests/test_mae_encoder.py's stub.

    ``patch_embed.proj`` is genuinely used by ``forward_features``, not merely
    hung on the module for the guard: a decorative conv would be the only
    parameter in the backbone that never receives a gradient, which would make
    the "gradients reach the backbone" test fail for a reason that exists
    nowhere in timm.
    """

    def __init__(
        self,
        embed_dim: int = STUB_EMBED_DIM,
        num_prefix_tokens: int = 1,
        patch: int = 16,
        prefix_value: float | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        # timm's default for the .mae tag. Never read by MAEEmbedding — the
        # point is that it pools itself — but present so a test can assert it.
        self.global_pool = "token"
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch,
                                          stride=patch)
        # A SEPARATE projection for the prefix token, so CLS is a genuinely
        # different function of the frame than the patch mean is. With one
        # shared layer the two coincide and every pooling test passes vacuously.
        self.cls_proj = nn.Linear(embed_dim, embed_dim)
        self._prefix_value = prefix_value
        self.last_input: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        tokens = self.patch_embed.proj(x)                  # (B, d, g, g)
        tokens = tokens.flatten(2).transpose(1, 2)         # (B, P, d)
        if self._prefix_value is None:
            prefix = self.cls_proj(tokens.mean(dim=1))[:, None, :]
            prefix = prefix.expand(-1, self.num_prefix_tokens, -1)
        else:
            prefix = torch.full(
                (x.shape[0], self.num_prefix_tokens, self.embed_dim),
                self._prefix_value, dtype=tokens.dtype, device=tokens.device,
            )
        return torch.cat([prefix, tokens], dim=1)


@pytest.fixture
def stub_timm(monkeypatch):
    """Install a fake ``timm``; yields the recorded build kwargs.

    ``seen["_vit_kwargs"]`` is test-writable, so a test can ask for a poisoned
    prefix or a different patch size before constructing.
    """
    seen: dict = {}
    vit_kwargs: dict = {}

    def create_model(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        return _StubViT(**vit_kwargs)

    module = types.ModuleType("timm")
    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)
    seen["_vit_kwargs"] = vit_kwargs
    return seen


def _factory(**kwargs):
    """`MAEEmbedding` pre-bound like a Hydra `_partial_` would."""
    kwargs.setdefault("image_size", STUB_IMAGE_SIZE)
    return functools.partial(MAEEmbedding, **kwargs)


def _real(**kwargs) -> MAEEmbedding:
    kwargs.setdefault("model_name", REAL_MODEL)
    kwargs.setdefault("pretrained", False)      # architecture only, no network
    kwargs.setdefault("image_size", STUB_IMAGE_SIZE)
    return MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, **kwargs)


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

def test_timm_is_not_a_module_level_import_of_networks():
    """src.networks must import with timm absent.

    It is imported by src/algorithms/nec.py and by every DQN/DDPG/A2C config,
    so a module-scope `import timm` there would take the whole repo down on a
    machine without the optional extra — not just the mae arm. Checked by
    parsing the AST rather than by string matching, so the lazy import inside
    `MAEEmbedding.__init__` does not produce a false positive.
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
    assert "timm" not in top_level, (
        "timm must be imported lazily inside MAEEmbedding.__init__"
    )


def test_missing_timm_raises_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)
    with pytest.raises(ImportError, match="uv sync --extra mae"):
        MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def test_build_kwargs_match_the_frozen_encoders(stub_timm):
    """num_classes=0, img_size forwarded, and global_pool NEVER passed.

    Passing global_pool='avg' would make timm's `norm` an nn.Identity and add a
    randomly-initialised `fc_norm` the MAE checkpoint does not contain, so the
    pretrained final LayerNorm would be silently dropped. This module pools
    from forward_features itself, so timm's pooling is never used either way.
    """
    MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    assert stub_timm["model_name"] == "vit_base_patch16_224.mae"
    assert stub_timm["num_classes"] == 0
    assert stub_timm["img_size"] == STUB_IMAGE_SIZE
    assert stub_timm["pretrained"] is True
    assert "global_pool" not in stub_timm
    assert "pretrained_cfg_overlay" not in stub_timm


def test_local_weights_path_is_routed_through_timms_filter_fn(stub_timm):
    """A path becomes pretrained_cfg_overlay=dict(file=...), not torch.load.

    That is what runs timm's checkpoint_filter_fn, which unwraps a {"model":
    ...} wrapper, remaps the original mae_pretrain_vit_base.pth key names, and
    resamples pos_embed for a non-native image_size. A raw
    load_state_dict(strict=True) does none of that.
    """
    MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                 weights_path="/models/mae_pretrain_vit_base.pth")

    assert stub_timm["pretrained_cfg_overlay"] == {
        "file": "/models/mae_pretrain_vit_base.pth"
    }


def test_a_weights_path_overrides_pretrained_false(stub_timm):
    """A path means "load this file", whatever `pretrained` says."""
    MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                 pretrained=False, weights_path="/models/mae.pth")
    assert stub_timm["pretrained"] is True


def test_pretrained_false_builds_the_architecture_untrained(stub_timm):
    MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                 pretrained=False)
    assert stub_timm["pretrained"] is False


def test_feature_dim_is_probed_from_a_real_forward(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    assert net.feature_dim == STUB_EMBED_DIM
    assert net.head.in_features == STUB_EMBED_DIM
    assert net.head.out_features == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 3. Pooling — the choice that decides whether this arm measures MAE
# ---------------------------------------------------------------------------

def test_mean_pooling_is_the_default_and_drops_the_prefix_tokens(stub_timm):
    """MAE's CLS token is undertrained; averaging it in would bias the key.

    The stub's prefix tokens are poisoned with a huge sentinel, so a mean that
    wrongly includes them cannot be mistaken for a patch mean.
    """
    stub_timm["_vit_kwargs"]["prefix_value"] = _PREFIX_SENTINEL
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    assert net.pooling == "mean"

    with torch.no_grad():
        tokens = net.backbone.forward_features(
            torch.rand(2, 3, STUB_IMAGE_SIZE, STUB_IMAGE_SIZE)
        )
    pooled = net._pool(tokens)

    assert pooled.shape == (2, STUB_EMBED_DIM)
    assert float(pooled.abs().max()) < _PREFIX_SENTINEL / 1e3, (
        "a prefix token leaked into the patch mean"
    )
    assert torch.allclose(pooled, tokens[:, net._num_prefix:].mean(dim=1))


def test_cls_pooling_takes_token_zero(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                       pooling="cls")
    tokens = net.backbone.forward_features(
        torch.rand(2, 3, STUB_IMAGE_SIZE, STUB_IMAGE_SIZE)
    )
    assert torch.allclose(net._pool(tokens), tokens[:, 0])


def test_the_two_pooling_modes_disagree(stub_timm):
    """Otherwise both pooling tests would pass for the wrong reason."""
    obs = torch.rand(3, *OBS_SHAPE)
    torch.manual_seed(0)
    mean_net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    torch.manual_seed(0)
    cls_net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                           pooling="cls")
    cls_net.load_state_dict(mean_net.state_dict())

    assert not torch.allclose(mean_net(obs), cls_net(obs), atol=1e-5)


def test_prefix_token_count_is_read_not_assumed(stub_timm):
    """A model with register tokens reports more than 1; hardcoding 1 would
    average a register token into the embedding."""
    stub_timm["_vit_kwargs"]["num_prefix_tokens"] = 5
    stub_timm["_vit_kwargs"]["prefix_value"] = _PREFIX_SENTINEL
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    assert net._num_prefix == 5
    with torch.no_grad():
        tokens = net.backbone.forward_features(
            torch.rand(2, 3, STUB_IMAGE_SIZE, STUB_IMAGE_SIZE)
        )
    assert float(net._pool(tokens).abs().max()) < _PREFIX_SENTINEL / 1e3


def test_an_unknown_pooling_mode_is_rejected(stub_timm):
    with pytest.raises(ValueError, match="not a pooling mode"):
        MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, pooling="avg")


# ---------------------------------------------------------------------------
# 4. Patch-grid guard — timm drops pixels silently without it
# ---------------------------------------------------------------------------

def test_image_size_not_divisible_by_the_patch_size_is_rejected(stub_timm):
    """img_size=100 on a patch-16 ViT discards 4 px per axis.

    timm builds and runs it without complaint (verified against 1.0.28: the
    patch conv yields a 6x6 grid covering 96 of 100 pixels). On Ms. Pac-Man the
    dropped strip is the score/lives row.
    """
    with pytest.raises(ValueError, match="not divisible"):
        MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100)


def test_the_patch_guard_names_valid_sizes(stub_timm):
    with pytest.raises(ValueError, match=r"Valid sizes: \[16, 32, 48"):
        MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100)


@pytest.mark.parametrize("image_size", [112, 224])
def test_divisible_image_sizes_are_accepted(stub_timm, image_size):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=image_size)
    assert net.image_size == image_size
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, EMBEDDING_DIM)


def test_backbones_without_a_patch_conv_skip_the_guard(stub_timm, monkeypatch):
    """Not a patch-embed ViT; the arithmetic does not apply."""
    def create_model(model_name, **kwargs):
        vit = _StubViT()
        del vit.patch_embed.proj
        vit.patch_embed = nn.Identity()
        vit.forward_features = lambda x: torch.zeros(
            x.shape[0], 1 + STUB_GRID ** 2, STUB_EMBED_DIM
        )
        return vit

    module = types.ModuleType("timm")
    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)

    assert MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100).image_size == 100


# ---------------------------------------------------------------------------
# 5. Preprocessing — ImageNet stats, whole-frame resize
# ---------------------------------------------------------------------------

def test_imagenet_stats_are_used_not_clips(stub_timm):
    """MAE was pretrained with ImageNet statistics, not CLIP's."""
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    # A constant 4-channel frame -> adapter mean -> constant 0.5 RGB.
    net(torch.full((1, *OBS_SHAPE), 0.5))

    got = net.backbone.last_input[0, :, 0, 0]
    expected = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    assert torch.allclose(got, expected, atol=1e-6)

    clip = (0.5 - torch.tensor(_CLIP_MEAN)) / torch.tensor(
        (0.26862954, 0.26130258, 0.27577711)
    )
    assert not torch.allclose(got, clip, atol=1e-3), (
        "MAE is being fed CLIP's statistics; they are different numbers and "
        "MAE was pretrained on ImageNet's."
    )


def test_the_normalisation_buffers_are_non_persistent(stub_timm):
    """Constants, not state: they must not bloat or gate the checkpoint."""
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    assert "_mean" not in net.state_dict() and "_std" not in net.state_dict()


def test_frames_are_resized_to_image_size(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    net(torch.rand(2, *OBS_SHAPE))
    assert net.backbone.last_input.shape == (2, 3, STUB_IMAGE_SIZE, STUB_IMAGE_SIZE)


def test_the_whole_frame_is_resized_not_centre_cropped(stub_timm):
    """A centre crop would cut the maze edges and the score row off-screen."""
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    obs = torch.zeros(1, *OBS_SHAPE)
    obs[..., :4] = 1.0                        # paint the leftmost columns only
    net(obs)

    raw = net.backbone.last_input * net._std + net._mean
    assert raw[..., :2].max() > 0.9, "left edge lost — this is a centre crop"


# ---------------------------------------------------------------------------
# 6. Channel adapter and the NECEmbeddingNetwork contract
# ---------------------------------------------------------------------------

def test_channel_adapter_starts_as_grayscale_to_rgb(stub_timm):
    """weight = 1/C, bias = 0: the frame stack's mean, replicated to R=G=B.

    A default-initialised nn.Conv2d emits channels of arbitrary scale and sign,
    so the ViT's first forward sees input nowhere near the ImageNet
    distribution its normalisation assumes — and a pretrained representation
    that only becomes useful after the adapter has been *learned* is not a
    pretrained representation.
    """
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    x = torch.rand(3, *OBS_SHAPE)                  # [0, 1], as ToTensorImage gives
    with torch.no_grad():
        out = net.channel_adapter(x)

    assert out.shape == (3, 3, 84, 84)
    expected = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[:, 0], out[:, 1]) and torch.allclose(out[:, 0], out[:, 2])
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert torch.allclose(net.channel_adapter.weight,
                          torch.full_like(net.channel_adapter.weight, 0.25))
    assert torch.equal(net.channel_adapter.bias,
                       torch.zeros_like(net.channel_adapter.bias))


def test_channel_adapter_is_trainable_not_a_fixed_transform(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    assert isinstance(net.channel_adapter, nn.Conv2d)
    assert all(p.requires_grad for p in net.channel_adapter.parameters())

    net(torch.rand(2, *OBS_SHAPE)).sum().backward()
    assert net.channel_adapter.weight.grad is not None
    assert net.channel_adapter.weight.grad.abs().sum() > 0


def test_rgb_observations_skip_the_adapter(stub_timm):
    net = MAEEmbedding((3, 84, 84), EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    assert isinstance(net.channel_adapter, nn.Identity)
    assert net(torch.rand(2, 3, 84, 84)).shape == (2, EMBEDDING_DIM)


def test_forward_shape_and_dtype(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    out = net(torch.rand(5, *OBS_SHAPE))
    assert out.shape == (5, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_output_is_not_prenormalised(stub_timm):
    """Contract clause 3: NEC normalises downstream; the module must not.

    Pre-normalising makes NEC's F.normalize a no-op and hides gradient scale.
    """
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    norms = net(torch.rand(8, *OBS_SHAPE)).norm(dim=-1)

    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    assert (norms > 0).all(), (
        "all-zero rows would survive F.normalize as zeros and collapse every "
        "DND kernel distance"
    )


def test_all_parameters_are_trainable_by_default(stub_timm):
    """Contract clause 2 — the difference from MFEC's frozen MAEEncoder."""
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    assert all(p.requires_grad for p in net.parameters())
    assert all(p.requires_grad for p in net.backbone.parameters()), (
        "src/encoders/mae_encoder.py freezes the ViT for MFEC; the NEC variant "
        "is finetunable and must not"
    )


def test_freeze_backbone_leaves_the_head_trainable(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                       freeze_backbone=True)
    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.head.parameters())
    assert any(p.requires_grad for p in net.parameters()), (
        "freezing everything would hand RMSProp an empty parameter list"
    )


# ---------------------------------------------------------------------------
# 7. Param groups
# ---------------------------------------------------------------------------

def test_param_groups_split_backbone_from_head(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                       backbone_lr_scale=0.1)
    groups = net.param_groups(1e-3)

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-4)     # backbone: scaled down
    assert groups[1]["lr"] == pytest.approx(1e-3)     # adapter + head: base

    grouped = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert grouped == trainable, (
        "param_groups must cover exactly the trainable parameters — a "
        "parameter missing from every group is silently never optimised"
    )
    assert {id(p) for p in net.backbone.parameters()} == {
        id(p) for p in groups[0]["params"]
    }


def test_param_groups_scale_of_one_is_a_uniform_learning_rate(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                       backbone_lr_scale=1.0)
    assert [g["lr"] for g in net.param_groups(1e-3)] == [
        pytest.approx(1e-3), pytest.approx(1e-3)
    ]


def test_param_groups_drop_the_backbone_when_frozen(stub_timm):
    net = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                       freeze_backbone=True)
    groups = net.param_groups(1e-3)

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert {id(p) for g in groups for p in g["params"]} == {
        id(p) for m in (net.channel_adapter, net.head) for p in m.parameters()
    }


def test_setup_builds_the_optimizer_from_param_groups(stub_timm):
    alg = _make_nec(_factory(backbone_lr_scale=0.1), lr=1e-3)

    assert len(alg.optimizer.param_groups) == 2
    assert [g["lr"] for g in alg.optimizer.param_groups] == [
        pytest.approx(1e-4), pytest.approx(1e-3)
    ]
    assert {id(p) for g in alg.optimizer.param_groups for p in g["params"]} == {
        id(p) for p in alg.embedding_net.parameters() if p.requires_grad
    }


# ---------------------------------------------------------------------------
# 8. NEC actually finetunes it
# ---------------------------------------------------------------------------

def test_step_trains_the_backbone_end_to_end(stub_timm):
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
        "the MAE backbone was not updated -- this is the frozen-MFEC "
        "behaviour, and the NEC variant exists specifically to avoid it"
    )
    assert any(n.startswith("head.") for n in moved)
    assert any(n.startswith("channel_adapter.") for n in moved)


def test_frozen_backbone_trains_only_the_head(stub_timm):
    torch.manual_seed(0)
    alg = _make_nec(_factory(freeze_backbone=True), lr=1e-2)

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    alg.step(_episode_batch())

    moved = {
        n for n, p in alg.embedding_net.named_parameters()
        if not torch.equal(p.detach(), before[n])
    }
    assert moved, "freezing the backbone must not freeze the whole network"
    assert not any(n.startswith("backbone.") for n in moved)


def test_gradient_reaches_the_backbone_through_the_dnd_kernel(stub_timm):
    """Gradients must arrive via the kernel distance term, not a shortcut."""
    torch.manual_seed(0)
    alg = _make_nec(_factory())
    alg.step(_episode_batch())          # populate DND + replay buffer

    alg.optimizer.zero_grad()
    assert alg._gradient_step() is not None, "gradient step skipped"

    grads = [p.grad for p in alg.embedding_net.backbone.parameters()
             if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# 9. Checkpointing — param groups must survive a resume
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_preserves_weights_and_param_groups(stub_timm):
    """Resume rebuilds the optimizer, so the grouping must be reproducible.

    `_load_training_state` calls `_build_optimizer()` and then
    `load_state_dict`, which raises on a group-count mismatch.
    """
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


def test_the_whole_module_state_lives_in_state_dict(stub_timm):
    """AGENTS.md §3a step 5: nothing outside state_dict() needs checkpointing.

    A fresh module of the same construction, loaded with the saved
    state_dict(), must reproduce the original's output exactly.
    """
    torch.manual_seed(0)
    src = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)
    torch.manual_seed(1)
    dst = MAEEmbedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    obs = torch.rand(4, *OBS_SHAPE)
    assert not torch.allclose(src(obs), dst(obs))

    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert not missing and not unexpected
    assert torch.allclose(src(obs), dst(obs), atol=1e-6)


# ---------------------------------------------------------------------------
# 10. Hydra composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["mspacman", "qbert", "frostbite"])
def test_experiment_config_selects_the_finetunable_mae(game):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(f"nec/{game}_mae", ["logger=[]"])

    assert cfg.algorithm.embedding_network._target_ == "src.networks.MAEEmbedding"
    assert cfg.algorithm.embedding_network.freeze_backbone is False
    assert cfg.algorithm.embedding_network.model_name == "vit_base_patch16_224.mae"
    assert cfg.algorithm.embedding_network.pooling == "mean"
    assert cfg.algorithm.embedding_network.image_size == 112
    # The run directory must not collide with the other encoder arms.
    assert cfg.run.encoder == "mae"
    assert cfg.run.name == f"nec_{game}_mae_seed42"

    alg = instantiate(cfg.algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)


@pytest.mark.parametrize("game", ["mspacman", "qbert", "frostbite"])
def test_the_mae_arm_holds_every_learning_knob_identical(game):
    """The ablation is void if one arm is tuned. These must match the nature arm."""
    from tests.conftest import load_experiment_cfg

    mae = load_experiment_cfg(f"nec/{game}_mae", ["logger=[]"])
    nature = load_experiment_cfg(f"nec/{game}", ["logger=[]"])

    for key in ("num_updates", "eps_end", "annealing_frames",
                "init_random_frames", "eval_eps", "kernel_delta", "gamma",
                "n_step", "embedding_dim", "k", "dnd_capacity", "lr",
                "batch_size"):
        assert mae.algorithm[key] == nature.algorithm[key], key
    assert mae.trainer.total_frames == nature.trainer.total_frames
    assert mae.trainer.seed == nature.trainer.seed
    assert mae.trainer.eval_every_n_steps == nature.trainer.eval_every_n_steps
    # Same env pair -> the encoder is the only variable.
    assert mae.environment.name == nature.environment.name
    assert mae.eval_environment.name == nature.eval_environment.name
    assert any("CatFrames" in t["_target_"] for t in mae.environment.transforms)


def test_experiment_config_builds_the_network(stub_timm):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg("nec/mspacman_mae", ["logger=[]"])
    alg = instantiate(cfg.algorithm, device=None)
    net = alg._make_embedding_network(OBS_SHAPE, cfg.algorithm.embedding_dim)

    assert isinstance(net, MAEEmbedding)
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, cfg.algorithm.embedding_dim)
    assert all(p.requires_grad for p in net.parameters())
    assert net.backbone_lr_scale == pytest.approx(0.1)


def test_cli_group_override_works_on_any_nec_experiment(stub_timm):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(
        "nec/pong",
        ["logger=[]", "algorithm/embedding_network=mae_finetune",
         "run.encoder=mae"],
    )
    assert cfg.algorithm.embedding_network._target_ == "src.networks.MAEEmbedding"
    assert cfg.run.encoder == "mae"


# ---------------------------------------------------------------------------
# 11. Real MAE architecture — opt-in (NEC_MAE_REAL=1)
# ---------------------------------------------------------------------------

@real_only
@pytest.mark.parametrize("image_size", [112, 224])
def test_real_vit_forward_shape_at_both_documented_image_sizes(image_size):
    pytest.importorskip("timm")
    net = _real(image_size=image_size)
    out = net(torch.rand(2, *OBS_SHAPE))

    assert out.shape == (2, EMBEDDING_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert net.feature_dim == REAL_EMBED_DIM
    # 85.7M in the backbone; the head is Linear(768, 16).
    n_params = sum(p.numel() for p in net.backbone.parameters())
    assert 80e6 < n_params < 90e6, f"unexpected backbone size {n_params/1e6:.1f}M"


@real_only
def test_real_patch_grid_and_prefix_tokens():
    """112 -> 7x7 = 49 patch tokens + 1 CLS, i.e. the CLIP arm's token count."""
    pytest.importorskip("timm")
    net = _real(image_size=112)

    assert net._num_prefix == 1
    assert tuple(net.backbone.patch_embed.grid_size) == (7, 7)
    tokens = net.backbone.forward_features(torch.zeros(1, 3, 112, 112))
    assert tokens.shape == (1, 50, REAL_EMBED_DIM)


@real_only
def test_real_timm_rejects_nothing_so_the_guard_must():
    """timm builds img_size=100 happily and silently drops 4 px per axis."""
    import timm

    m = timm.create_model(REAL_MODEL, pretrained=False, num_classes=0, img_size=100)
    assert tuple(m.patch_embed.grid_size) == (6, 6), (
        "timm no longer produces a 6x6 grid for img_size=100; re-check the "
        "pixel-dropping arithmetic in _assert_patch_grid_covers_the_image"
    )
    with pytest.raises(ValueError, match="not divisible"):
        _real(image_size=100)


@real_only
def test_real_vit_transformer_blocks_receive_gradients():
    """Finetuning has to reach the attention weights, not just the head."""
    pytest.importorskip("timm")
    net = _real(image_size=112)
    net(torch.rand(2, *OBS_SHAPE)).square().mean().backward()

    named = dict(net.backbone.named_parameters())
    for key in ("blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight",
                "patch_embed.proj.weight", "cls_token", "pos_embed"):
        grad = named[key].grad
        assert grad is not None, f"no gradient reached backbone.{key}"
        assert grad.abs().sum() > 0, f"gradient at backbone.{key} is all zero"


@real_only
def test_real_vit_has_no_batch_dependent_layers():
    """NEC keeps the embedding net in train() mode because it is being
    optimised. A BatchNorm would make the same frame embed differently during
    collection (batch=num_envs), gradient steps (batch=batch_size) and
    evaluation (batch=1), destabilising every DND key. timm ViTs are
    LayerNorm-only, which is why this class needs no warning."""
    pytest.importorskip("timm")
    net = _real()

    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        for m in net.modules()
    )
    obs = torch.rand(4, *OBS_SHAPE)
    net.train()
    with torch.no_grad():
        batched = net(obs)
        singly = torch.cat([net(obs[i: i + 1]) for i in range(4)])
    assert torch.allclose(batched, singly, atol=1e-4), (
        "the forward depends on the batch it was run in"
    )


@real_only
def test_real_vit_runs_end_to_end_through_nec():
    pytest.importorskip("timm")
    torch.manual_seed(0)
    alg = _make_nec(
        functools.partial(MAEEmbedding, model_name=REAL_MODEL, pretrained=False,
                          image_size=112, backbone_lr_scale=0.1),
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
    ), "the real MAE ViT was not updated by NEC's loop"

    q = alg._dnd_policy(torch.rand(2, *OBS_SHAPE))
    assert q.shape == (2, 2)


# ---------------------------------------------------------------------------
# 12. Real pretrained checkpoint — needs weights (MAE_WEIGHTS or a download)
# ---------------------------------------------------------------------------

pretrained_only = pytest.mark.skipif(
    os.environ.get("NEC_MAE_REAL") != "1"
    or not (os.environ.get("MAE_WEIGHTS") or os.environ.get("NEC_MAE_DOWNLOAD") == "1"),
    reason="set NEC_MAE_REAL=1 plus MAE_WEIGHTS=/path/to/checkpoint (or "
           "NEC_MAE_DOWNLOAD=1 to pull vit_base_patch16_224.mae from the "
           "HuggingFace hub) to run the pretrained-weights tier",
)


@pretrained_only
@pytest.mark.parametrize("image_size", [112, 224])
def test_pretrained_weights_load_and_pos_embed_is_resampled(image_size):
    """timm resamples pos_embed for a non-native image_size — hub OR local file.

    This is the property that makes image_size=112 legitimate rather than a
    silent architecture mismatch. Verified in timm's own source
    (`_builder.load_pretrained` applies `filter_fn` after every source branch,
    and `vision_transformer.checkpoint_filter_fn` calls `resample_abs_pos_embed`
    when the checkpoint's pos_embed does not match the model's), and measured
    here end-to-end.
    """
    weights = os.environ.get("MAE_WEIGHTS")
    net = _real(image_size=image_size, pretrained=True, weights_path=weights)
    random_init = _real(image_size=image_size, pretrained=False)

    expected_tokens = (image_size // 16) ** 2 + 1
    assert net.backbone.pos_embed.shape == (1, expected_tokens, REAL_EMBED_DIM)

    assert not torch.allclose(
        net.backbone.blocks[0].attn.qkv.weight,
        random_init.backbone.blocks[0].attn.qkv.weight,
    ), "the pretrained weights were not actually loaded"

    out = net(torch.rand(2, *OBS_SHAPE))
    assert out.shape == (2, EMBEDDING_DIM)
    assert torch.isfinite(out).all()


@pretrained_only
def test_a_local_file_loads_identically_to_the_hub(tmp_path):
    """The offline-cluster path must not be a different model.

    The file is written with the upstream facebookresearch/mae `{"model": ...}`
    wrapper, which timm's checkpoint_filter_fn unwraps; a raw
    load_state_dict(strict=True) would reject it outright.
    """
    pytest.importorskip("timm")
    hub = _real(image_size=112, pretrained=True,
                weights_path=os.environ.get("MAE_WEIGHTS"))

    ckpt = tmp_path / "mae_pretrain_vit_base.pth"
    torch.save({"model": hub.backbone.state_dict()}, ckpt)

    local = _real(image_size=112, pretrained=False, weights_path=str(ckpt))
    for (n, a), (_, b) in zip(
        local.backbone.state_dict().items(), hub.backbone.state_dict().items()
    ):
        assert torch.equal(a, b), f"backbone parameter {n} did not round-trip"
