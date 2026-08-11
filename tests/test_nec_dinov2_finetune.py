"""Tests for NEC's finetunable DINOv2 embedding network.

`src/networks.py::DINOv2Embedding`, selected by
`algorithm/embedding_network=dinov2_finetune`. This is the *trainable*
counterpart to MFEC's frozen `src/encoders/dino_v2_encoder.py::DINOv2Encoder`
(covered by tests/test_dinov2_encoder.py), and the difference is the whole
point: MFEC's ViT must never change because its QEC hash depends on a
bit-exact phi, while NEC optimises its embedding end-to-end with the DND.

Same two tiers as tests/test_dinov2_encoder.py:

  * Unit tests (default): `torch.hub.load` monkeypatched to a tiny STUB
    backbone with the same (B,3,H,W) -> (B,embed_dim) contract. Fast, no
    weights, no network. These cover everything we actually wrote -- the
    channel adapter, the resize/normalise pipeline, the param-group split,
    and NEC's full setup()/step()/checkpoint loop around it.

  * Real-architecture tests: skipped unless NEC_DINOV2_REAL=1. These build
    the genuine `dinov2_vits14` (22M params, ~300 tokens of attention) from
    torch.hub -- either a local clone via NEC_DINOV2_REPO_DIR, or the
    ~/.cache/torch/hub copy, which torch.hub reuses without touching the
    network. They pin the things a stub structurally cannot: that a real
    DINOv2 state_dict loads with strict=True through this class, that the
    patch grid tolerates every documented image_size, and that gradients
    reach the ViT's own transformer blocks. Set DINOV2_WEIGHTS as well to
    additionally load the pretrained .pth.

Deliberately NOT covered: whether NEC scores better with DINOv2 than with
NatureEmbedding on Atari. That is the experiment this code exists to run.
"""
from __future__ import annotations

import functools
import os

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.nec import NECAlgorithm
from src.networks import DINOv2Embedding, NatureEmbedding


OBS_SHAPE = (4, 84, 84)          # Atari: 4 stacked grayscale frames
EMBEDDING_DIM = 16
STUB_EMBED_DIM = 32
STUB_IMAGE_SIZE = 28             # 2 * 14, keeps the stub path cheap

REAL_MODEL = "dinov2_vits14"
REAL_EMBED_DIM = 384

real_only = pytest.mark.skipif(
    os.environ.get("NEC_DINOV2_REAL") != "1",
    reason="set NEC_DINOV2_REAL=1 to build the genuine DINOv2 ViT "
           "(NEC_DINOV2_REPO_DIR=/path/to/facebookresearch/dinov2 if offline)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubViT(nn.Module):
    """(B, 3, H, W) -> (B, STUB_EMBED_DIM). Stands in for the real DINOv2 ViT.

    Same contract as tests/test_nec_embedding_network.py's stub. Spatially
    pooled, so it is insensitive to image_size -- which is exactly why the
    resize and patch-grid behaviour needs the real-architecture tier below.
    """

    embed_dim = STUB_EMBED_DIM

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, STUB_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-1, -2)))


@pytest.fixture
def stub_hub(monkeypatch):
    monkeypatch.setattr("torch.hub.load", lambda *a, **k: _StubViT())


def _stub_factory(**kwargs):
    """`DINOv2Embedding` pre-bound like a Hydra `_partial_` would."""
    kwargs.setdefault("image_size", STUB_IMAGE_SIZE)
    return functools.partial(DINOv2Embedding, **kwargs)


def _real_embedding(**kwargs) -> DINOv2Embedding:
    kwargs.setdefault("model_name", REAL_MODEL)
    kwargs.setdefault("repo_dir", os.environ.get("NEC_DINOV2_REPO_DIR"))
    return DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, **kwargs)


class _MockAtariEnv:
    """Duck-typed stand-in for a TorchRL EnvBase.

    `NECAlgorithm.setup()` only reads observation_spec / action_spec /
    batch_size off the proof env -- it never resets or steps it.
    """

    def __init__(self, obs_shape=OBS_SHAPE, num_actions=2):
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=obs_shape, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=num_actions)
        self.batch_size = torch.Size([])


def _make_nec(embedding_network, *, num_actions=2, T=8, lr=1e-2, obs_shape=OBS_SHAPE):
    """A NECAlgorithm wired through the real `setup()` path."""
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
    alg.setup(lambda: _MockAtariEnv(obs_shape, num_actions))
    return alg


def _episode_batch(num_actions=2, T=8, obs_shape=OBS_SHAPE):
    """One env, two complete episodes -- enough for step() to write the DND.

    Actions are round-robin rather than random on purpose: `_gradient_step`
    skips any action whose table still holds <= k entries, so a draw that
    happened to starve one action would make every "the network was trained"
    assertion pass vacuously. T=8 over 2 actions puts 4 entries in each,
    clear of the k=2 gate in `_make_nec`.
    """
    dones = torch.zeros(1, T, dtype=torch.bool)
    dones[0, T // 2 - 1] = True
    dones[0, T - 1] = True
    return TensorDict(
        {
            "pixels": torch.rand(1, T, *obs_shape),
            "action": (torch.arange(T) % num_actions).reshape(1, T),
            "next": TensorDict(
                {
                    "pixels":     torch.rand(1, T, *obs_shape),
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
# 1. Channel adapter -- the pretrained features must be usable at step 0
# ---------------------------------------------------------------------------

def test_channel_adapter_starts_as_grayscale_to_rgb(stub_hub):
    """The 4->3 adapter must begin as "mean of the frame stack, replicated".

    A default-initialised nn.Conv2d emits channels of arbitrary scale and
    sign, so the ViT's first forward pass sees input nowhere near the
    ImageNet distribution its normalisation assumes -- and a pretrained
    representation that only becomes useful after the adapter has been
    *learned* is not a pretrained representation. This init is the thing
    that makes DINOv2 worth anything before the first gradient step.
    """
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    x = torch.rand(3, *OBS_SHAPE)                  # [0, 1], as ToTensorImage gives
    with torch.no_grad():
        out = net.channel_adapter(x)

    assert out.shape == (3, 3, 84, 84)
    expected = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected, atol=1e-6)
    # R == G == B: a plain grayscale image, in-distribution for the ViT.
    assert torch.allclose(out[:, 0], out[:, 1]) and torch.allclose(out[:, 0], out[:, 2])
    # Still a valid image range -- ImageNet normalisation is meaningful.
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_channel_adapter_is_trainable_not_a_fixed_transform(stub_hub):
    """Initialised, not frozen: the adapter must still be free to learn."""
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    assert isinstance(net.channel_adapter, nn.Conv2d)
    assert all(p.requires_grad for p in net.channel_adapter.parameters())

    net(torch.rand(2, *OBS_SHAPE)).sum().backward()
    assert net.channel_adapter.weight.grad is not None
    assert net.channel_adapter.weight.grad.abs().sum() > 0


def test_rgb_observations_skip_the_adapter(stub_hub):
    """A 3-channel env (MFEC-style RGB) needs no adaptation at all."""
    net = DINOv2Embedding((3, 84, 84), EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE)

    assert isinstance(net.channel_adapter, nn.Identity)
    assert net(torch.rand(2, 3, 84, 84)).shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 2. Param groups -- discriminative learning rate for a pretrained trunk
# ---------------------------------------------------------------------------

def test_param_groups_split_backbone_from_head(stub_hub):
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                          backbone_lr_scale=0.1)
    groups = net.param_groups(1e-3)

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-4)     # backbone: scaled down
    assert groups[1]["lr"] == pytest.approx(1e-3)     # adapter + head: base

    grouped = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert grouped == trainable, (
        "param_groups must cover exactly the trainable parameters -- a "
        "parameter missing from every group is silently never optimised"
    )
    assert {id(p) for p in net.backbone.parameters()} == {
        id(p) for p in groups[0]["params"]
    }


def test_param_groups_scale_of_one_is_a_uniform_learning_rate(stub_hub):
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                          backbone_lr_scale=1.0)
    assert [g["lr"] for g in net.param_groups(1e-3)] == [
        pytest.approx(1e-3), pytest.approx(1e-3)
    ]


def test_param_groups_drop_the_backbone_when_frozen(stub_hub):
    """freeze_backbone=True must not hand RMSProp an all-frozen group."""
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=STUB_IMAGE_SIZE,
                          freeze_backbone=True)
    groups = net.param_groups(1e-3)

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert {id(p) for g in groups for p in g["params"]} == {
        id(p) for m in (net.channel_adapter, net.head) for p in m.parameters()
    }


def test_setup_builds_the_optimizer_from_param_groups(stub_hub):
    """`NECAlgorithm.setup()` must honour the split, not flatten it."""
    alg = _make_nec(_stub_factory(backbone_lr_scale=0.1), lr=1e-3)

    assert len(alg.optimizer.param_groups) == 2
    assert [g["lr"] for g in alg.optimizer.param_groups] == [
        pytest.approx(1e-4), pytest.approx(1e-3)
    ]
    assert {id(p) for g in alg.optimizer.param_groups for p in g["params"]} == {
        id(p) for p in alg.embedding_net.parameters() if p.requires_grad
    }


def test_networks_without_param_groups_are_unchanged():
    """Regression: NatureEmbedding must still get one flat parameter list."""
    alg = _make_nec(NatureEmbedding, lr=1e-3)

    assert not hasattr(alg.embedding_net, "param_groups")
    assert len(alg.optimizer.param_groups) == 1
    assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert {id(p) for p in alg.optimizer.param_groups[0]["params"]} == {
        id(p) for p in alg.embedding_net.parameters()
    }


# ---------------------------------------------------------------------------
# 3. NEC actually finetunes it -- the difference from MFEC's frozen encoder
# ---------------------------------------------------------------------------

def test_step_trains_the_backbone_end_to_end(stub_hub):
    """One `step()` must move the ViT's own parameters, not just the head.

    "Not frozen" is not the same claim as "trained": the backbone could be
    trainable and still never receive a gradient (wrong optimizer param
    list, a detach in the pipeline, a zero-lr group). This asserts the
    backbone weights actually change under NEC's real loop.
    """
    torch.manual_seed(0)
    alg = _make_nec(_stub_factory(backbone_lr_scale=0.1), lr=1e-2)

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    metrics = alg.step(_episode_batch())

    assert sum(alg.dnd._sizes) > 0, "step() stored nothing -- assertions are vacuous"
    assert metrics["train/updates"] > 0, "no gradient update ran"

    moved = {
        n for n, p in alg.embedding_net.named_parameters()
        if not torch.equal(p.detach(), before[n])
    }
    assert any(n.startswith("backbone.") for n in moved), (
        "the DINOv2 backbone was not updated -- this is the frozen-MFEC "
        "behaviour, and the NEC variant exists specifically to avoid it"
    )
    assert any(n.startswith("head.") for n in moved)
    assert any(n.startswith("channel_adapter.") for n in moved)


def test_frozen_backbone_trains_only_the_head(stub_hub):
    """The opt-in freeze must still leave a learnable network."""
    torch.manual_seed(0)
    alg = _make_nec(_stub_factory(freeze_backbone=True), lr=1e-2)

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    alg.step(_episode_batch())

    moved = {
        n for n, p in alg.embedding_net.named_parameters()
        if not torch.equal(p.detach(), before[n])
    }
    assert moved, "freezing the backbone must not freeze the whole network"
    assert not any(n.startswith("backbone.") for n in moved)


def test_gradient_reaches_the_backbone_through_the_dnd_kernel(stub_hub):
    """Gradients must arrive via the kernel distance term, not a shortcut.

    NEC's loss touches the network only through ||h - h_i||^2 against frozen
    stored keys, so this is the seam where a detach would silently disable
    finetuning while every shape check still passed.
    """
    torch.manual_seed(0)
    alg = _make_nec(_stub_factory())
    alg.step(_episode_batch())          # populate DND + replay buffer

    alg.optimizer.zero_grad()
    result = alg._gradient_step()
    assert result is not None, "gradient step skipped -- assertions are vacuous"

    backbone_grads = [
        p.grad for p in alg.embedding_net.backbone.parameters() if p.requires_grad
    ]
    assert backbone_grads and all(g is not None for g in backbone_grads)
    assert any(g.abs().sum() > 0 for g in backbone_grads)


# ---------------------------------------------------------------------------
# 4. Checkpointing -- param groups must survive a resume
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_preserves_weights_and_param_groups(stub_hub):
    """Resume rebuilds the optimizer, so the grouping must be reproducible.

    `_load_training_state` calls `_build_optimizer()` and then
    `load_state_dict`, which raises on a group-count mismatch. A DINOv2 run
    that rebuilt a flat optimizer would fail to resume outright -- or, worse,
    resume with the backbone silently promoted to the head's learning rate.
    """
    torch.manual_seed(0)
    src = _make_nec(_stub_factory(backbone_lr_scale=0.1), lr=1e-2)
    src.step(_episode_batch())
    state = src._get_training_state()

    torch.manual_seed(1)
    dst = _make_nec(_stub_factory(backbone_lr_scale=0.1), lr=1e-2)
    obs = torch.rand(3, *OBS_SHAPE)
    assert not torch.allclose(dst.embedding_net(obs), src.embedding_net(obs)), (
        "the two networks start identical -- the restore assertion below "
        "would pass without restoring anything"
    )

    dst._load_training_state(state)

    assert torch.allclose(dst.embedding_net(obs), src.embedding_net(obs), atol=1e-6)
    assert len(dst.optimizer.param_groups) == 2
    assert [g["lr"] for g in dst.optimizer.param_groups] == [
        pytest.approx(1e-3), pytest.approx(1e-2)
    ]


# ---------------------------------------------------------------------------
# 5. Hydra composition -- the CLI override is the intended entry point
# ---------------------------------------------------------------------------

def test_experiment_config_selects_the_finetunable_dinov2():
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg("nec/mspacman_dinov2", ["logger=[]"])

    assert cfg.algorithm.embedding_network._target_ == "src.networks.DINOv2Embedding"
    assert cfg.algorithm.embedding_network.freeze_backbone is False
    assert cfg.algorithm.embedding_network.weights_path, "weights_path unresolved"
    # The run directory must not collide with the NatureEmbedding arm.
    assert cfg.run.encoder == "dinov2"
    assert cfg.run.name == "nec_mspacman_dinov2_seed42"
    # Same env as nec/mspacman.yaml -> the encoder is the only variable.
    assert cfg.environment.name == "ALE/MsPacman-v5"
    assert any("CatFrames" in t["_target_"] for t in cfg.environment.transforms)

    alg = instantiate(cfg.algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)


def test_experiment_config_builds_the_network(stub_hub, tmp_path):
    from tests.conftest import load_experiment_cfg

    ckpt = tmp_path / "stub.pth"
    torch.save(_StubViT().state_dict(), ckpt)

    cfg = load_experiment_cfg(
        "nec/mspacman_dinov2",
        [
            "logger=[]",
            f"algorithm.embedding_network.weights_path={ckpt}",
            f"algorithm.embedding_network.image_size={STUB_IMAGE_SIZE}",
        ],
    )
    alg = instantiate(cfg.algorithm, device=None)
    net = alg._make_embedding_network(OBS_SHAPE, cfg.algorithm.embedding_dim)

    assert isinstance(net, DINOv2Embedding)
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, cfg.algorithm.embedding_dim)
    assert all(p.requires_grad for p in net.parameters())
    assert net.backbone_lr_scale == pytest.approx(0.1)


def test_bad_image_size_fails_loudly(stub_hub):
    with pytest.raises(AssertionError):
        DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100)   # 100 % 14 != 0


# ---------------------------------------------------------------------------
# 6. Real DINOv2 architecture -- opt-in (NEC_DINOV2_REAL=1)
# ---------------------------------------------------------------------------

@real_only
@pytest.mark.parametrize("image_size", [98, 112, 224])
def test_real_vit_forward_shape_at_every_documented_image_size(image_size):
    """98 / 112 / 224 are the sizes dinov2_finetune.yaml advertises.

    The patch grid is image_size/14 and DINOv2 interpolates its position
    embeddings to match, so a size the class accepts but the ViT rejects
    would only surface here -- the stub is spatially pooled and cannot see
    the difference.
    """
    net = _real_embedding(image_size=image_size)
    out = net(torch.rand(2, *OBS_SHAPE))

    assert out.shape == (2, EMBEDDING_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert net.backbone.embed_dim == REAL_EMBED_DIM


@real_only
def test_real_state_dict_loads_strictly_and_changes_the_output(tmp_path):
    """The weights_path path must load a genuine DINOv2 state_dict.

    `strict=True` is the assertion that matters: it fails on any key or
    shape mismatch, which is what would happen if this class ever wrapped
    or renamed the backbone. Checking the output *changes* proves the
    weights are actually used rather than loaded and discarded.
    """
    donor = _real_embedding()
    ckpt = tmp_path / "dinov2_vits14_pretrain.pth"
    torch.save(donor.backbone.state_dict(), ckpt)

    obs = torch.rand(2, *OBS_SHAPE)
    torch.manual_seed(7)
    fresh = _real_embedding()
    before = fresh(obs)

    loaded = _real_embedding(weights_path=str(ckpt))
    # Copy the non-backbone parts across so the ONLY difference is the ViT.
    loaded.head.load_state_dict(fresh.head.state_dict())
    loaded.channel_adapter.load_state_dict(fresh.channel_adapter.state_dict())

    assert not torch.allclose(loaded(obs), before, atol=1e-5), (
        "loading a different backbone left the embedding unchanged"
    )
    for (n, a), (_, b) in zip(
        loaded.backbone.state_dict().items(), donor.backbone.state_dict().items()
    ):
        assert torch.equal(a, b), f"backbone parameter {n} did not round-trip"


@real_only
def test_real_vit_transformer_blocks_receive_gradients():
    """Finetuning has to reach the attention weights, not just the head."""
    net = _real_embedding(image_size=98)
    net(torch.rand(2, *OBS_SHAPE)).square().mean().backward()

    named = dict(net.backbone.named_parameters())
    for key in ("blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight",
                "patch_embed.proj.weight", "cls_token"):
        grad = named[key].grad
        assert grad is not None, f"no gradient reached backbone.{key}"
        assert grad.abs().sum() > 0, f"gradient at backbone.{key} is all zero"


@real_only
def test_real_vit_runs_end_to_end_through_nec():
    """setup() -> step() with the genuine ViT: shapes, DND writes, updates."""
    torch.manual_seed(0)
    alg = _make_nec(
        functools.partial(
            DINOv2Embedding,
            model_name=REAL_MODEL,
            repo_dir=os.environ.get("NEC_DINOV2_REPO_DIR"),
            image_size=98,
            backbone_lr_scale=0.1,
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
    ), "the real ViT was not updated by NEC's loop"

    q = alg._dnd_policy(torch.rand(2, *OBS_SHAPE))
    assert q.shape == (2, 2)


@real_only
@pytest.mark.skipif(
    not os.environ.get("DINOV2_WEIGHTS"),
    reason="set DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth to run",
)
def test_pretrained_checkpoint_loads():
    """The actual released .pth, not a re-saved random init."""
    net = _real_embedding(weights_path=os.environ["DINOV2_WEIGHTS"], image_size=98)
    out = net(torch.rand(2, *OBS_SHAPE))

    assert out.shape == (2, EMBEDDING_DIM)
    assert torch.isfinite(out).all()
