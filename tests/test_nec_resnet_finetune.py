"""Tests for NEC's finetunable ImageNet ResNet embedding network.

``src/networks.py::ResNetEmbedding``, selected by
``algorithm/embedding_network=resnet_finetune``. The trainable counterpart to
MFEC's frozen ``src/encoders/resnet_encoder.py::ResNetEncoder`` (covered by
tests/test_resnet_encoder.py).

ONE TIER, unlike tests/test_nec_{clip,mae}_finetune.py. Those stub their
backbone because ``open_clip_torch`` / ``timm`` are optional dependencies and
because a genuine ViT-B is expensive to build. Neither applies here: torchvision
is a core dependency, and ``pretrained=False`` builds the real ``resnet18``
architecture with no download and no meaningful cost. So every test below runs
against the actual backbone, and there is no stub to drift out of sync with it.
``pretrained=True`` (the config default) is exercised nowhere -- it would hit
the network.

WHAT THIS FILE IS REALLY FOR. Sections 4 and 5 are the point. ResNet is the only
arm of the encoder ablation whose backbone carries BatchNorm, NEC keeps the
embedding network in ``train()`` mode, and NEC embeds at four different batch
sizes -- including 1, inside ``BaseTrainer.evaluate``. Batch-dependent phi
breaks the DND's premise that keys stay comparable between the phase that writes
them and the phase that reads them, and the failure is silent: it costs eval
score without raising anything. ``freeze_batchnorm=True`` is the default that
prevents it, ``train()`` is overridden to keep it that way, and the tests here
pin both, plus the batch-independence property they exist to buy.

Deliberately NOT covered: whether NEC scores better with ResNet than with
``nature``, ``dinov2_finetune``, ``clip_finetune`` or ``mae_finetune``. That is
the experiment.
"""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.nec import NECAlgorithm
from src.networks import _IMAGENET_MEAN, _IMAGENET_STD, ResNetEmbedding


OBS_SHAPE = (4, 84, 84)          # Atari: 4 stacked grayscale frames
EMBEDDING_DIM = 16
POOL_DIM = 512                   # resnet18 / resnet34 pooled width
SMALL = 64                       # cheap resize target; 224 only where it matters


def _net(**kwargs) -> ResNetEmbedding:
    """A real resnet18, untrained (no download), at a cheap resolution."""
    kwargs.setdefault("pretrained", False)
    kwargs.setdefault("image_size", SMALL)
    return ResNetEmbedding(OBS_SHAPE, EMBEDDING_DIM, **kwargs)


def _factory(**kwargs):
    """``ResNetEmbedding`` pre-bound like a Hydra ``_partial_`` would."""
    kwargs.setdefault("pretrained", False)
    kwargs.setdefault("image_size", SMALL)
    return functools.partial(ResNetEmbedding, **kwargs)


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
# 1. Backbone construction
# ---------------------------------------------------------------------------

def test_torchvision_is_not_a_module_level_import_of_networks():
    """The import is deferred into __init__.

    Not for optionality -- torchvision is a core dependency, unlike open_clip
    and timm -- but for import cost: src/networks.py is imported by
    src/algorithms/nec.py and by every DQN/DDPG/A2C config, none of which need
    torchvision.models.
    """
    import ast
    import inspect

    import src.networks as networks

    tree = ast.parse(inspect.getsource(networks))
    module_level = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for name in (
            [a.name for a in n.names]
            if isinstance(n, ast.Import)
            else [n.module or ""]
        )
        if name.startswith("torchvision")
    ]
    assert module_level == [], (
        "torchvision must not be imported at module scope in src/networks.py"
    )


def test_the_imagenet_classifier_is_dropped():
    net = _net()
    assert isinstance(net.backbone.fc, nn.Identity)
    assert net.pool_dim == POOL_DIM
    assert net.head.in_features == POOL_DIM
    assert net.head.out_features == EMBEDDING_DIM
    # The 1000-way head is 512*1000 params that would otherwise ride in every
    # RMSProp state and every checkpoint.
    assert not any(p.shape[0] == 1000 for p in net.parameters())


def test_pool_dim_is_read_off_the_backbone_not_hardcoded():
    """resnet50 pools to 2048; nothing downstream should need changing."""
    net = ResNetEmbedding(
        OBS_SHAPE, EMBEDDING_DIM,
        model_name="resnet50", pretrained=False, image_size=SMALL,
    )
    assert net.pool_dim == 2048
    assert net.head.in_features == 2048
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, EMBEDDING_DIM)


def test_a_non_resnet_torchvision_model_is_rejected():
    """`vgg16` has `classifier`, not `fc`; without the guard this would be a
    bare AttributeError three frames deep."""
    with pytest.raises(ValueError, match="not a torchvision ResNet"):
        ResNetEmbedding(
            OBS_SHAPE, EMBEDDING_DIM,
            model_name="vgg16", pretrained=False, image_size=SMALL,
        )


def test_local_weights_path_wins_over_the_download(tmp_path):
    """The offline-cluster path: a torch.save(state_dict) loads with strict=True."""
    import torchvision.models as tvm

    reference = tvm.get_model("resnet18", weights=None)
    for p in reference.parameters():
        with torch.no_grad():
            p.fill_(0.017)
    path = tmp_path / "resnet18.pth"
    torch.save(reference.state_dict(), path)

    net = _net(weights_path=str(path))
    # conv1 survived the load; fc was replaced afterwards, so it is not checked.
    assert torch.allclose(
        net.backbone.conv1.weight, torch.full_like(net.backbone.conv1.weight, 0.017)
    )


def test_a_state_dict_wrapper_is_unwrapped(tmp_path):
    """Mirrors ResNetEncoder: checkpoints often arrive as {"state_dict": ...}."""
    import torchvision.models as tvm

    reference = tvm.get_model("resnet18", weights=None)
    path = tmp_path / "wrapped.pth"
    torch.save({"state_dict": reference.state_dict()}, path)

    net = _net(weights_path=str(path))
    assert torch.allclose(net.backbone.conv1.weight, reference.conv1.weight)


# ---------------------------------------------------------------------------
# 2. image_size guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1, 16, 31])
def test_image_size_below_the_downsampling_factor_is_rejected(bad):
    with pytest.raises(ValueError, match="too small"):
        ResNetEmbedding(OBS_SHAPE, EMBEDDING_DIM, pretrained=False, image_size=bad)


def test_the_size_guard_explains_the_arithmetic_and_names_the_default():
    with pytest.raises(ValueError) as exc:
        ResNetEmbedding(OBS_SHAPE, EMBEDDING_DIM, pretrained=False, image_size=16)
    msg = str(exc.value)
    assert "32" in msg and "224" in msg


@pytest.mark.parametrize("size", [32, 64, 224])
def test_valid_sizes_build_and_forward(size):
    net = ResNetEmbedding(
        OBS_SHAPE, EMBEDDING_DIM, pretrained=False, image_size=size
    )
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 3. Input pipeline
# ---------------------------------------------------------------------------

def test_channel_adapter_starts_as_grayscale_to_rgb():
    """Init is mean-over-frames replicated to R=G=B, NOT random.

    Same rationale as DINOv2Embedding: a default-initialised Conv2d emits
    channels of a different scale and sign, so the trunk's first forward sees
    out-of-distribution input and the ImageNet pretraining is worth nothing at
    step 0 -- which defeats the point of using a PVM.
    """
    net = _net()
    assert isinstance(net.channel_adapter, nn.Conv2d)
    assert torch.allclose(
        net.channel_adapter.weight, torch.full_like(net.channel_adapter.weight, 0.25)
    )
    assert torch.count_nonzero(net.channel_adapter.bias) == 0

    x = torch.rand(2, *OBS_SHAPE)
    out = net.channel_adapter(x)
    expected = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rgb_observations_skip_the_adapter():
    net = ResNetEmbedding(
        (3, 84, 84), EMBEDDING_DIM, pretrained=False, image_size=SMALL
    )
    assert isinstance(net.channel_adapter, nn.Identity)


def test_imagenet_stats_are_used():
    net = _net()
    assert torch.allclose(net._mean.flatten(), torch.tensor(_IMAGENET_MEAN))
    assert torch.allclose(net._std.flatten(), torch.tensor(_IMAGENET_STD))


def test_frames_are_resized_to_image_size():
    net = _net(image_size=96)
    seen = {}
    net.backbone.register_forward_pre_hook(
        lambda m, args: seen.update(shape=args[0].shape)
    )
    net(torch.rand(2, *OBS_SHAPE))
    assert seen["shape"][-2:] == (96, 96)


def test_the_whole_frame_is_resized_not_centre_cropped():
    """A centre crop would cut away the left and right of the maze.

    Fed a frame that is bright only at the far left, the trunk input must still
    carry that energy -- a Resize+CenterCrop pipeline would have discarded it.
    """
    net = _net(image_size=SMALL)
    seen = {}
    net.backbone.register_forward_pre_hook(
        lambda m, args: seen.update(x=args[0].detach().clone())
    )
    frame = torch.zeros(1, *OBS_SHAPE)
    frame[..., :8] = 1.0                      # leftmost columns only
    net(frame)
    x = seen["x"]
    left_edge = x[..., : SMALL // 8]
    assert left_edge.std() > 0, "the left edge was cropped away"


# ---------------------------------------------------------------------------
# 4. BatchNorm: the mode invariant
# ---------------------------------------------------------------------------

def _bns(net):
    return [
        m for m in net.backbone.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


def test_the_backbone_really_does_carry_batchnorm():
    """Guards the premise of every test in this section (and of the config note)."""
    assert len(_bns(_net())) > 0


def test_batchnorm_is_in_eval_mode_after_construction():
    net = _net()
    assert all(not m.training for m in _bns(net))


def test_train_mode_does_not_release_batchnorm():
    """The load-bearing override.

    nn.Module.train() recurses into every submodule, so a single net.train()
    anywhere -- trainer, test, resume path -- would otherwise put the
    BatchNorms back into batch-statistics mode and silently reintroduce the
    failure the default exists to prevent.
    """
    net = _net()
    net.train()
    assert net.training is True, "the module itself must still be in train mode"
    assert all(not m.training for m in _bns(net))


def test_train_mode_is_restored_after_an_eval_round_trip():
    net = _net()
    net.eval()
    net.train()
    assert net.training is True
    assert all(not m.training for m in _bns(net))


def test_the_non_default_is_batch_dependent_and_warns():
    with pytest.warns(UserWarning, match="freeze_batchnorm=False"):
        net = _net(freeze_batchnorm=False)
    net.train()
    assert all(m.training for m in _bns(net))


def test_the_warning_names_the_evaluation_batch_size():
    """It must say *why*, not just *that* -- batch=1 in evaluate() is the sharp end."""
    with pytest.warns(UserWarning) as record:
        _net(freeze_batchnorm=False)
    msg = str(record[0].message)
    assert "batch=1" in msg and "DND" in msg


def test_the_default_does_not_warn(recwarn):
    _net()
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# 5. BatchNorm: the property the mode invariant buys
# ---------------------------------------------------------------------------

def test_phi_is_batch_independent_under_the_default():
    """THE test of this file.

    NEC writes DND keys during collection (batch=num_envs) and episode
    re-embedding (batch<=256), and reads them during evaluation (batch=1). If
    the same frame embeds differently in different phases, every stored key is
    stale the moment it is written.
    """
    net = _net()
    net.train()                                  # NEC's real mode
    frames = torch.rand(8, *OBS_SHAPE)

    alone = net(frames[:1])
    in_batch = net(frames)[:1]
    assert torch.allclose(alone, in_batch, atol=1e-5)

    # ... and the batch composition itself must not matter either.
    other_batch = net(torch.cat([frames[:1], torch.rand(15, *OBS_SHAPE)]))[:1]
    assert torch.allclose(alone, other_batch, atol=1e-5)


def test_phi_is_batch_dependent_without_it():
    """The failure the default prevents, pinned so the default cannot be
    quietly flipped without this test going red."""
    with pytest.warns(UserWarning):
        net = _net(freeze_batchnorm=False)
    net.train()
    frames = torch.rand(8, *OBS_SHAPE)
    alone = net(frames[:1])
    in_batch = net(frames)[:1]
    assert not torch.allclose(alone, in_batch, atol=1e-5)


def test_running_statistics_do_not_drift_during_training():
    """eval() mode means the forward pass never updates running_mean/var, so
    the ImageNet statistics stay exactly as the checkpoint left them."""
    net = _net()
    net.train()
    bn = _bns(net)[0]
    before_mean = bn.running_mean.clone()
    before_var = bn.running_var.clone()
    before_batches = int(bn.num_batches_tracked)

    for _ in range(3):
        net(torch.rand(4, *OBS_SHAPE)).sum().backward()

    assert torch.equal(bn.running_mean, before_mean)
    assert torch.equal(bn.running_var, before_var)
    assert int(bn.num_batches_tracked) == before_batches


def test_batchnorm_affine_parameters_still_train():
    """freeze_batchnorm freezes the STATISTICS, not the parameters -- it is not
    a back-door freeze_backbone."""
    net = _net()
    bn = _bns(net)[0]
    assert bn.weight.requires_grad and bn.bias.requires_grad

    net(torch.rand(4, *OBS_SHAPE)).sum().backward()
    assert bn.weight.grad is not None
    assert torch.count_nonzero(bn.weight.grad) > 0


def test_convolutions_still_train():
    net = _net()
    net(torch.rand(4, *OBS_SHAPE)).sum().backward()
    assert net.backbone.conv1.weight.grad is not None
    assert torch.count_nonzero(net.backbone.conv1.weight.grad) > 0


# ---------------------------------------------------------------------------
# 6. Output contract (NECEmbeddingNetwork clauses 1-3)
# ---------------------------------------------------------------------------

def test_forward_shape_and_dtype():
    net = _net()
    out = net(torch.rand(5, *OBS_SHAPE))
    assert out.shape == (5, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_leading_dims_are_flattened():
    """DNDPolicy and _gradient_step both hand in (E, T, C, H, W)-shaped data."""
    net = _net()
    assert net(torch.rand(2, 3, *OBS_SHAPE)).shape == (6, EMBEDDING_DIM)


def test_output_is_not_prenormalised():
    """Clause 3: NEC L2-normalises downstream, and a module that pre-normalises
    makes that a no-op which also hides gradient scale."""
    net = _net()
    out = net(torch.rand(8, *OBS_SHAPE))
    norms = out.norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    assert (norms > 0).all(), "an all-zero row would collapse every DND distance"


def test_normalize_features_puts_the_head_input_on_the_unit_sphere():
    net = _net(normalize_features=True)
    seen = {}
    net.head.register_forward_pre_hook(
        lambda m, args: seen.update(x=args[0].detach().clone())
    )
    net(torch.rand(4, *OBS_SHAPE))
    assert torch.allclose(
        seen["x"].norm(dim=-1), torch.ones(4), atol=1e-5
    )


def test_normalize_features_false_leaves_the_head_input_unscaled():
    net = _net(normalize_features=False)
    seen = {}
    net.head.register_forward_pre_hook(
        lambda m, args: seen.update(x=args[0].detach().clone())
    )
    net(torch.rand(4, *OBS_SHAPE))
    assert not torch.allclose(seen["x"].norm(dim=-1), torch.ones(4), atol=1e-3)


# ---------------------------------------------------------------------------
# 7. Trainability and param groups
# ---------------------------------------------------------------------------

def test_all_parameters_are_trainable_by_default():
    """Clause 2: unlike MFEC's frozen Encoder protocol, a frozen parameter here
    is silently a dead parameter."""
    assert all(p.requires_grad for p in _net().parameters())


def test_freeze_backbone_leaves_the_head_trainable():
    net = _net(freeze_backbone=True)
    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.head.parameters())
    assert all(p.requires_grad for p in net.channel_adapter.parameters())


def test_param_groups_split_backbone_from_head():
    net = _net(backbone_lr_scale=0.25)
    groups = net.param_groups(1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(2.5e-4)
    assert groups[1]["lr"] == pytest.approx(1e-3)

    grouped = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert grouped == trainable, "the groups must cover exactly the trainable params"


def test_param_groups_include_the_batchnorm_affine_parameters():
    """They are trainable, so they must be in a group -- running stats are
    buffers, not parameters, so no group ever contained them."""
    net = _net()
    backbone_group = net.param_groups(1e-3)[0]["params"]
    bn = _bns(net)[0]
    assert any(p is bn.weight for p in backbone_group)
    assert not any(b is bn.running_mean for b in backbone_group)


def test_param_groups_drop_the_backbone_when_frozen():
    net = _net(freeze_backbone=True)
    groups = net.param_groups(1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# 8. End-to-end through NEC
# ---------------------------------------------------------------------------

def test_setup_builds_the_optimizer_from_param_groups():
    alg = _make_nec(_factory(backbone_lr_scale=0.5), lr=1e-3)
    assert isinstance(alg.optimizer, torch.optim.RMSprop)
    assert len(alg.optimizer.param_groups) == 2
    assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert alg.optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)


def test_step_trains_the_trunk_end_to_end():
    alg = _make_nec(_factory(), lr=1e-1)
    before = alg.embedding_net.backbone.conv1.weight.detach().clone()
    alg.step(_episode_batch())
    after = alg.embedding_net.backbone.conv1.weight
    assert not torch.allclose(before, after), "the trunk did not move"


def test_gradient_reaches_the_trunk_through_the_dnd_kernel():
    """The gradient path is Q -> kernel weights -> ||h - h_i||^2 -> h -> trunk."""
    alg = _make_nec(_factory(), lr=1e-2)
    alg.step(_episode_batch())
    grads = [
        p.grad for p in alg.embedding_net.backbone.parameters() if p.grad is not None
    ]
    assert grads, "no backbone parameter received a gradient"
    assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_batchnorm_stays_frozen_across_a_real_nec_step():
    """setup() and step() must not release the invariant."""
    alg = _make_nec(_factory())
    alg.step(_episode_batch())
    assert all(not m.training for m in _bns(alg.embedding_net))


def test_checkpoint_roundtrip_preserves_weights_and_running_stats():
    """Running statistics are persistent buffers, so state_dict() covers them --
    which is what keeps this network inside NECEmbeddingNetwork clause 4 (no
    state beyond state_dict) despite the frozen BatchNorm."""
    alg = _make_nec(_factory(), lr=1e-1)
    alg.step(_episode_batch())
    state = alg._get_training_state()

    fresh = _make_nec(_factory(), lr=1e-1)
    fresh._load_training_state(state)

    for (ka, va), (kb, vb) in zip(
        alg.embedding_net.state_dict().items(),
        fresh.embedding_net.state_dict().items(),
    ):
        assert ka == kb
        assert torch.equal(va, vb), f"{ka} did not survive the round trip"

    assert len(fresh.optimizer.param_groups) == len(alg.optimizer.param_groups)
    assert all(not m.training for m in _bns(fresh.embedding_net))


# ---------------------------------------------------------------------------
# 9. Hydra composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "experiment,game,env_name",
    [
        ("nec/mspacman_resnet", "mspacman", "ALE/MsPacman-v5"),
        ("nec/qbert_resnet", "qbert", "ALE/Qbert-v5"),
        ("nec/frostbite_resnet", "frostbite", "ALE/Frostbite-v5"),
    ],
)
def test_experiment_configs_select_the_finetunable_resnet(experiment, game, env_name):
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(experiment, ["logger=[]"])

    assert cfg.algorithm.embedding_network._target_ == "src.networks.ResNetEmbedding"
    assert cfg.algorithm.embedding_network.freeze_backbone is False
    assert cfg.algorithm.embedding_network.freeze_batchnorm is True
    assert cfg.algorithm.embedding_network.model_name == "resnet18"
    assert cfg.algorithm.embedding_network.image_size == 224
    assert cfg.run.encoder == "resnet"
    assert cfg.run.name == f"nec_{game}_resnet_seed42"
    # Same env as nec/<game>.yaml -> the encoder is the only variable.
    assert cfg.environment.name == env_name
    assert any("CatFrames" in t["_target_"] for t in cfg.environment.transforms)

    alg = instantiate(cfg.algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)


def test_the_three_games_differ_only_in_the_game():
    """The ablation's premise. Anything that is not the env or the game name
    must be identical across the three configs."""
    from tests.conftest import load_experiment_cfg

    cfgs = [
        load_experiment_cfg(f"nec/{g}_resnet", ["logger=[]"])
        for g in ("mspacman", "qbert", "frostbite")
    ]
    ref = cfgs[0]
    for cfg in cfgs[1:]:
        assert cfg.algorithm.embedding_network == ref.algorithm.embedding_network
        assert cfg.algorithm.num_updates == ref.algorithm.num_updates
        assert cfg.algorithm.eps_end == ref.algorithm.eps_end
        assert cfg.algorithm.annealing_frames == ref.algorithm.annealing_frames
        assert cfg.algorithm.init_random_frames == ref.algorithm.init_random_frames
        assert cfg.algorithm.eval_eps == ref.algorithm.eval_eps
        assert cfg.trainer.total_frames == ref.trainer.total_frames
        assert cfg.trainer.num_envs == ref.trainer.num_envs
        assert cfg.trainer.num_eval_episodes == ref.trainer.num_eval_episodes
        assert cfg.trainer.eval_every_n_steps == ref.trainer.eval_every_n_steps


def test_the_resnet_arm_matches_the_clip_arm_everywhere_but_the_encoder():
    """Cross-arm parity: this is what makes a between-arm read a comparison of
    encoders rather than of trainer budgets."""
    from tests.conftest import load_experiment_cfg

    resnet = load_experiment_cfg("nec/mspacman_resnet", ["logger=[]"])
    clip = load_experiment_cfg("nec/mspacman_clip", ["logger=[]"])

    assert resnet.environment == clip.environment
    assert resnet.eval_environment == clip.eval_environment
    for key in (
        "num_updates", "eps_end", "annealing_frames", "init_random_frames",
        "eval_eps", "embedding_dim", "k", "n_step", "gamma",
    ):
        assert resnet.algorithm[key] == clip.algorithm[key], key
    for key in (
        "total_frames", "num_envs", "num_eval_episodes",
        "eval_every_n_steps", "log_every_n_steps",
    ):
        assert resnet.trainer[key] == clip.trainer[key], key


def test_experiment_config_builds_the_network():
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg("nec/mspacman_resnet", ["logger=[]"])
    alg = instantiate(cfg.algorithm, device=None)
    # pretrained=False so the test needs no network; everything else is the
    # config's own.
    net = alg._make_embedding_network(
        OBS_SHAPE, cfg.algorithm.embedding_dim, pretrained=False
    )

    assert isinstance(net, ResNetEmbedding)
    assert net.image_size == 224
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, cfg.algorithm.embedding_dim)
    assert all(p.requires_grad for p in net.parameters())
    assert net.backbone_lr_scale == pytest.approx(0.1)
    assert all(not m.training for m in _bns(net))


def test_cli_group_override_works_on_any_nec_experiment():
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(
        "nec/pong",
        ["logger=[]", "algorithm/embedding_network=resnet_finetune",
         "run.encoder=resnet"],
    )
    assert cfg.algorithm.embedding_network._target_ == "src.networks.ResNetEmbedding"
    assert cfg.run.encoder == "resnet"
