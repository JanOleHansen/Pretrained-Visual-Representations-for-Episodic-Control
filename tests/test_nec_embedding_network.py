"""Tests for NEC's pluggable, trainable embedding-network config group.

NEC's φ was always a `Callable` kwarg on `NECAlgorithm.__init__`, but the
concrete choice was baked into a nested YAML block inside
`configs/algorithm/nec.yaml` / `nec_atari.yaml`. It is now a first-class
Hydra config group (`configs/algorithm/embedding_network/`), so swapping is
`algorithm/embedding_network=<name>` on the CLI.

This is the trainable counterpart to MFEC's frozen `Encoder` system
(`src/encoders/`, covered by tests/test_mfec_encoder_refactor.py). The two
contracts are deliberately separate: MFEC's φ must be bit-exact and never
change (its QEC hash depends on it); NEC's φ is an `nn.Module` optimised
end-to-end by Adam. The contract is documented as
`src.networks.NECEmbeddingNetwork`.

Coverage:
  1. Shape/dtype contract for the standard `NatureEmbedding`.
  2. Gradient flow -- proof the encoder is genuinely learnable, i.e. one
     `_gradient_step()` produces nonzero grads and moves the parameters.
  3. Hydra composition regression -- both NEC algorithm configs still
     instantiate post-refactor and build the *same* architecture as the
     pre-refactor inline block.
  4. Config-swap smoke -- `setup()` + `step()` end-to-end with a deliberately
     different (test-only) factory, so pluggability is proven, not assumed.
  5. DINOv2 finetune scaffolding -- YAML composes and the factory builds a
     module with trainable params, against a STUB backbone (no downloads,
     mirroring tests/test_dinov2_encoder.py's approach for MFEC).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, LazyTensorStorage, TensorDictReplayBuffer

from src.algorithms.nec import DND, NECAlgorithm
from src.networks import DINOv2Embedding, NatureEmbedding

from tests.conftest import CONFIGS_DIR


OBS_SHAPE = (4, 84, 84)
EMBEDDING_DIM = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAtariEnv:
    """Duck-typed stand-in for a TorchRL EnvBase.

    `NECAlgorithm.setup()` only reads observation_spec / action_spec /
    batch_size off the proof env -- it never resets or steps it. Same trick
    as tests/test_mfec_encoder_refactor.py.
    """

    def __init__(self, obs_shape, num_actions):
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=obs_shape, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=num_actions)
        self.batch_size = torch.Size([])


def _tiny_linear_embedding(obs_shape, embedding_dim, *, hidden: int = 16):
    """Deliberately-different test-only embedding network.

    No convolutions at all -- flatten + MLP. Proves NEC does not assume the
    NatureDQN trunk (or even a 84x84 input) anywhere. Conforms to
    `NECEmbeddingNetwork`: two positional args, kwargs keyword-only,
    all params trainable, output unconstrained.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(int(math.prod(obs_shape)), hidden),
        nn.ReLU(),
        nn.Linear(hidden, embedding_dim),
    )


def _compose_algorithm_cfg(algorithm: str, extra_overrides=None):
    """Compose just the `algorithm` node of a train config."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    overrides = [
        f"algorithm={algorithm}",
        "environment=cartpole",
        *(extra_overrides or []),
    ]
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg.algorithm


# ---------------------------------------------------------------------------
# 1. Shape / dtype contract (NECEmbeddingNetwork clause 1)
# ---------------------------------------------------------------------------

def test_nature_embedding_shape_and_dtype():
    torch.manual_seed(0)
    net = NatureEmbedding(OBS_SHAPE, EMBEDDING_DIM)

    B = 5
    obs = torch.rand(B, *OBS_SHAPE)          # dummy pixel obs in [0, 1]
    out = net(obs)

    assert out.shape == (B, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_nature_embedding_output_is_not_prenormalised():
    """Clause 3: the module must not emit unit-norm rows itself.

    NEC L2-normalises downstream (DNDPolicy.forward / _gradient_step); a
    module that pre-normalises makes that a no-op and hides gradient scale.
    """
    torch.manual_seed(0)
    net = NatureEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    norms = net(torch.rand(8, *OBS_SHAPE)).norm(dim=-1)

    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    assert (norms > 0).all(), (
        "all-zero rows would survive F.normalize as zeros and collapse every "
        "DND kernel distance"
    )


def test_nature_embedding_params_are_all_trainable():
    """Clause 2: every parameter must arrive at the optimizer with grads on."""
    net = NatureEmbedding(OBS_SHAPE, EMBEDDING_DIM)
    params = list(net.parameters())

    assert params, "an embedding network with no parameters gives Adam an empty list"
    assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# 2. Gradient flow -- the encoder is genuinely learnable, not accidentally frozen
# ---------------------------------------------------------------------------

def _make_minimal_nec(embedding_network, obs_shape, *, num_actions=2, k=2,
                      batch_size=6):
    """A NECAlgorithm wired by hand, minus setup()'s env dependency.

    Same construction pattern as
    tests/test_nec_kernel_scale.py::test_gradient_step_normalises_before_kernel_lookup.
    """
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=embedding_network,
        obs_key="obs",
        embedding_dim=EMBEDDING_DIM,
        dnd_capacity=200,
        k=k,
        kernel_delta=1e-3,
        dnd_lr=0.1,
        n_step=5,
        lr=1e-2,
        batch_size=batch_size,
        init_random_frames=0,
        num_updates=1,
    )
    alg._obs_shape = obs_shape
    alg._num_actions = num_actions
    alg._buffer_device = torch.device("cpu")
    alg.embedding_net = embedding_network(obs_shape, alg.embedding_dim)
    alg.dnd = DND(num_actions, alg.dnd_capacity, alg.k, alg.kernel_delta,
                  alg._buffer_device)
    alg.optimizer = torch.optim.Adam(alg.embedding_net.parameters(), lr=alg.lr)
    alg.replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=100, device="cpu")
    )
    return alg


def test_gradient_step_trains_the_embedding_network():
    """After one `_gradient_step()`, embedding-net params must have received
    nonzero gradients AND actually moved.

    This is the property that distinguishes NEC's embedding network from
    MFEC's frozen encoder. `DND.values` is deliberately NOT in the optimizer
    (see the AGENTS.md "NEC -- DND values, blend-only" section), so if the
    CNN were frozen too, nothing in NEC would learn at all and the smoke
    tests would still pass.
    """
    torch.manual_seed(0)
    alg = _make_minimal_nec(NatureEmbedding, OBS_SHAPE)

    obs = torch.rand(6, *OBS_SHAPE)
    actions = torch.tensor([0, 0, 0, 1, 1, 1])
    targets = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0, 0.0])

    alg.replay_buffer.extend(TensorDict(
        {"obs": obs, "action": actions, "n_step_return": targets},
        batch_size=[6],
    ))

    # Seed both DND tables past the sparsity guard (`_sizes[a] <= k`), with
    # values that (a) differ from `targets`, so the regression loss is
    # nonzero, and (b) differ *within* an action, so the kernel-weighted
    # Q̂ actually depends on h -- identical neighbour values would make
    # ∂Q̂/∂h vanish and the gradient assertions vacuous for the wrong reason.
    with torch.no_grad():
        h0 = nn.functional.normalize(alg.embedding_net(obs), dim=-1)
    seed_vals = torch.tensor([2.0, -2.0, 0.5, 2.0, -2.0, 0.5])
    alg.dnd.write_batch(0, h0[0:3], seed_vals[0:3], dnd_lr=1.0)
    alg.dnd.write_batch(1, h0[3:6], seed_vals[3:6], dnd_lr=1.0)
    assert all(s > alg.k for s in alg.dnd._sizes), (
        "DND too sparse -- _gradient_step would skip both actions and the "
        "gradient assertions below would be vacuous"
    )

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}

    loss, _ = alg._gradient_step()

    assert loss != 0.0, "gradient step was skipped -- the assertions below are vacuous"

    grads = {n: p.grad for n, p in alg.embedding_net.named_parameters()}
    assert all(g is not None for g in grads.values()), (
        f"no gradient reached: {[n for n, g in grads.items() if g is None]}"
    )
    assert any(g.abs().sum() > 0 for g in grads.values()), (
        "every embedding-network gradient is exactly zero -- the regression "
        "loss is not backpropagating through the kernel distance term"
    )

    changed = [
        n for n, p in alg.embedding_net.named_parameters()
        if not torch.equal(p.detach(), before[n])
    ]
    assert changed, (
        "optimizer.step() left every embedding-network parameter untouched -- "
        "the encoder is effectively frozen"
    )


def test_dnd_values_stay_out_of_the_optimizer():
    """Guard for the documented deviation: only the embedding net is optimised.

    See AGENTS.md "1. DND `values` tensor is a plain (non-grad) tensor" --
    re-adding `dnd.values` to Adam is the change that previously drove stored
    Q-values negative.
    """
    alg = _make_minimal_nec(NatureEmbedding, OBS_SHAPE)

    optim_params = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    assert optim_params == {id(p) for p in alg.embedding_net.parameters()}
    assert not alg.dnd.values.requires_grad


# ---------------------------------------------------------------------------
# 3. Hydra composition regression -- goal: the refactor changed nothing
# ---------------------------------------------------------------------------

def _param_signature(net: nn.Module):
    return sorted((n, tuple(p.shape)) for n, p in net.named_parameters())


@pytest.mark.parametrize("algorithm_cfg", ["nec", "nec_atari"])
def test_nec_configs_still_instantiate_after_config_group_refactor(algorithm_cfg):
    cfg = _compose_algorithm_cfg(algorithm_cfg)

    # The group landed at algorithm.embedding_network with the same content
    # the inline block used to have.
    assert cfg.embedding_network._target_ == "src.networks.NatureEmbedding"
    assert cfg.embedding_network._partial_ is True

    alg = instantiate(cfg, device=None)
    assert isinstance(alg, NECAlgorithm)

    net = alg._make_embedding_network(OBS_SHAPE, cfg.embedding_dim)
    reference = NatureEmbedding(
        OBS_SHAPE, cfg.embedding_dim,
        num_cells_cnn=(32, 64, 64), kernel_sizes=(8, 4, 3), strides=(4, 2, 1),
        activation_class=nn.ReLU,
    )

    assert _param_signature(net) == _param_signature(reference), (
        "the embedding_network config group must build the exact architecture "
        "the pre-refactor inline YAML block did"
    )
    assert sum(p.numel() for p in net.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    assert all(p.requires_grad for p in net.parameters())


@pytest.mark.parametrize("experiment", ["nec/pong", "nec/mspacman", "nec/hero"])
def test_nec_experiment_configs_still_resolve(experiment):
    """Existing NEC experiments must be untouched by the refactor."""
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(experiment, ["logger=[]"])
    assert cfg.algorithm.embedding_network._target_ == "src.networks.NatureEmbedding"

    alg = instantiate(cfg.algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)


def test_embedding_network_group_is_overridable_from_the_cli():
    """The whole point of goal 2: swap by group selection, not YAML surgery."""
    cfg = _compose_algorithm_cfg(
        "nec_atari",
        [
            "algorithm/embedding_network=dinov2_finetune",
            "algorithm.embedding_network.weights_path=/nonexistent.pth",
        ],
    )
    assert cfg.embedding_network._target_ == "src.networks.DINOv2Embedding"
    assert cfg.embedding_network.freeze_backbone is False


# ---------------------------------------------------------------------------
# 4. Config-swap smoke: setup() + step() with a different factory
# ---------------------------------------------------------------------------

def test_setup_and_step_with_a_different_embedding_network():
    """A non-CNN, non-84x84 embedding network runs end-to-end.

    Exercises the default (non-hand-wired) path: setup() builds the net from
    the factory, wires DNDPolicy/QValueActor/optimizer around it, and one
    step() with two complete episodes writes the DND, fills the replay
    buffer, and runs gradient updates -- with no shape errors anywhere.
    """
    torch.manual_seed(0)
    obs_shape = (2, 12, 12)     # deliberately NOT Atari-shaped
    num_actions = 3
    E, T = 1, 8

    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=_tiny_linear_embedding,
        obs_key="pixels",
        embedding_dim=8,
        dnd_capacity=64,
        k=2,
        n_step=2,
        lr=1e-3,
        batch_size=4,
        frames_per_batch=E * T,
        init_random_frames=0,
        num_updates=2,
        annealing_frames=100,
    )
    alg.setup(lambda: _MockAtariEnv(obs_shape, num_actions))

    assert isinstance(alg.embedding_net, nn.Sequential)
    assert alg._dnd_policy.embedding_net is alg.embedding_net
    # setup() must register the swapped net's params -- not some default net's.
    assert {id(p) for g in alg.optimizer.param_groups for p in g["params"]} == {
        id(p) for p in alg.embedding_net.parameters()
    }

    dones = torch.zeros(E, T, dtype=torch.bool)
    dones[0, 3] = True          # episode 1: t=0..3
    dones[0, 7] = True          # episode 2: t=4..7
    batch = TensorDict(
        {
            "pixels": torch.rand(E, T, *obs_shape),
            "action": torch.randint(0, num_actions, (E, T)),
            "next": TensorDict(
                {
                    "pixels":     torch.rand(E, T, *obs_shape),
                    "reward":     torch.ones(E, T, 1),
                    "done":       dones.unsqueeze(-1),
                    "terminated": dones.unsqueeze(-1),
                },
                batch_size=[E, T],
            ),
        },
        batch_size=[E, T],
    )

    before = {n: p.detach().clone() for n, p in alg.embedding_net.named_parameters()}
    metrics = alg.step(batch)

    assert "train/dnd_size" in metrics and "train/epsilon" in metrics
    assert alg._collected_frames == E * T
    assert len(alg.replay_buffer) == E * T
    assert sum(alg.dnd._sizes) > 0, "step() stored nothing in the DND"

    # Action selection through the swapped net must also work.
    q = alg._dnd_policy(torch.rand(4, *obs_shape))
    assert q.shape == (4, num_actions)

    assert any(
        not torch.equal(p.detach(), before[n])
        for n, p in alg.embedding_net.named_parameters()
    ), "the swapped embedding network was not trained by step()"


# ---------------------------------------------------------------------------
# 5. DINOv2 finetune scaffolding -- stubbed backbone, no network calls
# ---------------------------------------------------------------------------

STUB_EMBED_DIM = 32


class _StubViT(nn.Module):
    """(B, 3, H, W) -> (B, STUB_EMBED_DIM). Stands in for the real DINOv2 ViT.

    Same tiering as tests/test_dinov2_encoder.py: the real ViT needs both the
    .pth and facebookresearch/dinov2 architecture code, so CI runs against a
    stub and the real-weights path is opt-in via an env var there.
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


def test_dinov2_embedding_builds_and_matches_the_shape_contract(stub_hub):
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=28)

    out = net(torch.rand(3, *OBS_SHAPE))
    assert out.shape == (3, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_dinov2_embedding_is_trainable_by_default(stub_hub):
    """The NEC variant must NOT freeze the backbone the way MFEC's does."""
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=28)

    assert all(p.requires_grad for p in net.parameters())
    assert all(p.requires_grad for p in net.backbone.parameters()), (
        "src/encoders/dino_v2_encoder.py freezes the backbone for MFEC; the "
        "NEC variant is finetunable and must not"
    )


def test_dinov2_embedding_freeze_backbone_leaves_head_trainable(stub_hub):
    net = DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=28,
                          freeze_backbone=True)

    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.head.parameters())
    assert any(p.requires_grad for p in net.parameters()), (
        "freezing everything would hand Adam an empty parameter list"
    )


def test_dinov2_embedding_rejects_bad_image_size(stub_hub):
    with pytest.raises(AssertionError):
        DINOv2Embedding(OBS_SHAPE, EMBEDDING_DIM, image_size=100)   # 100 % 14 != 0


def test_dinov2_finetune_yaml_instantiates(stub_hub, tmp_path):
    """The YAML composes and its `_partial_` builds a real, trainable module."""
    ckpt = tmp_path / "stub.pth"
    torch.save(_StubViT().state_dict(), ckpt)

    cfg = _compose_algorithm_cfg(
        "nec_atari",
        [
            "algorithm/embedding_network=dinov2_finetune",
            f"algorithm.embedding_network.weights_path={ckpt}",
            "algorithm.embedding_network.image_size=28",
        ],
    )
    alg = instantiate(cfg, device=None)

    net = alg._make_embedding_network(OBS_SHAPE, cfg.embedding_dim)
    assert isinstance(net, DINOv2Embedding)
    assert net(torch.rand(2, *OBS_SHAPE)).shape == (2, cfg.embedding_dim)
    assert all(p.requires_grad for p in net.parameters())
