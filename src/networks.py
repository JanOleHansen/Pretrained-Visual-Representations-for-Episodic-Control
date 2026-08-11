"""Network factories used by ``configs/algorithm/*.yaml``.

Each factory takes ``(obs_shape, action_dim)`` positionally and keeps the
rest as keyword-only args, so a Hydra ``_partial_`` config can pre-bind the
kwargs while the algorithm's ``setup()`` supplies the runtime shape and
action count. ``action_dim`` is the discrete action count for value-based
algorithms (DQN) and the continuous action vector size for actor/critic
algorithms (DDPG).
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence, Type, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import ConvNet, MLP


def make_mlp_q_net(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """Plain MLP Q-network. Flattens ``obs_shape`` to ``in_features``."""
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=num_actions,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG deterministic actor.

    Returns an MLP mapping the flattened observation to ``action_dim``
    unbounded outputs. The algorithm wraps this with ``TanhModule`` to
    rescale to the action spec, so this factory must NOT apply tanh itself.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=action_dim,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_critic(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG state-action value (critic).

    Returns an MLP mapping the concatenated ``[obs, action]`` vector to a
    single Q-value. ``ValueOperator`` concatenates inputs along the last
    dim before calling the module.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)) + int(action_dim),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C stochastic actor.

    Returns an MLP mapping the flattened observation to ``2 * action_dim``
    outputs. The algorithm chains it with ``NormalParamExtractor`` to split
    the output into ``loc`` and (positive) ``scale`` for a TanhNormal policy.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=2 * int(action_dim),
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_value(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C state-value critic.

    Takes ``(obs_shape, action_dim)`` for signature parity with the actor
    factory; ``action_dim`` is unused — the critic estimates V(s) only.
    Returns an MLP mapping the flattened observation to a single value.
    """
    del action_dim  # signature parity with actor factory
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def NatureDQN(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    num_cells_mlp: Sequence[int] = (512,),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """ConvNet -> MLP Q-network from Mnih et al. 2015 (\"Nature DQN\")."""
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=list(num_cells_cnn),
        kernel_sizes=list(kernel_sizes),
        strides=list(strides),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, *obs_shape))
    mlp = MLP(
        in_features=cnn_out.shape[-1],
        out_features=num_actions,
        num_cells=list(num_cells_mlp),
        activation_class=activation_class,
    )
    return nn.Sequential(cnn, mlp)


# ---------------------------------------------------------------------------
# NEC embedding networks (config group: configs/algorithm/embedding_network/)
# ---------------------------------------------------------------------------


@runtime_checkable
class NECEmbeddingNetwork(Protocol):
    """Call-signature contract for NEC's ``embedding_network`` factory.

    This is documentation, not machinery: nothing type-checks against it at
    runtime and factories need not inherit from it.  It exists so the
    contract has one place to live, referenced from
    ``NECAlgorithm.__init__``, ``configs/algorithm/embedding_network/*.yaml``
    and AGENTS.md ("Adding a new NEC embedding network").

    A conforming factory is any callable

        ``factory(obs_shape: Sequence[int], embedding_dim: int, **kwargs)
          -> nn.Module``

    where ``obs_shape`` is the per-sample observation shape ``(C, H, W)``
    (parallel-env batch dims already stripped by ``NECAlgorithm.setup()``)
    and ``embedding_dim`` is ``algorithm.embedding_dim``.  Everything after
    the two positional args must be keyword-only so a Hydra ``_partial_``
    config can pre-bind design kwargs without colliding with ``setup()``'s
    positional call.  A plain function, an ``nn.Module`` subclass, or a
    ``functools.partial`` all qualify.

    The returned ``nn.Module`` must satisfy:

    1. ``forward(obs: (B, *obs_shape) float32) -> (B, embedding_dim)
       float32``.  ``B`` is a flat batch — ``DNDPolicy.forward`` and
       ``NECAlgorithm._gradient_step`` reshape away any leading (E, T) dims
       before calling it.
    2. **All parameters trainable by default** (``requires_grad=True``).
       Unlike MFEC's frozen ``Encoder`` protocol (``src/encoders/base.py``),
       this network is optimised end-to-end: ``setup()`` hands
       ``embedding_net.parameters()` straight to Adam, so a frozen
       parameter is silently a dead parameter.  A factory MAY expose an
       opt-in freeze kwarg (see ``DINOv2Embedding``'s ``freeze_backbone``),
       but it must default to *not* frozen, and it must leave at least one
       trainable parameter or the optimizer gets an empty parameter list.
    3. Output must tolerate the L2 normalisation NEC applies before every
       DND read/write (``F.normalize(h, dim=-1)``, see
       ``tests/test_nec_kernel_scale.py``): rows must not be all-zero
       (``normalize`` would return zeros and collapse every kernel
       distance), and the module must not itself already emit unit-norm
       rows — the normalisation is what spreads embeddings far enough apart
       for the inverse-distance kernel to discriminate, and pre-normalising
       inside the module makes it a no-op that also hides gradient scale.
    4. No state beyond ``state_dict()`` — ``NECAlgorithm._get_training_state``
       checkpoints the network with ``embedding_net.state_dict()`` only.  A
       factory needing more must extend ``_get_training_state`` /
       ``_load_training_state`` (see AGENTS.md).

    Optional extension
    ------------------
    A module MAY define ``param_groups(base_lr: float) -> list[dict]``.  When
    present, ``NECAlgorithm._build_optimizer`` passes its result to RMSProp
    instead of a flat ``parameters()`` list, which is how a module splits
    itself across learning rates — see :class:`DINOv2Embedding`, whose
    pretrained backbone runs below its randomly-initialised head.  Modules
    without the attribute are unaffected.  The groups must cover exactly the
    trainable parameters: ``NECAlgorithm`` still checkpoints and restores the
    whole ``state_dict()``, and it rebuilds the optimizer through the same
    method on resume, so the grouping has to be reproducible from the
    constructor arguments alone.
    """

    def __call__(
        self, obs_shape: Sequence[int], embedding_dim: int
    ) -> nn.Module: ...


def NatureEmbedding(
    obs_shape: Sequence[int],
    embedding_dim: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """ConvNet trunk + single dense layer to ``embedding_dim``. No Q-head.

    The standard (default) NEC embedding network — selected by
    ``configs/algorithm/embedding_network/nature.yaml``.

    Used by NEC: maps (B, C, H, W) pixel observations to (B, embedding_dim)
    state embeddings.  Architecture follows NatureDQN's convolutional trunk
    (Mnih et al. 2015) with the MLP Q-head replaced by a single linear layer.

    Called as ``embedding_network(obs_shape, embedding_dim)`` inside
    ``NECAlgorithm.setup()``.  Hydra uses ``_partial_`` to pre-bind the
    kwarg-only design params; ``setup()`` supplies the two positional args.
    Satisfies :class:`NECEmbeddingNetwork`.
    """
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=list(num_cells_cnn),
        kernel_sizes=list(kernel_sizes),
        strides=list(strides),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, *obs_shape))
    dense = nn.Linear(cnn_out.shape[-1], embedding_dim)
    return nn.Sequential(cnn, dense)


# ImageNet stats DINOv2 was trained with (same constants as
# src/encoders/dino_v2_encoder.py, duplicated rather than imported so
# networks.py stays free of a dependency on MFEC's frozen-encoder package).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Embedding(nn.Module):
    """Finetunable DINOv2 ViT as a NEC embedding network.

    Selected by ``configs/algorithm/embedding_network/dinov2_finetune.yaml``,
    i.e. ``algorithm/embedding_network=dinov2_finetune`` on the CLI.

    Contrast with MFEC's :class:`src.encoders.dino_v2_encoder.DINOv2Encoder`,
    which is the *frozen* variant: it calls ``p.requires_grad_(False)`` on
    every backbone parameter because MFEC's QEC hash needs a bit-exact,
    never-changing φ.  NEC trains its embedding end-to-end, so this class
    leaves the backbone trainable by default (``freeze_backbone=False``) —
    see :class:`NECEmbeddingNetwork` clause 2.  That difference is the entire
    reason a NEC-specific class exists rather than reusing the encoder.

    Pipeline: ``(B, C, H, W)`` → 1×1 conv channel adapter (only when
    ``C != 3``; Atari gives 4 stacked grayscale frames, the ViT wants RGB) →
    bilinear resize to ``image_size`` → ImageNet normalisation → ViT →
    ``nn.Linear(backbone.embed_dim, embedding_dim)`` head.  The head keeps
    the output unconstrained (not unit-norm), as clause 3 of the contract
    requires: NEC L2-normalises downstream.

    Channel adapter is initialised, not random
    ------------------------------------------
    The adapter is seeded to ``weight = 1/C``, ``bias = 0``, which makes its
    output the **mean over the stacked frames, replicated to R=G=B** — a
    plain grayscale image in ``[0, 1]``, exactly the kind of input DINOv2's
    ImageNet normalisation expects.  A default-initialised ``nn.Conv2d``
    instead emits ~N(0, ·) channels of a different scale and sign, so the
    ViT's first forward pass sees out-of-distribution input and the
    pretrained features are worth approximately nothing at step 0.  Since a
    pretrained representation that is only useful *after* the adapter has
    been learned defeats the point of using a PVM at all, the identity-ish
    init is load-bearing, not cosmetic.

    The symmetry is broken by the first gradient (the three output channels
    receive different gradients through the ViT's patch-embedding conv), so
    the adapter is free to learn a frame-to-channel assignment that encodes
    motion if that helps.  It just does not *start* from noise.

    Discriminative learning rate
    ----------------------------
    :meth:`param_groups` puts the pretrained backbone in its own group at
    ``base_lr * backbone_lr_scale`` and leaves the freshly-initialised
    adapter + head at ``base_lr``.  ``NECAlgorithm.setup()`` uses it when
    present (see ``NECAlgorithm._build_optimizer``).  This is the standard
    finetuning arrangement — a randomly-initialised head needs a much larger
    step than a pretrained trunk that is already near a good solution — and
    it matters more than usual here because NEC's RMSProp settings
    (lr=1e-5, α=0.9, ε=0.01) were calibrated on ``NatureEmbedding``, where
    *every* parameter is random init.

    ``backbone_lr_scale`` is **not** a validated number: no training run has
    been used to tune it. 0.1 is the conventional default; 1.0 restores a
    single uniform learning rate.

    Parameters
    ----------
    obs_shape       : (C, H, W) per-sample observation shape.
    embedding_dim   : output width, i.e. ``algorithm.embedding_dim``.
    weights_path    : path to a ``dinov2_*_pretrain.pth`` state_dict.  If
        ``None`` the backbone keeps torch.hub's random init — useful only
        for tests.
    model_name      : torch.hub entrypoint, e.g. ``dinov2_vits14``.
    repo_dir        : local clone of facebookresearch/dinov2 for offline
        hosts; ``None`` fetches from torch.hub (which also reuses an
        already-downloaded ``~/.cache/torch/hub`` copy without network).
    image_size      : ViT input size; must be a multiple of 14.  This is the
        dominant throughput knob: cost is quadratic in the token count, so
        224 (16×16=256 patches) is ~5.3× the compute of 98 (7×7=49), which
        is the smallest multiple of 14 that does not *downsample* an 84×84
        Atari frame.
    freeze_backbone : ``True`` freezes the ViT and trains only the adapter +
        head.  Defaults to ``False`` (full finetuning).
    backbone_lr_scale : multiplier on the backbone's learning rate relative
        to the adapter + head; see above.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        embedding_dim: int,
        *,
        weights_path: str | None = None,
        model_name: str = "dinov2_vits14",
        repo_dir: str | None = None,
        image_size: int = 224,
        freeze_backbone: bool = False,
        backbone_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        assert image_size % 14 == 0, (
            "DINOv2 ViT requires image_size to be a multiple of 14"
        )
        self.image_size = image_size
        self.backbone_lr_scale = float(backbone_lr_scale)

        # Architecture only (pretrained=False); weights are loaded below so
        # the same code path works from a local .pth on an offline host.
        if repo_dir is not None:
            self.backbone = torch.hub.load(
                repo_dir, model_name, source="local", pretrained=False
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=False
            )

        if weights_path is not None:
            sd = torch.load(weights_path, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            self.backbone.load_state_dict(sd, strict=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        in_channels = int(obs_shape[0])
        self.channel_adapter: nn.Module = (
            nn.Identity() if in_channels == 3 else nn.Conv2d(in_channels, 3, kernel_size=1)
        )
        if isinstance(self.channel_adapter, nn.Conv2d):
            # Mean over input channels, replicated to R=G=B. See the class
            # docstring: a random init here throws away the pretraining.
            with torch.no_grad():
                self.channel_adapter.weight.fill_(1.0 / in_channels)
                self.channel_adapter.bias.zero_()

        self.head = nn.Linear(int(self.backbone.embed_dim), embedding_dim)

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def param_groups(self, base_lr: float) -> list[dict]:
        """Optimizer param groups: backbone at a scaled LR, head at ``base_lr``.

        ``NECAlgorithm._build_optimizer`` calls this when the embedding
        network defines it and falls back to a flat ``parameters()``
        otherwise, so this is an opt-in extension of
        :class:`NECEmbeddingNetwork`, not a new requirement on it.

        Frozen parameters are filtered out rather than handed to the
        optimizer with ``requires_grad=False``: they would never receive a
        gradient, but they would still be counted by anything that inspects
        ``param_groups``, and an all-frozen group is an empty group.
        """
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        head = [
            p
            for module in (self.channel_adapter, self.head)
            for p in module.parameters()
            if p.requires_grad
        ]
        groups: list[dict] = []
        if backbone:
            groups.append({"params": backbone, "lr": base_lr * self.backbone_lr_scale})
        if head:
            groups.append({"params": head, "lr": base_lr})
        return groups

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().reshape(-1, *obs.shape[-3:])          # (B, C, H, W)
        # Adapt channels BEFORE the resize: the 1x1 conv is ~7x cheaper at
        # 84x84 than at 224x224, and the result is identical either way.
        x = self.channel_adapter(x)                            # (B, 3, H, W)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self._mean) / self._std
        return self.head(self.backbone(x).float())             # (B, embedding_dim)
