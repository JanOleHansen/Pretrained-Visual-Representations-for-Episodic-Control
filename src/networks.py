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
    """**SCAFFOLDING — UNTESTED AGAINST REAL WEIGHTS. NOT A FINISHED FEATURE.**

    Finetunable DINOv2 ViT as a NEC embedding network, selected by
    ``configs/algorithm/embedding_network/dinov2_finetune.yaml``.

    What is actually verified (``tests/test_nec_embedding_network.py``, with
    ``torch.hub.load`` monkeypatched to a stub backbone): the YAML composes,
    the factory builds, the forward shape/dtype contract holds, and
    ``freeze_backbone`` gates ``requires_grad`` as documented.  What is
    **not** verified: that real ``dinov2_vits14`` weights load through this
    path, that the grayscale→RGB adapter and 84→224 upsample are the right
    preprocessing for Atari frames, or that NEC learns anything with it.
    Nobody has run a training job with this. Treat every hyperparameter
    below as a placeholder and validate before drawing conclusions from a
    run that uses it.

    Contrast with MFEC's :class:`src.encoders.dino_v2_encoder.DINOv2Encoder`,
    which is the *frozen* variant: it calls ``p.requires_grad_(False)`` on
    every backbone parameter because MFEC's QEC hash needs a bit-exact,
    never-changing φ.  NEC trains its embedding end-to-end, so this class
    deliberately leaves the backbone trainable by default
    (``freeze_backbone=False``) — see :class:`NECEmbeddingNetwork` clause 2.

    Pipeline: ``(B, C, H, W)`` → 1×1 conv channel adapter (only when
    ``C != 3``; Atari gives 4 stacked grayscale frames, the ViT wants RGB) →
    bilinear resize to ``image_size`` → ImageNet normalisation → ViT →
    ``nn.Linear(backbone.embed_dim, embedding_dim)`` head.  The head keeps
    the output unconstrained (not unit-norm), as clause 3 of the contract
    requires: NEC L2-normalises downstream.

    Parameters
    ----------
    obs_shape       : (C, H, W) per-sample observation shape.
    embedding_dim   : output width, i.e. ``algorithm.embedding_dim``.
    weights_path    : path to a ``dinov2_*_pretrain.pth`` state_dict.  If
        ``None`` the backbone keeps torch.hub's random init — useful only
        for tests.
    model_name      : torch.hub entrypoint, e.g. ``dinov2_vits14``.
    repo_dir        : local clone of facebookresearch/dinov2 for offline
        hosts; ``None`` fetches from torch.hub.
    image_size      : ViT input size; must be a multiple of 14.
    freeze_backbone : ``True`` freezes the ViT and trains only the adapter +
        head.  Defaults to ``False`` (full finetuning) — the whole point of
        having a NEC-specific variant.
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
    ) -> None:
        super().__init__()
        assert image_size % 14 == 0, (
            "DINOv2 ViT requires image_size to be a multiple of 14"
        )
        self.image_size = image_size

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
        self.head = nn.Linear(int(self.backbone.embed_dim), embedding_dim)

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().reshape(-1, *obs.shape[-3:])          # (B, C, H, W)
        x = self.channel_adapter(x)                            # (B, 3, H, W)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self._mean) / self._std
        return self.head(self.backbone(x).float())             # (B, embedding_dim)
