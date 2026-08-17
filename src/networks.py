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
import warnings
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


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

#: OpenAI CLIP's normalisation statistics — NOT ImageNet's, and NOT the same
#: numbers as ``_IMAGENET_MEAN`` above.  Same constants as
#: ``src/encoders/clip_encoder.py``, duplicated rather than imported for the
#: same reason the ImageNet ones are: networks.py stays free of a dependency on
#: MFEC's frozen-encoder package.  Used only as a fallback — open_clip exposes
#: the checkpoint's own stats on the vision tower from ~2.24 onward, and those
#: win.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_CLIP_INSTALL_HINT = (
    "CLIPEmbedding needs the `open_clip_torch` package, which is not "
    "installed.\n"
    "    uv sync --extra clip         (or: pip install open_clip_torch)\n"
    "open_clip is used rather than timm because it exposes the *projected* "
    "image embedding — the space CLIP's contrastive objective actually "
    "operates on, which is the whole reason to try CLIP as an episodic-memory "
    "key."
)


class CLIPEmbedding(nn.Module):
    """Finetunable CLIP vision tower as a NEC embedding network.

    Selected by ``configs/algorithm/embedding_network/clip_finetune.yaml``,
    i.e. ``algorithm/embedding_network=clip_finetune`` on the CLI.

    The NEC counterpart to MFEC's frozen
    :class:`src.encoders.clip_encoder.CLIPEncoder`.  Both isolate the
    **projected** image embedding (``model.visual``, 512-d for ViT-B-32) —
    the space the contrastive image-text loss was computed on — and both use
    CLIP's own normalisation constants and skip the centre crop.  The
    difference is that this one is trained end-to-end with the DND rather
    than frozen, per :class:`NECEmbeddingNetwork` clause 2.

    Pipeline: ``(B, C, H, W)`` → 1×1 conv channel adapter (only when
    ``C != 3``) → bicubic resize to the tower's resolution → CLIP
    normalisation → vision tower → optional L2 → ``nn.Linear(proj_dim,
    embedding_dim)`` head.  The head output is deliberately unconstrained;
    NEC L2-normalises downstream (contract clause 3).

    What ``normalize_features`` does here — and does NOT
    ----------------------------------------------------
    In MFEC, ``clip_normalize`` is load-bearing in a way it cannot be here.
    There φ *is* the memory key, so putting it on the unit sphere makes
    MFEC's Euclidean kNN exactly the cosine kNN CLIP was trained under
    (``||a-b||² = 2 - 2cos(a,b)``).

    In NEC a learned ``nn.Linear(512, embedding_dim)`` head sits between the
    CLIP embedding and the DND, and NEC normalises *that* output.  So the
    DND's metric is cosine distance in the **head's** space, not in CLIP's,
    and no amount of normalisation here recovers the guarantee.  Do not
    describe this arm as "cosine kNN in CLIP space" — it is not.

    What it does buy is worth having anyway: it fixes the head's input to the
    unit sphere.  A pretrained ViT-B-32 emits projected vectors of norm ~10.7
    (~23 at random init), measured on Ms. Pac-Man frames, and feeding that
    scale into a default-initialised ``nn.Linear`` makes the head's output
    scale — and therefore its gradient scale — depend on a quantity nothing
    else in NEC controls.  ``false`` ablates it.

    CLIP embeddings of Atari frames start nearly collinear
    ------------------------------------------------------
    Every frame is "a screenshot of Pac-Man" to a contrastive image-text
    model, so raw pairwise cosine in CLIP space measures **0.9935** over 60
    real frames — and 0.9949 on the MFEC arm's RGB 210×160 pair, so this is
    a property of CLIP on Atari, not of NEC's grayscale env.  What the
    pretraining buys sits in the residual after the common component is
    removed: 7.4% of the embedding norm pretrained against 0.75% at random
    init, a 10× difference.

    The consequence for NEC is that the DND kernel starts flatter than it
    does for ``NatureEmbedding``.  ``configs/algorithm/embedding_network/
    clip_finetune.yaml`` carries the measured table and the ``kernel_delta``
    note; read it before diagnosing a slow start as a bug.

    Checkpoint / architecture pairing — QuickGELU
    ---------------------------------------------
    OpenAI's CLIP was trained with **QuickGELU** activations; open_clip's
    plain ``ViT-B-32`` config uses standard GELU.  Pairing them loads without
    error and only emits a ``UserWarning`` — then returns subtly wrong
    features, which is the worst possible failure for an encoder whose entire
    value is the geometry of its output.  ``__init__`` makes it a hard error,
    exactly as ``CLIPEncoder`` does:

    ===========================  ==========================
    pretrained tag               model name
    ===========================  ==========================
    ``openai``                   ``ViT-B-32-quickgelu``
    ``laion2b_*`` / ``datacomp`` ``ViT-B-32``  (plain GELU)
    ===========================  ==========================

    A locally-cached OpenAI checkpoint is no exception: the file is correct,
    the *architecture* it is loaded into is what must match.

    Cost
    ----
    ViT-B-32's patch grid is 7×7 = 49 tokens at 224, against DINOv2
    ViT-S/14's 16×16 = 256, so the forward is comparable despite the wider
    model.  The **parameter count is not**: 87.8 M in the vision tower
    against DINOv2 ViT-S/14's 22 M.  That is ~350 MB of weights plus an equal
    RMSProp ``square_avg``, so NEC checkpoints run ~820 MB including the DND
    — budget disk accordingly at ``checkpoint.save_every_n_steps``.

    Parameters
    ----------
    obs_shape        : (C, H, W) per-sample observation shape.
    embedding_dim    : output width, i.e. ``algorithm.embedding_dim``.
    weights_path     : local open_clip checkpoint.  ``None`` lets open_clip
        resolve ``pretrained_tag`` from its hub, which **requires network
        access**.  A local path wins over the tag — the offline-cluster path.
    model_name       : open_clip architecture, spelled with hyphens
        (``ViT-B-32``), not OpenAI's ``ViT-B/32``.
    pretrained_tag   : open_clip tag used when ``weights_path`` is ``None``.
        Kept meaningful even with a local path because it declares the
        checkpoint's provenance, which is what the QuickGELU rule keys on.
        ``None`` with no path gives a randomly-initialised tower.
    image_size       : ``None`` (default) uses the tower's native resolution —
        the safe choice.  Any other value is passed as ``force_image_size``,
        which interpolates the positional embeddings away from the
        pretraining resolution, and must be divisible by the patch size (see
        the guard in ``__init__``).
    interpolation    : resize mode; ``"bicubic"`` is CLIP's own.  Set
        ``"bilinear"`` to hold resizing identical to the DINOv2 arm.
    normalize_features : L2-normalise the CLIP embedding before the head.
        See above.
    freeze_backbone  : ``True`` trains only the adapter + head.  Default
        ``False`` (full finetuning).
    backbone_lr_scale : multiplier on the vision tower's learning rate
        relative to the adapter + head.  Untuned; see :meth:`param_groups`.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        embedding_dim: int,
        *,
        weights_path: str | None = None,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained_tag: str | None = "openai",
        image_size: int | None = None,
        interpolation: str = "bicubic",
        normalize_features: bool = True,
        freeze_backbone: bool = False,
        backbone_lr_scale: float = 0.1,
        image_mean: tuple[float, float, float] | None = None,
        image_std: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:                        # pragma: no cover
            raise ImportError(_CLIP_INSTALL_HINT) from exc

        # OpenAI's CLIP uses QuickGELU; open_clip's plain configs use GELU.
        # Mismatched, open_clip loads the weights anyway and only warns, so the
        # run silently produces the wrong embedding geometry.  Fail loudly.
        if (
            (pretrained_tag or "").lower() == "openai"
            and "quickgelu" not in model_name.lower()
        ):
            raise ValueError(
                f"clip_model_name={model_name!r} is paired with "
                f"pretrained_tag='openai', but OpenAI's CLIP was trained with "
                f"QuickGELU activations and {model_name!r} uses standard GELU. "
                f"open_clip loads this combination with only a warning and "
                f"returns subtly wrong features.\n"
                f"    Fix: model_name={model_name + '-quickgelu'!r}\n"
                f"LAION/DataComp tags (laion2b_*, datacomp_*) want the plain "
                f"name instead — they were trained with standard GELU."
            )

        self.model_name = model_name
        self.interpolation = interpolation
        self.normalize_features = bool(normalize_features)
        self.backbone_lr_scale = float(backbone_lr_scale)

        # A local file wins over the hub tag; that is the offline-cluster path.
        pretrained = weights_path if weights_path is not None else pretrained_tag

        build_kwargs: dict = {"pretrained": pretrained, "precision": "fp32"}
        if image_size is not None:
            # Only pass it when asked: older open_clip releases lack the kwarg,
            # and passing it needlessly would break them for no benefit.
            build_kwargs["force_image_size"] = image_size

        try:
            model = open_clip.create_model_and_transforms(
                model_name, **build_kwargs
            )[0]
        except TypeError:
            if image_size is None:
                raise
            raise TypeError(
                f"The installed open_clip does not support force_image_size; "
                f"leave image_size unset to use {model_name}'s native "
                f"resolution, or upgrade open_clip_torch."
            )

        # `visual` is the vision tower *including* the projection into the
        # joint image-text space — open_clip's `CLIP.encode_image` is literally
        # `self.visual(image)`.  Keeping only this half drops the ~63 M-param
        # text tower, which for NEC is not merely dead weight: every parameter
        # of this module goes into RMSProp and into every checkpoint.
        visual = getattr(model, "visual", None)
        if visual is None:                                # pragma: no cover
            raise RuntimeError(
                f"open_clip model {model_name!r} has no `.visual` attribute; "
                "cannot isolate the vision tower."
            )
        self.backbone = visual

        # Native (or forced) input resolution. open_clip stores an int or an
        # (H, W) pair depending on version and architecture (3.3.0 returns a
        # tuple even for square inputs).
        native = getattr(visual, "image_size", image_size or 224)
        if isinstance(native, (tuple, list)):
            native = native[0]
        self.image_size = int(native)

        self._assert_patch_grid_covers_the_image(image_size)
        self._warn_if_batch_dependent()

        # CLIP's constants, preferring whatever the checkpoint declares.
        mean = image_mean or getattr(visual, "image_mean", None) or _CLIP_MEAN
        std = image_std or getattr(visual, "image_std", None) or _CLIP_STD
        self.register_buffer(
            "_mean", torch.tensor(tuple(mean)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(tuple(std)).view(1, 3, 1, 1), persistent=False
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        in_channels = int(obs_shape[0])
        self.channel_adapter: nn.Module = (
            nn.Identity() if in_channels == 3
            else nn.Conv2d(in_channels, 3, kernel_size=1)
        )
        if isinstance(self.channel_adapter, nn.Conv2d):
            # Mean over input channels, replicated to R=G=B — see
            # DINOv2Embedding for why a random init here throws the
            # pretraining away.
            with torch.no_grad():
                self.channel_adapter.weight.fill_(1.0 / in_channels)
                self.channel_adapter.bias.zero_()

        # Probed from a real forward rather than read off `output_dim`: it is
        # the ground truth, it costs one image, and `output_dim` is absent on
        # some open_clip vision towers (TimmModel wrappers, older releases).
        self.proj_dim = self._probe_proj_dim()
        self.head = nn.Linear(self.proj_dim, embedding_dim)

    # ------------------------------------------------------------------
    # Construction-time guards
    # ------------------------------------------------------------------

    def _assert_patch_grid_covers_the_image(self, requested: int | None) -> None:
        """Reject a forced ``image_size`` that the patch grid cannot tile.

        open_clip accepts ``force_image_size=112`` on a patch-32 tower without
        complaint, but the patch-embedding conv has kernel = stride = 32, so
        it produces a 3×3 grid covering only 96 pixels and **silently discards
        the bottom and right 16** — 14% of each axis.  On Ms. Pac-Man that is
        the row carrying the score and remaining lives.  Measured:

            force_image_size   grid   pixels covered   dropped
                  96           3×3        96/96           0
                 112           3×3        96/112         16
                 224           7×7       224/224          0

        Nothing upstream raises, so the check lives here.  It is skipped when
        the tower exposes no patch-embedding ``conv1`` (non-ViT towers), where
        the arithmetic does not apply.
        """
        if requested is None:
            return                       # native resolution always tiles
        conv = getattr(self.backbone, "conv1", None)
        if not isinstance(conv, nn.Conv2d):
            return                       # not a ViT tower; no patch grid
        patch = int(conv.stride[0])
        if self.image_size % patch:
            valid = [s for s in range(patch, 257, patch)]
            raise ValueError(
                f"image_size={self.image_size} is not divisible by "
                f"{self.model_name}'s patch size ({patch}). open_clip accepts "
                f"this without error, but the patch-embedding conv would tile "
                f"only {self.image_size // patch * patch} pixels and silently "
                f"discard the last {self.image_size % patch} of each axis.\n"
                f"    Valid sizes: {valid}"
            )

    def _warn_if_batch_dependent(self) -> None:
        """Warn when the tower carries BatchNorm (CLIP's RN* variants do).

        NEC keeps the embedding network in training mode — it is being
        optimised — so BatchNorm would normalise with *batch* statistics.  The
        same frame then embeds differently depending on what it was batched
        with, and NEC batches differently in each phase: ``num_envs`` rows
        during collection, ``batch_size`` during gradient steps, and **1** in
        ``BaseTrainer.evaluate``.  Keys written in one phase are queried in
        another, so the DND's whole premise (§6, "keys ... remain relatively
        stable") breaks.  Measured: ``RN50``'s vision tower has 55 BatchNorm
        modules; every ViT-* tower has none.

        A warning rather than an error: it is a legitimate thing to ablate,
        and ``freeze_backbone=True`` plus a manual ``.eval()`` would make it
        sound.  It must not be silent.
        """
        n_bn = sum(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
            for m in self.backbone.modules()
        )
        if n_bn:
            warnings.warn(
                f"CLIP vision tower {self.model_name!r} contains {n_bn} "
                f"BatchNorm layers. NEC trains the embedding network, so it "
                f"stays in train() mode and BatchNorm will use batch "
                f"statistics — the same observation then embeds differently "
                f"during collection (batch=num_envs), gradient steps "
                f"(batch=batch_size) and evaluation (batch=1), which "
                f"destabilises every key in the DND. Prefer a ViT-* tower.",
                UserWarning,
                stacklevel=3,
            )

    @torch.no_grad()
    def _probe_proj_dim(self) -> int:
        dev = next(self.backbone.parameters()).device
        probe = torch.zeros(1, 3, self.image_size, self.image_size, device=dev)
        out = self.backbone(probe)
        if out.ndim != 2:                                 # pragma: no cover
            raise RuntimeError(
                f"CLIP vision tower returned shape {tuple(out.shape)}; "
                "expected (B, d). NEC needs one vector per observation."
            )
        return int(out.shape[-1])

    # ------------------------------------------------------------------
    # NECEmbeddingNetwork optional extension
    # ------------------------------------------------------------------

    def param_groups(self, base_lr: float) -> list[dict]:
        """Vision tower at a scaled LR, adapter + head at ``base_lr``.

        Identical arrangement (and identical caveat) to
        :meth:`DINOv2Embedding.param_groups`: a randomly-initialised head
        needs a far larger step than a pretrained trunk, and NEC's RMSProp
        trio was calibrated on ``NatureEmbedding``, where every parameter is
        random init.  ``backbone_lr_scale`` is **not** a validated number.
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

    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().reshape(-1, *obs.shape[-3:])           # (B, C, H, W)
        # Adapt channels BEFORE the resize: the 1x1 conv is far cheaper at
        # 84x84 than at 224x224, and the result is identical either way.
        x = self.channel_adapter(x)                             # (B, 3, H, W)
        if x.shape[-2:] != (self.image_size, self.image_size):
            # Whole-frame resize, NOT CLIP's Resize+CenterCrop: a centre crop
            # of a 210x160 Atari frame cuts away the left and right of the
            # maze. Losing pixels is much worse than distorting them when the
            # embedding is a memory key. (align_corners is valid for both
            # bilinear and bicubic.)
            x = F.interpolate(
                x, (self.image_size, self.image_size),
                mode=self.interpolation, align_corners=False,
            )
        x = (x - self._mean) / self._std
        h = self.backbone(x).float()                            # (B, proj_dim)
        if self.normalize_features:
            h = F.normalize(h, dim=-1)
        return self.head(h)                                     # (B, embedding_dim)
