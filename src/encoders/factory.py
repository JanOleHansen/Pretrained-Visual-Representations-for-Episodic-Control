import torch
from .random_projectins import RandomProjectionEncoder
from .vae_encoder import VAEEncoder
from .dino_v2_encoder import DINOv2Encoder
from .resnet_encoder import ResNetEncoder
# Safe to import at module scope: clip_encoder.py imports `open_clip` lazily,
# inside CLIPEncoder.__init__.  A top-level import there would make the missing
# optional dependency break *every* MFEC run, not just the clip arm.
from .clip_encoder import CLIPEncoder
# Same deal: mae_encoder.py imports `timm` inside MAEEncoder.__init__.
from .mae_encoder import MAEEncoder


def pin_fp32_conv_precision() -> None:
    """Force true FP32 (not TF32) for convolutions, process-wide.

    MFEC keys its episodic memory on ``round(φ(o) · key_scale)``
    (``QEC._key_to_slot``), so φ must return the *same bits* for the same frame
    whatever batch shape it is called with: the collector embeds ``num_envs``
    rows per policy call, while ``BaseTrainer.evaluate`` builds a single env and
    embeds 1.  cuDNN selects a convolution algorithm by input shape, so the two
    disagree in the low bits — and how many bits depends entirely on the
    accumulation precision.

    **``torch.backends.cudnn.allow_tf32`` defaults to ``True``.**  On Ampere and
    later that runs every convolution at TF32's 10-bit mantissa, i.e. a unit
    roundoff of 2⁻¹¹ ≈ 4.9e-4 *per operation*, versus FP32's 2⁻²⁴ ≈ 6e-8.
    (``torch.backends.cuda.matmul.allow_tf32`` defaults to ``False``, which is
    why this bites the conv backbones hardest — but a ViT's patch embedding is
    a ``Conv2d`` too, so no arm is exempt.)

    Why that is fatal rather than cosmetic.  The hash path is lost either way at
    d ≫ 1 (measured ``key b/s = 0.000`` for every float32 encoder on CUDA — see
    AGENTS.md).  What is supposed to catch it is ``QEC``'s near-exact rescue,
    which accepts the top-1 neighbour within ``3e-5 · (1 + ‖q‖)``.  Measured on
    400 real Ms. Pac-Man frames with ``resnet18``:

        ‖φ(o)‖ ≈ 25.0        -> rescue budget 7.8e-4 in L2
                             -> ~3.1e-5 relative, per coordinate, over d = 512
        nearest *distinct* frame sits 0.307 away — 400x of headroom, so the
        rescue cannot false-merge; it only has to survive the drift.

    A single TF32 rounding is ~16x that per-coordinate budget, before anything
    accumulates over ResNet-18's twenty-odd conv layers.  Measured consequence
    on the ``mfec/resnet`` Ms. Pac-Man run: ``eval/memory_hit_rate`` identically
    **0.000** for every seed and every eval point — dict path *and* rescue dead,
    so all |A| Q-estimates per frame were k-neighbour means over nearly the same
    neighbourhood, the argmax was noise, and ``eval/return_mean`` sat at
    random-play level (~400) while ``train/episode_reward`` climbed past 2000.
    ``eval/exact_minus_knn_value`` is absent from those runs for the same
    reason: ``value_stats`` had zero samples on the exact side.

    Called from :func:`make_encoder`, which is MFEC's and only MFEC's φ factory
    — NEC/DQN/DDPG/A2C build their own (gradient-trained) networks and never
    come through here, so their conv throughput is untouched.  It applies to
    every arm rather than just ``resnet`` because the ablation's whole premise
    is that φ is the only thing that varies; a per-arm numerics regime would
    make a between-arm read a comparison of float precisions.
    ``random_projection`` is unaffected either way (no convolution, and it
    already accumulates in float64 for exactly this reason).

    Two things this changes that are worth knowing:

    * **It is slower.**  TF32 is roughly a 2x throughput win on conv-bound work,
      and φ is the bottleneck for every PVM arm.  That cost is held equal across
      arms and is the price of a memory that can be read back.
    * **Checkpoints do not cross it.**  A QEC written before this change holds
      keys from the TF32 regime; resuming into an FP32 process re-embeds the
      same frames to different keys and every stored entry becomes unreachable.
      Re-run rather than resume.

    Do not "optimise" this back on.  ``eval/memory_hit_rate`` near 0 on a
    non-empty QEC is the signature if it ever is.
    """
    # Both setters exist and are safe on a CPU-only build; they are plain
    # globals, so this is a no-op there rather than a guard-and-skip.
    torch.backends.cudnn.allow_tf32 = False
    # Already the default, pinned so a future default flip — or anything else
    # in the process setting it globally — cannot reintroduce the same failure
    # through the ViT arms' GEMMs instead of their patch-embed convolution.
    torch.backends.cuda.matmul.allow_tf32 = False


def make_encoder(
        name: str,
        *,
        obs_flat_dim: int,
        in_channels: int,
        state_dim: int = 64,
        vae_checkpoint_path: str | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
        # dinov2
        dinov2_weights_path: str | None = None,
        dinov2_model_name: str = "dinov2_vits14",
        dinov2_repo_dir: str | None = None,
        dinov2_image_size: int = 224,
        # resnet
        resnet_weights_path: str | None = None,
        resnet_model_name: str = "resnet18",
        resnet_image_size: int = 224,
        # clip
        clip_weights_path: str | None = None,
        clip_model_name: str = "ViT-B-32-quickgelu",
        clip_pretrained_tag: str | None = "openai",
        clip_image_size: int | None = None,
        clip_normalize: bool = True,
        clip_interpolation: str = "bicubic",
        # mae
        mae_weights_path: str | None = None,
        mae_model_name: str = "vit_base_patch16_224.mae",
        mae_image_size: int = 224,
        mae_pooling: str = "mean",
):
    """Build MFEC's φ.

    NOTE: ``MFECAlgorithm.setup()`` passes *every* encoder-specific keyword on
    every call, whatever ``name`` is — so a keyword added here for one encoder
    must be accepted here before the call site starts sending it, or *all*
    encoders break with a TypeError, not just the new one.
    ``tests/test_encoder_factory.py`` pins that seam.
    """
    # Before any encoder is built, and therefore before any φ forward pass:
    # MFEC's episodic memory is keyed on φ's bits, and TF32 convolutions are
    # not reproducible across the batch shapes the collector and evaluate()
    # use.  See pin_fp32_conv_precision() for the measurement.
    pin_fp32_conv_precision()

    if name == "random_projection":
        return RandomProjectionEncoder(obs_flat_dim, state_dim, seed)
    if name == "vae":
        if vae_checkpoint_path is None:
            raise ValueError("vae_checkpoint_path must be provided for VAE encoder")
        return VAEEncoder(vae_checkpoint_path, in_channels, state_dim, device)
    if name == "dinov2":
        if dinov2_weights_path is None:
            raise ValueError("dinov2_weights_path must be provided for DINOv2 encoder")
        return DINOv2Encoder(
            weights_path=dinov2_weights_path,
            model_name=dinov2_model_name,
            repo_dir=dinov2_repo_dir,
            image_size=dinov2_image_size,
            device=device,
        )
    if name == "resnet":
        # weights_path=None is legal here (unlike dinov2): torchvision downloads
        # IMAGENET1K_V1 into ~/.cache/torch.  Set a path on an offline cluster.
        return ResNetEncoder(
            model_name=resnet_model_name,
            weights_path=resnet_weights_path,
            image_size=resnet_image_size,
            device=device,
        )
    if name == "clip":
        # weights_path=None is legal (like resnet, unlike dinov2): open_clip
        # resolves clip_pretrained_tag from its hub, which needs network
        # access.  Pass a path on an offline cluster.
        return CLIPEncoder(
            weights_path=clip_weights_path,
            model_name=clip_model_name,
            pretrained_tag=clip_pretrained_tag,
            image_size=clip_image_size,
            device=device,
            normalize=clip_normalize,
            interpolation=clip_interpolation,
        )
    if name == "mae":
        # weights_path=None is legal (like resnet and clip, unlike dinov2):
        # timm resolves mae_model_name from the HuggingFace hub, which needs
        # network access.  Pass a path on an offline cluster.
        return MAEEncoder(
            weights_path=mae_weights_path,
            model_name=mae_model_name,
            image_size=mae_image_size,
            pooling=mae_pooling,
            device=device,
        )
    raise ValueError(f"Unknown encoder name: {name}")
