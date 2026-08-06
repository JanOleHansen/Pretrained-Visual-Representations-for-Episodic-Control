import torch
from .random_projectins import RandomProjectionEncoder
from .vae_encoder import VAEEncoder
from .dino_v2_encoder import DINOv2Encoder
from .resnet_encoder import ResNetEncoder


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
):
    """Build MFEC's φ.

    NOTE: ``MFECAlgorithm.setup()`` passes *every* encoder-specific keyword on
    every call, whatever ``name`` is — so a keyword added here for one encoder
    must be accepted here before the call site starts sending it, or *all*
    encoders break with a TypeError, not just the new one.
    ``tests/test_encoder_factory.py`` pins that seam.
    """
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
    raise ValueError(f"Unknown encoder name: {name}")
