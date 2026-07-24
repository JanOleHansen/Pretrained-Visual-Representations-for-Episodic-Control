import torch
from .random_projectins import RandomProjectionEncoder
from .vae_encoder import VAEEncoder
from .dino_v2_encoder import DINOv2Encoder

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
):
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
    raise ValueError(f"Unknown encoder name: {name}")