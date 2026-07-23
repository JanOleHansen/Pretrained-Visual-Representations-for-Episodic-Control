import torch
from .random_projectins import RandomProjectionEncoder
from .vae_encoder import VAEEncoder

def make_encoder(
        name: str,
        *,
        obs_flat_dim: int,
        in_channels: int,
        state_dim: int = 64,
        vae_checkpoint_path: str | None = None,
        device: torch.device | None = None,
        seed: int | None = None,       
): 
    if name == "random_projection":
        return RandomProjectionEncoder(obs_flat_dim, state_dim, seed)
    if name == "vae":
        if vae_checkpoint_path is None:
            raise ValueError("vae_checkpoint_path must be provided for VAE encoder")
        return VAEEncoder(vae_checkpoint_path, in_channels, state_dim, device)
    raise ValueError(f"Unknown encoder name: {name}")