import torch
from .base import Encoder
from src.models.conv_vae import ConvVAE

class VAEEncoder(Encoder):
    """Frozen VAE. Embeds with concat(mean, log-std) of q(z|x) — deterministic,
    no sampling. MFEC's VAE baseline encoder (Blundell et al. 2016, Appendix D).

    state_dim is the total exposed embedding width (mean ⊕ log-std), so the
    underlying ConvVAE's latent_dim is state_dim // 2 (32 in the paper, for
    state_dim=64).
    """

    def __init__(self, checkpoint_path: str, in_channels: int = 1, state_dim: int = 64, device: torch.device | None = None):
        assert state_dim % 2 == 0, "state_dim must be even (mean + log-std halves)"
        self.state_dim = state_dim
        self.vae = ConvVAE(in_channels=in_channels, latent_dim=state_dim // 2)
        sd = torch.load(checkpoint_path, map_location=device)
        self.vae.load_state_dict(sd)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.vae.to(device)

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        if next(self.vae.parameters()).device != obs.device:
            self.vae.to(obs.device)
        x = obs.reshape(-1, *obs.shape[-3:])
        return self.vae.embed(x)

    def state(self) -> dict:
        return {"vae_state_dict": self.vae.state_dict()}

    def load_state(self, state: dict) -> None:
        self.vae.load_state_dict(state["vae_state_dict"])
        self.vae.eval()
