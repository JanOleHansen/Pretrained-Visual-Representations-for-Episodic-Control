import math

import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """Convolutional VAE for Atari frames — MFEC's VAE baseline encoder.

    Architecture matches Blundell et al. 2016 ("Model-Free Episodic
    Control"), Appendix D: encoder is 4 conv layers ({32,32,64,64} kernels
    {4,5,5,4}, strides {2,2,2,2}, no padding, ReLU) followed by a 512-unit FC
    ReLU layer and a linear layer outputting the mean and log-std of the
    approximate posterior q(z|x). The decoder mirrors this with a 512-unit FC
    layer and the transposed convolutions, producing the mean and log-std of
    p(x|z) (both diagonal Gaussians).

    encode() is a deterministic function of x (no sampling) — sampling only
    happens in reparameterize(), used for the training-time ELBO. This
    determinism is required when this network is used as a frozen MFEC
    encoder: identical frames must map to identical embeddings, or the QEC
    exact-hit hash path never fires.

    Input:  (B, in_channels, 84, 84)   in_channels=1 for a single grayscale frame
    Latent: (B, latent_dim)            latent_dim=32 in the paper
    """

    _SPATIAL_AFTER_CONV = 3  # 84 -> 41 -> 19 -> 8 -> 3 through the 4 conv layers

    _DEC_LOGSTD_MIN = -4    
    _DEC_LOGSTD_MAX = 2.0

    def __init__(self, in_channels: int = 1, latent_dim: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2), nn.ReLU(),
        )
        self._flat = 64 * self._SPATIAL_AFTER_CONV * self._SPATIAL_AFTER_CONV  # 576

        self.enc_fc = nn.Sequential(nn.Linear(self._flat, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logstd = nn.Linear(512, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, self._flat), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2), nn.ReLU(),
        )
        # Two heads (mean, log-std), each in_channels wide, in one conv.
        self.dec_out = nn.ConvTranspose2d(32, 2 * in_channels, kernel_size=4, stride=2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic: returns (mean, log-std) of q(z|x)."""
        h = self.enc(x).flatten(1)
        h = self.enc_fc(h)
        return self.fc_mu(h), self.fc_logstd(h)

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(logstd) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log-std) of p(x|z)."""
        h = self.dec_fc(z).view(
            -1, 64, self._SPATIAL_AFTER_CONV, self._SPATIAL_AFTER_CONV
        )
        h = self.dec(h)
        recon_mu, recon_logstd = self.dec_out(h).chunk(2, dim=1)
        recon_logstd = recon_logstd.clamp(self._DEC_LOGSTD_MIN, self._DEC_LOGSTD_MAX)
        return recon_mu, recon_logstd

    def forward(self, x: torch.Tensor):
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        recon_mu, recon_logstd = self.decode(z)
        return recon_mu, recon_logstd, mu, logstd

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """MFEC state embedding: concat(mean, log-std) of q(z|x) — (B, 2*latent_dim).

        Per Blundell et al. 2016: "both the mean and log-standard-deviation
        parameters ... were used as dimensions for computing Euclidean
        distances in the episodic controller." Deterministic — no sampling.
        """
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        
        mu, logstd = self.encode(x)
        return torch.cat([mu, logstd], dim=-1)


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of x under a diagonal Gaussian N(mu, exp(logstd)^2)."""
    var = torch.exp(2 * logstd)
    nll = 0.5 * math.log(2 * math.pi) + logstd + (x - mu) ** 2 / (2 * var)
    return nll.sum(dim=[1, 2, 3]).mean()


def kl_diag_gaussian(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) for a diagonal Gaussian posterior."""
    kl = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1 - 2 * logstd)
    return kl.sum(dim=1).mean()


def vae_loss(
    recon_mu: torch.Tensor,
    recon_logstd: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logstd: torch.Tensor,
    beta: float = 1.0,
):
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    
    recon_loss = gaussian_nll(x, recon_mu, recon_logstd)
    kl = kl_diag_gaussian(mu, logstd)
    return recon_loss + beta * kl, recon_loss, kl
