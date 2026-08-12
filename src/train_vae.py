"""VAE pretraining entry point for the MFEC "vae" encoder.

Collects raw pixel frames from an environment with a random policy, trains
``src.models.conv_vae.ConvVAE`` to reconstruct them, and saves a state_dict
checkpoint that ``src.encoders.vae_encoder.VAEEncoder`` can load at MFEC
setup() time. Defaults reproduce Blundell et al. 2016 ("Model-Free Episodic
Control"), Appendix D: 1M random-policy frames, RMSProp lr=1e-5, batch=100,
400,000 SGD steps.

Usage:
    python src/train_vae.py
    python src/train_vae.py device=cuda:0
    python src/train_vae.py collect.frames=50_000 train.steps=20_000  # quick run
    python src/train_vae.py collect.num_envs=8   # parallelize ALE frame collection

Then feed the resulting checkpoint into an MFEC run:
    python src/train.py experiment=mfec/vae \\
        algorithm.vae_checkpoint=<checkpoint.save_path printed above>
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train_vae", version_base="1.3")
def train_vae(cfg: DictConfig) -> None:
    path = _train_vae(cfg)
    print(f"\nSaved VAE checkpoint to {path}")


def _train_vae(cfg: DictConfig) -> str:
    """Separated from the Hydra decorator for testability."""
    import os

    from omegaconf import OmegaConf

    from src.environments.environment import Environment
    from src.models.conv_vae import ConvVAE
    from src.utils.seeding import seed_everything

    seed_everything(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    env_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
        if k != "_target_"
    }
    environment = Environment(**env_kwargs)

    # ALE env stepping is CPU-bound and single-threaded per env; collect.num_envs
    # parallelizes it across worker processes (ParallelEnv). PyTorch/OMP size
    # their default thread pool to the *visible* CPU count, which on a
    # cgroup-quota-limited container (e.g. 8 real CPUs but 64 visible) is a
    # large oversubscription once num_envs workers each spin up their own
    # thread pool too -- that contention is what makes collection look hung
    # rather than just slow. Cap threads to 1 for the collection phase only;
    # VAE training itself runs on GPU by default so this doesn't cost anything
    # there, and we restore the original count before falling back to CPU training.
    orig_num_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # exceed a single GPU's VRAM even though any one training minibatch is
    # tiny. _fit() moves only each minibatch to `device`.
    frames = _collect_frames(
        environment,
        int(cfg.collect.frames),
        int(cfg.collect.frames_per_batch),
        int(cfg.collect.num_envs),
    )

    torch.set_num_threads(orig_num_threads)
    in_channels = frames.shape[1]   # inferred from the env, not hardcoded

    vae = ConvVAE(in_channels=in_channels, latent_dim=int(cfg.vae.latent_dim)).to(device)
    optimizer = torch.optim.RMSprop(vae.parameters(), lr=float(cfg.train.lr))

    _fit(vae, optimizer, frames, device, cfg)

    save_path = str(cfg.checkpoint.save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), save_path)
    return save_path


def _collect_frames(
    environment, num_frames: int, frames_per_batch: int, num_envs: int = 1
) -> torch.Tensor:
    """Roll out a random policy and return raw pixel frames, shape (N, C, H, W).

    num_envs > 1 uses ParallelEnv (one process per env) — ALE stepping is
    CPU-bound and single-threaded per env, so this is how to use more than
    one core for collection. Tune it to the machine's actual CPU budget
    (cgroup quota, not just os.cpu_count()) to avoid oversubscription.
    """
    import time

    from torchrl.collectors import Collector, RandomPolicy

    env = environment.make_env(num_envs=num_envs, device="cpu")
    collector = Collector(
        create_env_fn=env,
        policy=RandomPolicy(env.action_spec),
        frames_per_batch=frames_per_batch,
        total_frames=num_frames,
        device="cpu",
        storing_device="cpu",
    )

    chunks: list[torch.Tensor] = []
    collected = 0
    t0 = time.time()
    for batch in collector:
        pixels = batch["pixels"].reshape(-1, *batch["pixels"].shape[-3:])
        chunks.append((pixels * 255.0).round().to(torch.uint8))   # cast to uint8 per minibatch to save RAM
        collected += pixels.shape[0]
        elapsed = time.time() - t0
        print(
            f"collecting: {collected}/{num_frames} frames "
            f"({collected / elapsed:.1f} fps, {elapsed:.0f}s elapsed)",
            flush=True,
        )
        if collected >= num_frames:
            break
    collector.shutdown()

    return torch.cat(chunks, dim=0)[:num_frames]    #keep uint8 cast per minibatch in _fit() to save RAM


def _fit(vae, optimizer, frames: torch.Tensor, device: torch.device, cfg: DictConfig) -> None:
    from src.models.conv_vae import vae_loss

    n = frames.shape[0]
    batch_size = int(cfg.train.batch_size)
    if n < batch_size:
            raise ValueError(
                f"Collected only {n} frames but train.batch_size={batch_size}; "
                f"the epoch loop would never yield a batch. Increase "
                f"collect.frames or lower train.batch_size."
        )


    beta_target = float(cfg.vae.beta)
    beta_warmup = int(cfg.vae.get("beta_warmup_steps", 0))
    log_every = int(cfg.train.log_every_n_steps)
    total_steps = int(cfg.train.steps)

    step = 0
    while step < total_steps:
        perm = torch.randperm(n)   # frames lives on CPU; perm stays on CPU too
        for start in range(0, n - batch_size + 1, batch_size):
            if step >= total_steps:
                break
            x = frames[perm[start : start + batch_size]].to(device, non_blocking=True).float().div_(255.0)   # cast to float per minibatch to save RAM

            beta = beta_target if beta_warmup <= 0 else beta_target * min(1.0, step / beta_warmup)
            recon_mu, recon_logstd, mu, logstd = vae(x)
            loss, recon_loss, kl_loss = vae_loss(recon_mu, recon_logstd, x, mu, logstd, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % log_every == 0:
                print(
                    f"step {step}/{total_steps}  beta={beta:.3f}"
                    f"recon={recon_loss.item():.4f}  kl={kl_loss.item():.4f}"
                )


if __name__ == "__main__":
    train_vae()
