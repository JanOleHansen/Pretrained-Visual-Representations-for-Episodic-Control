#!/usr/bin/env python
"""Diagnose an MFEC encoder φ *before* spending GPU-weeks training with it.

Motivation
----------
MFEC's whole learning signal passes through two properties of φ:

1. **Key stability.**  Eq. (1)'s max-update fires only on an *exact* hash
   match (``QEC._key_to_slot``, quantised at ``key_scale``).  The paper
   measures ~50% exact state-action matches on Ms. Pac-Man; at 0% MFEC
   degenerates into plain kNN averaging with no one-shot latching, and no
   amount of training recovers it.
2. **Discriminability.**  The kNN fallback averages the k nearest stored
   returns.  If φ maps visually different game states to nearly the same
   point, Q(s, a) stops depending on s and the greedy policy is noise.

Both are cheap to measure and neither needs a training run.  This script
measures them on real frames, for every encoder you can build, side by side.

What the numbers mean
---------------------
``key match (batch vs single)``
    Fraction of frames whose quantised key is identical when embedded in a
    batch versus one row at a time.  **This must be 1.000.**  Training embeds
    ``num_envs`` rows at a time while ``BaseTrainer.evaluate`` builds a single
    env and embeds 1 row; below 1.000 those two disagree about what "the same
    state" is, so evaluation silently never takes the exact-match path.
    ``RandomProjectionEncoder`` accumulates in float64 specifically to keep
    this at 1.000 (see its docstring and
    ``tests/test_mfec_qec_dict.py::test_key_is_batch_shape_invariant``).  A
    float32 ViT has no such guarantee.

``key match (repeat)``
    Same batch shape, embedded twice.  Below 1.000 means φ is not even
    run-to-run deterministic, which breaks the ``Encoder`` contract in
    ``src/encoders/base.py`` outright.

``relative contrast``
    ``(mean pairwise distance - nearest-neighbour distance) / mean``.  Near 0
    means every stored point is about as close to a query as every other, so
    "the k nearest" is arbitrary.  Higher is better; compare arms, not the
    absolute value.

``adjacency AUC``
    Probability that a temporally adjacent frame pair (|i-j| <= 2, i.e. nearly
    the same game state) is closer than a distant pair (|i-j| >= 50).  0.5 =
    φ cannot tell related states from unrelated ones.  1.0 = perfect ordering.
    This is the single most informative number here.

``dist CV``
    Coefficient of variation of pairwise distances.  Collapses toward 0 when
    all embeddings concentrate on a shell (the classic high-dimensional
    concentration failure).

Usage
-----
    python scripts/encoder_diagnostics.py                       # random projection only
    python scripts/encoder_diagnostics.py \
        --dinov2-weights /datahome/rschwinger/models/dinov2_vits14_pretrain.pth

``--dinov2-random-init`` builds the ViT architecture with *untrained* weights.
That tells you nothing about representation quality, but the numerics (and
therefore the key-stability rows) are identical to the pretrained model, so it
reproduces key-stability findings on a host with no weights available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoders.random_projectins import RandomProjectionEncoder  # noqa: E402
from src.environments.factory import make_env  # noqa: E402


# ---------------------------------------------------------------------------
# Frame collection — drives the REAL config pipelines, not a hand-rolled copy
# ---------------------------------------------------------------------------

def collect_frames(env_config_name: str, n_frames: int, seed: int) -> torch.Tensor:
    """Roll out one env built from ``configs/environment/<name>.yaml``.

    Returns the stacked ``pixels`` observations, shape ``(n_frames, C, H, W)``.
    Uses a fixed seed and a fixed action stream so different pipelines see the
    same underlying game — the arms stay comparable.
    """
    from omegaconf import OmegaConf

    # Environment configs are standalone (no `defaults:`), so load the file
    # directly rather than composing the whole train config — no algorithm
    # override needed, and it stays honest about which YAML was read.
    path = (Path(__file__).resolve().parents[1]
            / "configs" / "environment" / f"{env_config_name}.yaml")
    cfg = OmegaConf.load(path)

    env_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k != "_target_"
    }
    env = make_env(**env_kwargs, num_envs=1, device="cpu")

    torch.manual_seed(seed)
    td = env.reset()
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_spec.space.n)

    frames = []
    for _ in range(n_frames):
        td["action"] = torch.tensor(int(rng.integers(n_actions)))
        td = env.step(td)
        frames.append(td["next"]["pixels"].clone())
        if bool(td["next"]["done"].any()):
            td = env.reset()
        else:
            td = td["next"].clone()
    env.close()
    return torch.stack(frames)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def key_stability(encoder, frames: torch.Tensor, key_scale: float) -> tuple[float, float]:
    """(batch-vs-single match rate, repeat match rate) of the QEC hash key."""
    from src.algorithms.mfec import QEC

    qec = QEC(1, 8, 1, torch.device("cpu"), key_scale=key_scale)

    batched = qec._make_keys(encoder.embed(frames))
    repeat = qec._make_keys(encoder.embed(frames))
    single = [
        qec._make_keys(encoder.embed(frames[i : i + 1]))[0]
        for i in range(frames.shape[0])
    ]

    n = frames.shape[0]
    bs = sum(batched[i] == single[i] for i in range(n)) / n
    rp = sum(batched[i] == repeat[i] for i in range(n)) / n
    return bs, rp


def geometry(emb: torch.Tensor) -> dict[str, float]:
    """Distance-geometry health of an embedding cloud."""
    emb = emb.double()
    d = torch.cdist(emb, emb)
    n = d.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    vals = d[off]

    mean_d = vals.mean().item()
    std_d = vals.std().item()

    # Nearest non-self neighbour per query.
    d_nn = d.clone()
    d_nn.fill_diagonal_(float("inf"))
    nn_d = d_nn.min(dim=1).values.mean().item()

    return {
        "mean_dist": mean_d,
        "dist_cv": std_d / mean_d if mean_d > 0 else 0.0,
        "rel_contrast": (mean_d - nn_d) / mean_d if mean_d > 0 else 0.0,
    }


def adjacency_auc(emb: torch.Tensor, near: int = 2, far: int = 50) -> float:
    """P(distance(adjacent pair) < distance(distant pair)).

    Adjacent frames are almost the same game state, so a useful φ must place
    them closer than frames tens of steps apart.  0.5 == no signal.
    """
    emb = emb.double()
    d = torch.cdist(emb, emb)
    n = d.shape[0]
    idx = torch.arange(n)
    gap = (idx[:, None] - idx[None, :]).abs()

    pos = d[(gap <= near) & (gap > 0)]
    neg = d[gap >= far]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")

    # Exact AUC via rank statistic (Mann-Whitney U).
    allv = torch.cat([pos, neg])
    ranks = allv.argsort().argsort().double() + 1
    r_pos = ranks[: pos.numel()].sum().item()
    n_p, n_n = pos.numel(), neg.numel()
    u = r_pos - n_p * (n_p + 1) / 2
    # pos should be SMALLER, so invert.
    return 1.0 - u / (n_p * n_n)


# ---------------------------------------------------------------------------
# Encoder construction
# ---------------------------------------------------------------------------

def build_encoders(args) -> list[tuple[str, object, str]]:
    """-> list of (label, encoder, env_config_name)."""
    out: list[tuple[str, object, str]] = []

    # Random projection: 84x84 grayscale, as configs/environment/mspacman_mfec_train.yaml
    out.append((
        "random_projection (64-d)",
        RandomProjectionEncoder(84 * 84, 64, seed=args.seed),
        "mspacman_mfec_train",
    ))

    if args.dinov2_weights or args.dinov2_random_init:
        from src.encoders.dino_v2_encoder import DINOv2Encoder

        if args.dinov2_random_init:
            # Architecture only; numerics (and key stability) match the real
            # model, representation quality does NOT.
            import torch.nn.functional as F

            class _RandomInitDINOv2(DINOv2Encoder):
                def __init__(self, model_name, image_size, repo_dir):
                    self.model_name, self.image_size = model_name, image_size
                    if repo_dir is not None:
                        self.model = torch.hub.load(
                            repo_dir, model_name, source="local", pretrained=False
                        )
                    else:
                        self.model = torch.hub.load(
                            "facebookresearch/dinov2", model_name, pretrained=False
                        )
                    self.state_dim = int(self.model.embed_dim)
                    self.model.eval()
                    for p in self.model.parameters():
                        p.requires_grad_(False)
                    self._mean = torch.tensor(
                        (0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                    self._std = torch.tensor(
                        (0.229, 0.224, 0.225)).view(1, 3, 1, 1)

            enc = _RandomInitDINOv2(
                args.dinov2_model, args.dinov2_image_size, args.dinov2_repo_dir
            )
            label = f"DINOv2 {args.dinov2_model} RANDOM-INIT ({enc.state_dim}-d)"
        else:
            enc = DINOv2Encoder(
                weights_path=args.dinov2_weights,
                model_name=args.dinov2_model,
                repo_dir=args.dinov2_repo_dir,
                image_size=args.dinov2_image_size,
            )
            label = f"DINOv2 {args.dinov2_model} ({enc.state_dim}-d)"

        out.append((label, enc, "mspacman_mfec_train_dinov2"))

    if args.vae_checkpoint:
        from src.encoders.vae_encoder import VAEEncoder

        out.append((
            "VAE (64-d)",
            VAEEncoder(args.vae_checkpoint, 1, 64, torch.device("cpu")),
            "mspacman_mfec_train",
        ))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=150,
                   help="frames to sample (default 150)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key-scale", type=float, default=1e5,
                   help="must match algorithm.key_scale (default 1e5)")
    p.add_argument("--dinov2-weights", default=None)
    p.add_argument("--dinov2-random-init", action="store_true",
                   help="build the ViT untrained: valid for key stability ONLY")
    p.add_argument("--dinov2-model", default="dinov2_vits14")
    p.add_argument("--dinov2-repo-dir", default=None)
    p.add_argument("--dinov2-image-size", type=int, default=224)
    p.add_argument("--vae-checkpoint", default=None)
    p.add_argument("--device", default="cpu",
                   help="device to embed on (default cpu). RUN THIS ON 'cuda' "
                        "TOO: the key-stability failure mode is a cuBLAS "
                        "shape-dependent-kernel effect and does not reproduce "
                        "on CPU, where reduction order is stable.")
    args = p.parse_args()

    dev = torch.device(args.device)
    print(f"[device ] embedding on {dev}\n")

    encoders = build_encoders(args)
    if len(encoders) == 1:
        print("NOTE: only the random-projection baseline was built. Pass "
              "--dinov2-weights (or --dinov2-random-init) to compare arms.\n")

    frame_cache: dict[str, torch.Tensor] = {}
    rows = []

    for label, enc, env_name in encoders:
        if env_name not in frame_cache:
            print(f"[collect] {env_name} ...", flush=True)
            frame_cache[env_name] = collect_frames(env_name, args.frames, args.seed)
        frames = frame_cache[env_name].to(dev)

        print(f"[embed  ] {label}  (obs {tuple(frames.shape[1:])}) ...", flush=True)
        emb = enc.embed(frames).cpu()
        bs, rp = key_stability(enc, frames, args.key_scale)
        g = geometry(emb)
        rows.append({
            "label": label,
            "dim": emb.shape[-1],
            "key_batch_single": bs,
            "key_repeat": rp,
            "auc": adjacency_auc(emb),
            **g,
        })

    w = max(len(r["label"]) for r in rows) + 2
    print("\n" + "=" * (w + 78))
    print(f"{'encoder':<{w}}{'dim':>5}{'key b/s':>10}{'key rep':>9}"
          f"{'adj AUC':>10}{'rel contr':>11}{'dist CV':>10}{'mean d':>12}")
    print("-" * (w + 78))
    for r in rows:
        print(f"{r['label']:<{w}}{r['dim']:>5}{r['key_batch_single']:>10.3f}"
              f"{r['key_repeat']:>9.3f}{r['auc']:>10.3f}{r['rel_contrast']:>11.3f}"
              f"{r['dist_cv']:>10.3f}{r['mean_dist']:>12.4g}")
    print("=" * (w + 78))

    print("\nInterpretation:")
    bad = False
    for r in rows:
        if r["key_batch_single"] < 1.0:
            bad = True
            print(f"  FAIL  {r['label']}: key match (batch vs single) = "
                  f"{r['key_batch_single']:.3f}, must be 1.000.\n"
                  "        Training embeds num_envs rows, evaluate() embeds 1 —\n"
                  "        they disagree on the key, so eval never takes the\n"
                  "        exact-match path and MFEC's Eq. (1) latching is lost.\n"
                  "        Fix: accumulate φ in float64, as RandomProjectionEncoder does.")
        if r["key_repeat"] < 1.0:
            bad = True
            print(f"  FAIL  {r['label']}: not run-to-run deterministic "
                  f"({r['key_repeat']:.3f}); violates src/encoders/base.py.")
        if r["auc"] == r["auc"] and r["auc"] < 0.65:
            bad = True
            print(f"  WEAK  {r['label']}: adjacency AUC {r['auc']:.3f} — φ barely\n"
                  "        separates near-identical game states from unrelated ones,\n"
                  "        so the kNN average is close to state-independent.")
    if not bad:
        print("  All encoders pass key stability and show usable discriminability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
