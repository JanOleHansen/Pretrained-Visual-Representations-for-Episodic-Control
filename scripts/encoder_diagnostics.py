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
    python scripts/encoder_diagnostics.py --clip --resnet --mae  # every PVM arm
    python scripts/encoder_diagnostics.py --clip --device cuda  # key stability!
    python scripts/encoder_diagnostics.py --mae --mae-pooling cls  # pooling ablation

Two arms need optional packages: ``--clip`` needs ``open_clip_torch``
(``uv sync --extra clip``) and ``--mae`` needs ``timm`` (``uv sync --extra
mae``); nothing else here does.  The CLIP ``--clip-model`` default carries the
``-quickgelu`` suffix, which is mandatory with the ``openai`` tag — see
"Checkpoint / architecture pairing" in ``src/encoders/clip_encoder.py``.

The MAE arm's ``--mae-pooling`` is worth running both ways.  MAE's CLS token is
never directly supervised by its reconstruction loss, so ``mean`` (over patch
tokens) is the conventional and expected-better choice; a ``cls`` row that
scores far worse is a property of MAE's objective, not of this code.  See
``src/encoders/mae_encoder.py``.

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

#: key_scale candidates tried when recommending a coarser quantisation, from
#: the current default downwards. Each step is 10x coarser.
_KEY_SCALE_LADDER = (1e5, 1e4, 1e3, 1e2, 1e1, 1e0)


def key_stability(
    encoder, frames: torch.Tensor, key_scale: float
) -> dict[str, float]:
    """How well the QEC hash key survives a change of batch shape.

    Returns ``batch_single`` / ``repeat`` match rates at ``key_scale``, plus the
    two numbers that make a failure actionable:

    ``drift``
        Largest absolute per-coordinate difference between embedding a frame
        inside a batch and embedding it alone. This is the physical quantity —
        float32 GEMM kernels on CUDA are chosen by batch size, so the reduction
        order (and therefore the last bits) changes with it.

    ``key_scale_ok``
        The coarsest-to-finest ladder search for the largest ``key_scale`` that
        still gives a **1.000** batch-vs-single match rate, or ``nan`` if even
        the coarsest does not. This matters because the failure is not "the
        embedding is wrong", it is "the quantisation grid is finer than the
        noise": a key is `round(phi * key_scale)` over `d` coordinates, so the
        key survives only if *no* coordinate lands within ``drift`` of a grid
        boundary — probability ``(1 - 2*drift*key_scale)**d``. With d=384 and
        drift=1e-6 that is ~1e-18 at key_scale=1e5, i.e. exactly the 0.000 a
        float32 ViT reports.
    """
    from src.algorithms.mfec import QEC

    # Embed once, reuse for every scale — a ViT forward per frame is the
    # expensive part and the ladder search must not multiply it.
    emb_batched = encoder.embed(frames).cpu()
    emb_repeat = encoder.embed(frames).cpu()
    emb_single = torch.cat(
        [encoder.embed(frames[i : i + 1]).cpu() for i in range(frames.shape[0])]
    )

    n = frames.shape[0]

    def match_rate(scale: float) -> float:
        qec = QEC(1, 8, 1, torch.device("cpu"), key_scale=scale)
        a, b = qec._make_keys(emb_batched), qec._make_keys(emb_single)
        return sum(a[i] == b[i] for i in range(n)) / n

    qec = QEC(1, 8, 1, torch.device("cpu"), key_scale=key_scale)
    keys_b, keys_r = qec._make_keys(emb_batched), qec._make_keys(emb_repeat)

    key_scale_ok = float("nan")
    for scale in _KEY_SCALE_LADDER:
        if scale > key_scale:
            continue
        if match_rate(scale) == 1.0:
            key_scale_ok = scale
            break

    return {
        "key_batch_single": match_rate(key_scale),
        "key_repeat": sum(keys_b[i] == keys_r[i] for i in range(n)) / n,
        "drift": float((emb_batched - emb_single).abs().max()),
        "key_scale_ok": key_scale_ok,
    }


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

    # Random projection: 84x84 grayscale, as configs/environment/atari_mfec_train.yaml
    out.append((
        "random_projection (64-d)",
        RandomProjectionEncoder(84 * 84, 64, seed=args.seed),
        "atari_mfec_train",
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

        out.append((label, enc, "atari_mfec_train_rgb"))

    # NOTE: this block is deliberately at function scope, NOT nested inside the
    # DINOv2 branch above.  Nesting it there (as it was until this was caught)
    # made the ResNet arms unreachable without DINOv2 flags, and made its `else`
    # bind to the *DINOv2* `if` — so a plain `--dinov2_weights` run fell into
    # the ResNet constructor with the import skipped, i.e. a NameError.
    if args.resnet or args.resnet_random_init:
        from src.encoders.resnet_encoder import ResNetEncoder

        if args.resnet_random_init:
            # Architecture only: numerics (and therefore key stability, which is
            # a BatchNorm/float32 property) match the real model; representation
            # quality does NOT.  Same trick as _RandomInitDINOv2 above.
            import torchvision.models as tvm

            class _RandomInitResNet(ResNetEncoder):
                def __init__(self, model_name, image_size):
                    import torch.nn as nn
                    self.model = tvm.get_model(model_name, weights=None)
                    self.state_dim = int(self.model.fc.in_features)
                    self.model.fc = nn.Identity()
                    self.image_size = image_size
                    self.model.eval()
                    for p in self.model.parameters():
                        p.requires_grad_(False)
                    self._mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                    self._std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

            enc = _RandomInitResNet(args.resnet_model, args.resnet_image_size)
            label = f"ResNet {args.resnet_model} RANDOM-INIT ({enc.state_dim}-d)"
        else:
            enc = ResNetEncoder(
                model_name=args.resnet_model,
                weights_path=args.resnet_weights,
                image_size=args.resnet_image_size,
            )
            label = f"ResNet {args.resnet_model} ({enc.state_dim}-d)"

        out.append((label, enc, "atari_mfec_train_rgb"))


    if args.clip or args.clip_weights or args.clip_random_init:
        from src.encoders.clip_encoder import CLIPEncoder

        # No _RandomInit subclass is needed here, unlike DINOv2/ResNet:
        # open_clip builds an untrained model when `pretrained` is None, and
        # CLIPEncoder passes that straight through.  Same caveat applies —
        # valid for the key-stability columns ONLY, not for AUC or contrast.
        if args.clip_random_init:
            enc = CLIPEncoder(
                weights_path=None,
                pretrained_tag=None,
                model_name=args.clip_model,
                image_size=args.clip_image_size,
                normalize=not args.clip_no_normalize,
            )
            label = f"CLIP {args.clip_model} RANDOM-INIT ({enc.state_dim}-d)"
        else:
            enc = CLIPEncoder(
                weights_path=args.clip_weights,
                pretrained_tag=args.clip_pretrained,
                model_name=args.clip_model,
                image_size=args.clip_image_size,
                normalize=not args.clip_no_normalize,
            )
            label = f"CLIP {args.clip_model} ({enc.state_dim}-d)"
        if enc.normalize:
            label += " L2"

        out.append((label, enc, "atari_mfec_train_rgb"))

    if args.mae or args.mae_weights or args.mae_random_init:
        from src.encoders.mae_encoder import MAEEncoder

        # No _RandomInit subclass is needed here, unlike DINOv2/ResNet:
        # MAEEncoder takes `pretrained`, which timm passes straight through, so
        # the untrained build reuses the real preprocessing rather than
        # duplicating it (the duplication is what left _RandomInitDINOv2 with
        # its own copy of the ImageNet constants).  Same caveat applies —
        # valid for the key-stability columns ONLY, not for AUC or contrast.
        enc = MAEEncoder(
            weights_path=args.mae_weights,
            model_name=args.mae_model,
            image_size=args.mae_image_size,
            pooling=args.mae_pooling,
            pretrained=not args.mae_random_init,
        )
        label = f"MAE {args.mae_model}"
        if args.mae_random_init:
            label += " RANDOM-INIT"
        # The pooling is in the label because it is the arm's load-bearing
        # choice: MAE's CLS token is undertrained, so a 'cls' row is measuring
        # something quite different from a 'mean' row.
        label += f" [{args.mae_pooling}] ({enc.state_dim}-d)"

        out.append((label, enc, "atari_mfec_train_rgb"))

    if args.vae_checkpoint:
        from src.encoders.vae_encoder import VAEEncoder

        out.append((
            "VAE (64-d)",
            VAEEncoder(args.vae_checkpoint, 1, 64, torch.device("cpu")),
            "atari_mfec_train",
        ))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=150,
                   help="frames to sample (default 150). Must comfortably "
                        "exceed --far or the adjacency AUC has no negative "
                        "pairs to score against.")
    p.add_argument("--near", type=int, default=2,
                   help="|i-j| <= near counts as the same game state (default 2)")
    p.add_argument("--far", type=int, default=50,
                   help="|i-j| >= far counts as unrelated (default 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key-scale", type=float, default=1e5,
                   help="must match algorithm.key_scale (default 1e5)")
    p.add_argument("--dinov2-weights", default=None)
    p.add_argument("--dinov2-random-init", action="store_true",
                   help="build the ViT untrained: valid for key stability ONLY")
    p.add_argument("--dinov2-model", default="dinov2_vits14")
    p.add_argument("--dinov2-repo-dir", default=None)
    p.add_argument("--dinov2-image-size", type=int, default=224)
    #resnet
    p.add_argument("--resnet", action="store_true",
                   help="add an ImageNet-pretrained ResNet arm (torchvision "
                        "downloads the weights unless --resnet-weights is given)")
    p.add_argument("--resnet-weights", default=None,
                   help="local .pth for offline boxes; implies --resnet")
    p.add_argument("--resnet-random-init", action="store_true",
                   help="build the ResNet untrained: valid for key stability ONLY")
    p.add_argument("--resnet-model", default="resnet18",
                   help="any torchvision resnet: resnet18/34/50/101/152")
    p.add_argument("--resnet-image-size", type=int, default=224)
    # clip -- needs the optional `open_clip_torch` package (uv add open_clip_torch)
    p.add_argument("--clip", action="store_true",
                   help="add a CLIP vision-tower arm (open_clip downloads the "
                        "weights unless --clip-weights is given)")
    p.add_argument("--clip-weights", default=None,
                   help="local open_clip checkpoint for offline boxes; implies --clip")
    p.add_argument("--clip-random-init", action="store_true",
                   help="build the CLIP ViT untrained: valid for key stability ONLY")
    p.add_argument("--clip-model", default="ViT-B-32-quickgelu",
                   help="open_clip architecture name, hyphenated: ViT-B-32, "
                        "ViT-B-16, ViT-L-14 (NOT OpenAI's 'ViT-B/32'). The "
                        "-quickgelu suffix is REQUIRED with --clip-pretrained "
                        "openai; LAION tags want the plain name.")
    p.add_argument("--clip-pretrained", default="openai",
                   help="open_clip pretrained tag, used only when "
                        "--clip-weights is absent (e.g. openai, laion2b_s34b_b79k)")
    p.add_argument("--clip-image-size", type=int, default=None,
                   help="default None = the backbone's native resolution")
    p.add_argument("--clip-no-normalize", action="store_true",
                   help="skip the L2 normalisation, i.e. do NOT put MFEC's "
                        "Euclidean kNN on CLIP's cosine metric (ablation)")

    # mae -- needs the optional `timm` package (uv sync --extra mae)
    p.add_argument("--mae", action="store_true",
                   help="add a frozen MAE ViT arm (timm downloads the weights "
                        "from the HuggingFace hub unless --mae-weights is given)")
    p.add_argument("--mae-weights", default=None,
                   help="local checkpoint for offline boxes; implies --mae. "
                        "Handed to timm as pretrained_cfg_overlay=dict(file=...), "
                        "so a raw mae_pretrain_vit_base.pth works too")
    p.add_argument("--mae-random-init", action="store_true",
                   help="build the MAE ViT untrained: valid for key stability "
                        "ONLY, but needs no network and no 350MB download")
    p.add_argument("--mae-model", default="vit_base_patch16_224.mae",
                   help="timm tag: vit_base_patch16_224.mae (768-d) or "
                        "vit_large_patch16_224.mae (1024-d). NOT .mae_ft_in1k, "
                        "which is MAE + supervised ImageNet finetuning and would "
                        "be a second supervised arm.")
    p.add_argument("--mae-image-size", type=int, default=224,
                   help="224 is the pretraining resolution; smaller cuts the "
                        "token count but interpolates the positional embeddings")
    p.add_argument("--mae-pooling", default="mean", choices=["mean", "cls"],
                   help="'mean' averages the PATCH tokens (what MAE evaluation "
                        "conventionally does — its CLS token is never directly "
                        "supervised by the reconstruction loss); 'cls' ablates "
                        "that and is expected to look worse")

    p.add_argument("--vae-checkpoint", default=None)
    p.add_argument("--device", default="cpu",
                   help="device to embed on (default cpu). RUN THIS ON 'cuda' "
                        "TOO: the key-stability failure mode is a cuBLAS "
                        "shape-dependent-kernel effect and does not reproduce "
                        "on CPU, where reduction order is stable.")
    args = p.parse_args()

    if args.frames < args.far * 2:
        p.error(
            f"--frames {args.frames} is too small for --far {args.far}: the "
            f"adjacency AUC needs pairs at least {args.far} apart, and with so "
            "few frames there are none (or too few to mean anything). Use "
            f"--frames >= {args.far * 2}, or lower --far."
        )

    dev = torch.device(args.device)
    print(f"[device ] embedding on {dev}\n")

    encoders = build_encoders(args)
    if len(encoders) == 1:
        print("NOTE: only the random-projection baseline was built. Pass "
              "--dinov2-weights, --resnet, --clip or --mae (or their "
              "*-random-init variants) to compare arms.\n")

    frame_cache: dict[str, torch.Tensor] = {}
    rows = []

    for label, enc, env_name in encoders:
        if env_name not in frame_cache:
            print(f"[collect] {env_name} ...", flush=True)
            frame_cache[env_name] = collect_frames(env_name, args.frames, args.seed)
        frames = frame_cache[env_name].to(dev)

        print(f"[embed  ] {label}  (obs {tuple(frames.shape[1:])}) ...", flush=True)
        emb = enc.embed(frames).cpu()
        ks = key_stability(enc, frames, args.key_scale)
        g = geometry(emb)
        rows.append({
            "label": label,
            "dim": emb.shape[-1],
            "auc": adjacency_auc(emb, near=args.near, far=args.far),
            **ks,
            **g,
        })

    w = max(len(r["label"]) for r in rows) + 2
    width = w + 96
    print("\n" + "=" * width)
    print(f"{'encoder':<{w}}{'dim':>5}{'key b/s':>10}{'key rep':>9}"
          f"{'drift':>11}{'key_scale*':>12}"
          f"{'adj AUC':>10}{'rel contr':>11}{'dist CV':>10}{'mean d':>12}")
    print("-" * width)
    for r in rows:
        ok = r["key_scale_ok"]
        ok_s = "none" if ok != ok else f"{ok:.0e}"          # nan check
        print(f"{r['label']:<{w}}{r['dim']:>5}{r['key_batch_single']:>10.3f}"
              f"{r['key_repeat']:>9.3f}{r['drift']:>11.2e}{ok_s:>12}"
              f"{r['auc']:>10.3f}{r['rel_contrast']:>11.3f}"
              f"{r['dist_cv']:>10.3f}{r['mean_dist']:>12.4g}")
    print("=" * width)
    print("drift      = max |phi(batch) - phi(single)| per coordinate")
    print("key_scale* = coarsest-to-finest largest key_scale giving key b/s = 1.000")

    print("\nInterpretation:")
    bad = False
    for r in rows:
        if r["key_batch_single"] < 1.0:
            bad = True
            ok = r["key_scale_ok"]
            print(f"  FAIL  {r['label']}: key match (batch vs single) = "
                  f"{r['key_batch_single']:.3f}, must be 1.000.\n"
                  "        Training embeds num_envs rows, evaluate() embeds 1 —\n"
                  "        they disagree on the key, so eval never takes the\n"
                  "        O(1) exact-match path.  MFEC then leans entirely on\n"
                  "        QEC's near-exact rescue (a relative-tolerance kNN\n"
                  f"        test), which is slower and unverified per encoder.\n"
                  f"        Cause: coordinate drift {r['drift']:.1e} against a "
                  f"quantisation step of {1 / args.key_scale:.1e}\n"
                  f"        ({r['dim']} coordinates must ALL survive rounding).")
            if ok == ok:                                    # not nan
                print(f"        Fix: algorithm.key_scale={ok:.0e} gives 1.000 "
                      f"for this encoder.\n"
                      "        Check first that it does not merge distinct "
                      "states — compare against\n"
                      "        the nearest-neighbour distance in 'mean d' x "
                      "(1 - 'rel contr').")
            else:
                print("        No key_scale in the ladder fixes it; this "
                      "encoder cannot use the\n        hash path at all.")
            print("        NOTE: 'accumulate in float64' only works for "
                  "RandomProjectionEncoder\n"
                  "        (one matmul). A float32 ViT/CNN cannot give "
                  "bit-identical output\n"
                  "        across batch shapes — cuBLAS picks kernels by "
                  "batch size.")
        if r["key_repeat"] < 1.0:
            bad = True
            print(f"  FAIL  {r['label']}: not run-to-run deterministic "
                  f"({r['key_repeat']:.3f}); violates src/encoders/base.py.")
        if r["auc"] != r["auc"]:          # NaN
            bad = True
            print(f"  ERROR {r['label']}: adjacency AUC could not be computed "
                  "(no valid pairs).\n        This row measured NOTHING — do not "
                  "read the other columns as a pass.")
        elif r["auc"] < 0.65:
            bad = True
            print(f"  WEAK  {r['label']}: adjacency AUC {r['auc']:.3f} — φ barely\n"
                  "        separates near-identical game states from unrelated ones,\n"
                  "        so the kNN average is close to state-independent.")
        if r["rel_contrast"] > 0.999:
            print(f"  NOTE  {r['label']}: relative contrast {r['rel_contrast']:.3f} "
                  "means the\n        nearest neighbour sits at ~zero distance, i.e. the "
                  "sample contains\n        duplicate frames. Expected on Atari (idle "
                  "frames); it makes this\n        one column uninformative, not wrong.")
    if not bad:
        print("  All encoders pass key stability and show usable discriminability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
