#!/usr/bin/env python
"""Pull the embedding geometry out of a trained checkpoint, into one small npz.

This is the cluster-side half of the embedding analysis.  A NEC checkpoint is
~820 MB (fine-tuned ViT + RMSProp square_avg + the DND) and there are 165 runs,
so the checkpoints cannot come home — roughly 120 GB.  What comes home is what
this script writes: keys, values and probe embeddings at a few MB per run.

Two things are extracted per run, answering two different questions.

**The stored memory** — ``mem_*``.  The keys the run's episodic memory actually
holds, with the value recorded against each.  This is what the algorithm
retrieves from, so it is the honest object to ask "what does this encoder's
memory look like".  It is *not* comparable across arms without care: each arm's
memory holds the states its own policy visited, so both the state distribution
and the encoder differ between two pictures.

    MFEC  qec_state.action_states / action_values
          keys are raw φ(o) at the encoder's native width (RP 64, DINOv2 384,
          CLIP/ResNet 512, MAE 768); values are discounted MC returns at γ=1,
          **max-aggregated** by Eq. (1) and therefore upward-biased.
    NEC   dnd_state.action_keys / action_values
          keys are L2-normalised at embedding_dim=64 for every arm; values are
          N-step return estimates (N=100, γ=0.99) that have been α-blended on
          write *and* moved by the gradient step, so they are bootstrapped Q
          estimates rather than realised returns.

**The matched probe** — ``probe_*``.  This run's own encoder applied to the
shared frame set from ``build_probe_set.py``: the same states for every arm, so
the encoder is the only thing that varies.  For MFEC the encoder is restored
from ``encoder_state`` (which is what pins the random projection's matrix, so an
RP arm is reconstructed exactly rather than re-drawn from its seed).  For NEC
the fine-tuned ``embedding_net`` is rebuilt from the run's Hydra config and
loaded from ``policy_state_dict``, then L2-normalised — NEC normalises at the
DND boundary (nec.py:1295), so the un-normalised head output is not what the
memory is keyed on.

Because the frozen MFEC encoders are identical across seeds and games, probe
embeddings are cached by encoder signature and computed once per distinct φ
rather than once per run.  NEC's are fine-tuned and therefore unique per run.

Usage
-----
    # one run
    python scripts/extract_embeddings.py \
        --run-dir logs/train/runs/mfec_MsPacman_clip_seed42 \
        --probe-set probe_sets/probe_MsPacman.pt --out embeddings/

    # a whole grid (see scripts/extract_all.sh)
    python scripts/extract_embeddings.py --run-root logs/train/runs \
        --probe-root probe_sets --out embeddings/

Output: ``<out>/<run_name>.npz`` per run, plus ``<out>/probe_labels_<game>.npz``
carrying the returns/episode ids the probe embeddings are scored against, so the
analysis side never needs the multi-GB frame file.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: MFEC encoders whose φ depends on ``trainer.seed``.  Everything else loads
#: fixed pretrained weights, so one probe embedding serves all five seeds.
SEED_DEPENDENT = {"random_projection", "random_projection_rgb", "vae"}

#: encoder -> which observation pipeline of the probe set it consumes.
MFEC_PIPELINE = {"random_projection": "mfec_gray"}   # everything else: mfec_rgb

_RUN_RE = re.compile(r"^(?P<algo>mfec|nec)_(?P<game>[A-Za-z]+)_"
                     r"(?P<encoder>.+)_seed(?P<seed>\d+)$")

#: The two algorithms spell the game differently in ``run.name``.  MFEC's
#: experiment configs set ``run.game: ${game}``, so they inherit the ALE id
#: ("MsPacman"); NEC's name their env explicitly and set ``run.game: mspacman``
#: (configs/experiment/nec/mspacman.yaml:30).  Both forms are directory names on
#: the cluster, and the probe sets are keyed by the ALE id, so the token is
#: normalised on the way in rather than special-cased at every use.
GAME_CANON = {"mspacman": "MsPacman", "qbert": "Qbert", "frostbite": "Frostbite"}


def parse_run_name(name: str) -> dict:
    """Split ``mfec_MsPacman_random_projection_rgb_seed42`` into its fields.

    The encoder token itself contains underscores, which is why this is a regex
    anchored on the seed suffix rather than a ``split("_")``.
    """
    m = _RUN_RE.match(name)
    if not m:
        raise ValueError(f"unparseable run name: {name!r}")
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    d["algo"] = d["algo"].upper()
    game = d["game"]
    if game.lower() not in GAME_CANON:
        raise ValueError(f"unknown game token {game!r} in run name {name!r}")
    d["game"] = GAME_CANON[game.lower()]
    return d


# ---------------------------------------------------------------------------
# Memory contents
# ---------------------------------------------------------------------------

def read_memory(extra: dict) -> dict:
    """Flatten a serialised QEC/DND into (keys, values, action) arrays.

    Both memories emit one array per action with only the live slots filled
    (``mfec.py:1474``, ``nec.py:1078``), so the per-action lists are simply
    concatenated and an action id carried alongside.  An action whose table is
    empty contributes nothing and is recorded in ``sizes`` as 0.
    """
    if "qec_state" in extra:
        state, kf, vf = extra["qec_state"], "action_states", "action_values"
    elif "dnd_state" in extra:
        state, kf, vf = extra["dnd_state"], "action_keys", "action_values"
    else:
        raise KeyError("checkpoint carries neither qec_state nor dnd_state")

    keys_l, vals_l, act_l = [], [], []
    sizes = []
    per_action_keys = state[kf]
    per_action_vals = state[vf]

    for a in range(int(state["num_actions"])):
        k = None if per_action_keys is None else per_action_keys[a]
        if k is None or len(k) == 0:
            sizes.append(0)
            continue
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(per_action_vals[a], dtype=np.float64)
        if len(k) != len(v):
            raise ValueError(f"action {a}: {len(k)} keys vs {len(v)} values")
        keys_l.append(k)
        vals_l.append(v)
        act_l.append(np.full(len(k), a, dtype=np.int16))
        sizes.append(len(k))

    if not keys_l:
        raise ValueError("memory is empty — nothing was ever written")

    return {
        "keys":   np.concatenate(keys_l, axis=0),
        "values": np.concatenate(vals_l, axis=0),
        "action": np.concatenate(act_l, axis=0),
        "sizes":  np.asarray(sizes, dtype=np.int64),
    }


def subsample(n: int, cap: int, seed: int) -> np.ndarray:
    """Indices of at most ``cap`` rows, drawn without replacement, seeded.

    Uniform over rows rather than balanced over actions: the per-action fill
    levels are themselves a property of the run worth preserving in the plot,
    and re-balancing them would draw a picture of a memory that does not exist.
    """
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


# ---------------------------------------------------------------------------
# Rebuilding this run's encoder
# ---------------------------------------------------------------------------

def load_run_config(run_dir: Path):
    """Read the Hydra config Hydra itself wrote next to the run.

    Rebuilding φ from the run's own resolved config is the only way to be sure
    the probe uses the encoder the run actually trained with — reconstructing it
    from the experiment YAML would silently miss any command-line override.
    """
    from omegaconf import OmegaConf

    path = run_dir / ".hydra" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no {path} — cannot rebuild this run's encoder")
    return OmegaConf.load(path)


def build_mfec_encoder(cfg, encoder_state, obs_shape, device):
    """Rebuild MFEC's frozen φ and restore its saved state.

    Mirrors ``MFECAlgorithm.setup`` (mfec.py:320): the algorithm object is
    instantiated from the run's own config so every encoder keyword comes from
    one place, then ``make_encoder`` is called off its attributes.  Loading
    ``encoder_state`` afterwards is what makes a random-projection arm exact —
    the projection matrix is drawn at construction, and the checkpoint's copy is
    the one the run keyed its memory with.
    """
    import hydra
    from src.encoders.factory import make_encoder

    algo = hydra.utils.instantiate(cfg.algorithm, _convert_="partial")

    encoder = make_encoder(
        algo.encoder_name,
        obs_flat_dim=int(np.prod(obs_shape[-3:])),
        in_channels=obs_shape[-3],
        state_dim=algo.state_dim,
        vae_checkpoint_path=algo.vae_checkpoint,
        device=device,
        seed=algo.seed,
        dinov2_weights_path=algo.dinov2_weights,
        dinov2_model_name=algo.dinov2_model_name,
        dinov2_repo_dir=algo.dinov2_repo_dir,
        dinov2_image_size=algo.dinov2_image_size,
        resnet_weights_path=algo.resnet_weights_path,
        resnet_model_name=algo.resnet_model_name,
        resnet_image_size=algo.resnet_image_size,
        clip_weights_path=algo.clip_weights_path,
        clip_model_name=algo.clip_model_name,
        clip_pretrained_tag=algo.clip_pretrained_tag,
        clip_image_size=algo.clip_image_size,
        clip_normalize=algo.clip_normalize,
        clip_interpolation=algo.clip_interpolation,
        mae_weights_path=algo.mae_weights_path,
        mae_model_name=algo.mae_model_name,
        mae_image_size=algo.mae_image_size,
        mae_pooling=algo.mae_pooling,
    )
    if encoder_state is not None:
        encoder.load_state(encoder_state)

    def embed(batch: torch.Tensor) -> torch.Tensor:
        return encoder.embed(batch)

    return embed


def build_nec_encoder(cfg, policy_state_dict, obs_shape, device):
    """Rebuild NEC's fine-tuned embedding network from this run's weights.

    ``F.normalize`` is applied because NEC does: the DND is read and written
    through ``F.normalize(h, dim=-1)`` (nec.py:1295, :1800, :2158), so the raw
    head output is not the space the memory is keyed on and probing it would
    describe a geometry the algorithm never uses.
    """
    import hydra

    algo = hydra.utils.instantiate(cfg.algorithm, _convert_="partial")
    net = algo._make_embedding_network(tuple(obs_shape), algo.embedding_dim)
    net.load_state_dict(policy_state_dict)
    net = net.to(device).eval()

    def embed(batch: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(net(batch), dim=-1)

    return embed


def embed_frames(embed_fn, frames: torch.Tensor, device, batch_size: int = 128):
    """Run φ over the probe frames in batches, returning (N, d) float32.

    The uint8 RGB stream is divided by 255 to undo ``build_probe_set``'s
    byte-exact storage, which puts it back in the [0, 1] range ``ToTensorImage``
    produced and every encoder expects.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            chunk = frames[i: i + batch_size]
            if chunk.dtype == torch.uint8:
                chunk = chunk.float().div_(255.0)
            else:
                chunk = chunk.float()
            out.append(embed_fn(chunk.to(device)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def probe_signature(info: dict) -> str:
    """Cache key for a probe embedding — one entry per *distinct φ*.

    MFEC's pretrained encoders load fixed weights, so all five seeds share one
    embedding and it is computed once.  The random projections are drawn from
    the seed, and every NEC encoder is fine-tuned by its own run, so those stay
    per-run.
    """
    if info["algo"] == "NEC" or info["encoder"] in SEED_DEPENDENT:
        return f"{info['game']}_{info['algo']}_{info['encoder']}_seed{info['seed']}"
    return f"{info['game']}_{info['algo']}_{info['encoder']}"


def write_probe_labels(probe: dict, out_dir: Path, game: str) -> None:
    """Emit the probe set's labels once per game, without the frames.

    The frame file is 1-2 GB and stays on the cluster; the analysis only needs
    these columns, which are a few hundred KB.
    """
    path = out_dir / f"probe_labels_{game}.npz"
    if path.exists():
        return
    np.savez_compressed(
        path,
        game=game,
        seed=probe["seed"],
        frames=probe["frames"],
        n_actions=probe["n_actions"],
        policy=probe.get("policy", "uniform_random"),
        reward_horizon=probe["reward_horizon"],
        action=probe["action"].numpy(),
        reward=probe["reward"].numpy(),
        done=probe["done"].numpy(),
        episode=probe["episode"].numpy(),
        return_g1=probe["return_g1"].numpy(),
        return_g099=probe["return_g099"].numpy(),
        complete=probe["complete"].numpy(),
        reward_soon=probe["reward_soon"].numpy(),
    )
    print(f"  wrote {path.name}")


def process_run(run_dir: Path, out_dir: Path, probe: dict | None,
                probe_cache: Path, args) -> str:
    info = parse_run_name(run_dir.name)
    out_path = out_dir / f"{run_dir.name}.npz"
    if out_path.exists() and not args.overwrite:
        return "skip (exists)"

    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        return f"MISSING {ckpt_path.relative_to(run_dir)}"

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = state.extra or {}

    mem = read_memory(extra)
    idx = subsample(len(mem["keys"]), args.max_rows, args.subsample_seed)

    payload = {
        "algo": info["algo"], "game": info["game"],
        "encoder": info["encoder"], "seed": info["seed"],
        "step": int(state.step),
        "mem_keys":   mem["keys"][idx],
        "mem_values": mem["values"][idx],
        "mem_action": mem["action"][idx],
        "mem_sizes":  mem["sizes"],
        "mem_total":  len(mem["keys"]),
        "collected_frames": int(extra.get("collected_frames", -1)),
    }

    if probe is not None and not args.no_probe:
        pipeline = ("nec_gray4" if info["algo"] == "NEC"
                    else MFEC_PIPELINE.get(info["encoder"], "mfec_rgb"))
        if pipeline not in probe["pipelines"]:
            return f"probe set has no {pipeline!r} stream"

        sig = probe_signature(info)
        cache_file = probe_cache / f"{sig}.npy"
        if cache_file.exists():
            emb = np.load(cache_file)
        else:
            frames = probe["pipelines"][pipeline]
            device = torch.device(args.device)
            cfg = load_run_config(run_dir)
            if info["algo"] == "MFEC":
                embed_fn = build_mfec_encoder(
                    cfg, extra.get("encoder_state"), frames.shape[1:], device)
            else:
                embed_fn = build_nec_encoder(
                    cfg, state.policy_state_dict, frames.shape[1:], device)
            emb = embed_frames(embed_fn, frames, device, args.batch_size)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, emb)

        payload["probe_emb"] = emb
        payload["probe_pipeline"] = pipeline
        payload["probe_signature"] = sig

    np.savez_compressed(out_path, **payload)
    mb = out_path.stat().st_size / 1e6
    return (f"ok  keys {len(idx)}/{len(mem['keys'])} d={mem['keys'].shape[1]}  "
            f"{mb:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="one run directory")
    src.add_argument("--run-root", help="directory of run directories")
    p.add_argument("--probe-set", help="probe_<Game>.pt for --run-dir")
    p.add_argument("--probe-root", help="directory of probe_<Game>.pt files")
    p.add_argument("--out", default="embeddings", help="output directory")
    p.add_argument("--checkpoint", default="last.pt",
                   help="checkpoint filename inside <run>/checkpoints")
    p.add_argument("--max-rows", type=int, default=20_000,
                   help="cap on memory keys kept per run (default 20000)")
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-probe", action="store_true",
                   help="memory keys only; skip the matched probe embeddings")
    p.add_argument("--only", nargs="+", default=None,
                   help="substrings; a run is processed if any occurs in its name")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_cache = out_dir / "_probe_cache"

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = sorted(d for d in Path(args.run_root).iterdir()
                          if d.is_dir() and _RUN_RE.match(d.name))
    if args.only:
        run_dirs = [d for d in run_dirs if any(s in d.name for s in args.only)]
    if not run_dirs:
        raise SystemExit("no run directories matched")

    probes: dict[str, dict] = {}

    def get_probe(game: str):
        if args.no_probe:
            return None
        if game not in probes:
            if args.probe_set:
                path = Path(args.probe_set)
            elif args.probe_root:
                path = Path(args.probe_root) / f"probe_{game}.pt"
            else:
                return None
            if not path.exists():
                print(f"  ! no probe set at {path}; memory keys only")
                probes[game] = None
            else:
                probes[game] = torch.load(path, map_location="cpu",
                                          weights_only=False)
                write_probe_labels(probes[game], out_dir, game)
        return probes[game]

    print(f"{len(run_dirs)} run(s) -> {out_dir}  (device={args.device})")
    failures = []
    for i, d in enumerate(run_dirs, 1):
        t0 = time.time()
        try:
            game = parse_run_name(d.name)["game"]
            msg = process_run(d, out_dir, get_probe(game), probe_cache, args)
        except Exception as exc:                       # noqa: BLE001
            msg = f"FAILED {type(exc).__name__}: {exc}"
            failures.append((d.name, traceback.format_exc()))
        print(f"[{i:3d}/{len(run_dirs)}] {d.name:52s} {msg}  "
              f"({time.time() - t0:.0f}s)")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, tb in failures:
            print(f"\n--- {name}\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
