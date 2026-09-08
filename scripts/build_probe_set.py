#!/usr/bin/env python
"""Build the *matched* frame set the embedding analysis is probed on.

Why this exists
---------------
The obvious way to look at an encoder's geometry is to UMAP the keys a trained
run actually stored (``scripts/extract_embeddings.py``, which does exactly
that).  Those key sets are **not comparable across arms**: each arm's memory
holds the states *its own* policy visited, so a difference between the CLIP
picture and the RP picture is part encoder and part behaviour, with no way to
tell the two apart from the plot.

This script removes that confound by fixing the data.  One seeded rollout per
game, one action stream, replayed through every observation pipeline the study
uses.  Every encoder is then asked about the *same frames*, so φ is the only
thing that varies — which is the premise the whole encoder ablation rests on,
applied to the embeddings instead of to the returns.

The three pipelines are not cosmetic variants; they are what the arms really
see, and they disagree about channels, resolution and history:

    mfec_gray   configs/environment/atari_mfec_train.yaml       (1, 84, 84)
                -> random_projection
    mfec_rgb    configs/environment/atari_mfec_train_rgb.yaml   (3, 210, 160)
                -> random_projection_rgb, resnet, dinov2, clip, mae
    nec_gray4   configs/environment/<game>_nec_train.yaml       (4, 84, 84)
                -> every NEC arm (grayscale, CatFrames N=4)

They are collected by *replaying the same action sequence through three
separately built envs* rather than by deriving one from another.  Deriving
would mean re-implementing GrayScale/Resize/CatFrames and their episode-boundary
behaviour here, and quietly diverging from the real pipeline the first time a
transform changes.  Replaying is faithful by construction, and it is checkable:
ALE is deterministic at ``repeat_action_probability=0`` and the only other
source of variation, ``NoopResetEnv``, is seeded — so the three rollouts must
produce *identical* reward and done streams.  The script asserts that, and the
assertion is the proof the frames line up index-for-index.

The label
---------
``return_to_go`` is the discounted Monte-Carlo return from each step, computed
backwards within an episode exactly as ``MFECAlgorithm.step`` computes it:

    G_t = r_{t+1} + γ · G_{t+1},   G reset at every episode boundary

γ=1.0 (``return_g1``) is MFEC's value (configs/algorithm/mfec_atari.yaml) and
γ=0.99 (``return_g099``) is NEC's, so each algorithm's probe can be scored
against the target its own memory actually stores.

The trailing partial episode has no terminal to count back from, so its returns
are truncated and wrong.  Those rows are flagged ``complete=False`` and every
consumer drops them; they are kept in the file only so the frame indices stay
aligned with the raw rollout.

The behaviour policy is uniform-random by default.  That is a real limitation
and it is stated rather than hidden: a random policy does not visit the states a
trained agent visits, so this set measures the geometry of φ over the *reachable
early-game distribution*, not over the on-policy one.  It buys the thing the
checkpoint keys cannot give — an identical input set for every arm — and the two
analyses are meant to be read together.  ``--policy checkpoint`` replays a
trained agent's greedy policy from one MFEC checkpoint instead, which moves the
distribution on-policy for *one reference arm* while keeping it identical across
the encoders being compared.

Usage
-----
    python scripts/build_probe_set.py --game MsPacman --out probe_sets/
    python scripts/build_probe_set.py --game Qbert --frames 10000 \
        --pipelines mfec_rgb nec_gray4

Output: ``<out>/probe_<game>.pt``, roughly 1.2 GB per game at the default
10,000 frames (the RGB stream dominates; it is stored as uint8).  It never
leaves the cluster — only the embeddings computed from it come home.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.factory import make_env  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "environment"

#: pipeline name -> environment config, ``{game}`` filled in per game.
PIPELINES = {
    "mfec_gray": "atari_mfec_train",
    "mfec_rgb":  "atari_mfec_train_rgb",
    "nec_gray4": "{game_lower}_nec_train",
}

#: ALE game id -> the token the per-game NEC env configs are named with.
GAME_SLUG = {"MsPacman": "mspacman", "Qbert": "qbert", "Frostbite": "frostbite"}


def _env_from_config(config_name: str, game: str, seed: int):
    """Build one env from ``configs/environment/<config_name>.yaml``.

    The MFEC env configs name themselves ``ALE/${oc.select:game,MsPacman}-v5``,
    i.e. they read Hydra's global ``game`` key that this script does not
    compose.  Resolving them standalone would silently fall back to Ms. Pac-Man
    for every game, so ``name`` is overwritten explicitly *after* the load and
    the per-game NEC configs (which hard-code their own name) are checked
    against it rather than overwritten blind.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(CONFIG_DIR / f"{config_name}.yaml")
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
              if k != "_target_"}

    want = f"ALE/{game}-v5"
    if kwargs.get("name") not in (want, None) and "${" not in str(kwargs.get("name")):
        # A per-game config that names a different game: the caller mapped the
        # game to the wrong file, which would silently probe the wrong ROM.
        if not kwargs["name"].startswith(f"ALE/{game}"):
            raise SystemExit(
                f"{config_name}.yaml names {kwargs['name']!r}, not {want!r} — "
                "GAME_SLUG and the config file disagree about the game."
            )
    kwargs["name"] = want

    return make_env(**kwargs, num_envs=1, device="cpu", seed=seed)


def rollout(config_name: str, game: str, n_frames: int, seed: int,
            actions: np.ndarray | None = None):
    """Roll one env for ``n_frames`` steps under a fixed action stream.

    Returns ``(obs, reward, done, actions, n_actions)``.  ``obs`` is uint8 when
    the pipeline yields RGB in [0, 1] (exactly invertible: ``ToTensorImage``
    only divides a uint8 frame by 255) and float16 otherwise, because the
    grayscale streams have been through ``Resize``'s interpolation and are no
    longer integral.

    ``actions`` is drawn from ``default_rng(seed)`` when not supplied, and is
    returned so the next pipeline can be driven by the identical stream.
    """
    env = _env_from_config(config_name, game, seed)

    # NoopResetEnv draws its no-op count from the *global* torch RNG (it has no
    # _set_seed hook), and with repeat_action_probability=0 that count is the
    # only thing distinguishing one episode opening from another.  Seeding here
    # is what makes the three pipelines replay the same game.
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_actions = int(env.action_spec.space.n)
    if actions is None:
        rng = np.random.default_rng(seed)
        actions = rng.integers(n_actions, size=n_frames).astype(np.int64)
    elif len(actions) < n_frames:
        raise ValueError(f"need {n_frames} actions, got {len(actions)}")

    td = env.reset()
    obs_buf, rew_buf, done_buf = [], [], []

    for t in range(n_frames):
        # The observation is taken BEFORE the step, so obs[t] is s_t against
        # reward[t] = r_{t+1} -- the pairing MFECAlgorithm.step uses (it embeds
        # batch[obs_key] and reads batch["next", "reward"]).  Recording the
        # post-step frame instead would shift every state one place against its
        # own return, which is invisible in the plots and silently costs the
        # probe most of its signal.
        # make_env(num_envs=1) returns a BARE env, so pixels is already
        # (C, H, W).  Do not squeeze: on the grayscale pipelines C == 1 and a
        # squeeze would strip the channel axis, leaving (84, 84) frames that no
        # encoder's embed() accepts.
        pix = td["pixels"].clone()
        td["action"] = torch.tensor(int(actions[t]))
        td = env.step(td)
        nxt = td["next"]

        obs_buf.append(pix)
        rew_buf.append(float(nxt["reward"].sum()))
        done_buf.append(bool(nxt["done"].any()))

        if done_buf[-1]:
            td = env.reset()
        else:
            td = nxt.clone()

    env.close()

    obs = torch.stack(obs_buf)
    if obs.ndim != 4:
        raise SystemExit(
            f"{config_name}: expected (T, C, H, W) frames, got {tuple(obs.shape)} — "
            "the env is batched or a transform changed the observation rank."
        )
    # RGB frames are a uint8 image divided by 255; recover the bytes exactly.
    if obs.shape[-3] == 3:
        obs = (obs * 255.0).round().clamp(0, 255).to(torch.uint8)
    else:
        obs = obs.to(torch.float16)

    return (obs,
            np.asarray(rew_buf, dtype=np.float64),
            np.asarray(done_buf, dtype=bool),
            actions[:n_frames],
            n_actions)


def returns_to_go(rewards: np.ndarray, dones: np.ndarray, gamma: float):
    """Discounted MC return from every step, reset at episode boundaries.

    Mirrors ``MFECAlgorithm.step``'s backward recursion (mfec.py, "Discounted
    return for the complete portion"): ``G ← r_{t+1} + γ·G``, with G forced to 0
    at each terminal so no return leaks across an episode boundary.

    Returns ``(G, complete)``.  ``complete`` is False for the trailing steps
    after the last terminal — their G counts back from an arbitrary cut of the
    rollout rather than from a real episode end, so it under-states the true
    return by however much the episode had left to pay out.
    """
    n = len(rewards)
    G = np.empty(n, dtype=np.float64)
    running = 0.0
    for t in range(n - 1, -1, -1):
        if dones[t]:
            running = 0.0
        running = rewards[t] + gamma * running
        G[t] = running

    complete = np.zeros(n, dtype=bool)
    ends = np.flatnonzero(dones)
    if len(ends):
        complete[: ends[-1] + 1] = True
    return G, complete


def episode_index(dones: np.ndarray) -> np.ndarray:
    """Per-step episode id, so a CV split can hold whole episodes out.

    Splitting at random over steps would put step t in train and step t+1 in
    test.  Consecutive frames are near-identical and carry near-identical
    returns, so that leaks the target and inflates every R² in the study —
    exactly the failure mode a probe result is easiest to be wrong about.
    """
    ep = np.zeros(len(dones), dtype=np.int64)
    ep[1:] = np.cumsum(dones[:-1])
    return ep


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True, choices=sorted(GAME_SLUG))
    p.add_argument("--frames", type=int, default=10_000,
                   help="rollout length in agent steps (default 10000)")
    p.add_argument("--seed", type=int, default=12345,
                   help="fixes the action stream AND the no-op draws; the probe "
                        "set is only reproducible if this is held")
    p.add_argument("--pipelines", nargs="+", default=sorted(PIPELINES),
                   choices=sorted(PIPELINES))
    p.add_argument("--reward-horizon", type=int, default=25,
                   help="N for the 'reward within the next N steps' "
                        "classification label (default 25)")
    p.add_argument("--out", default="probe_sets", help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = GAME_SLUG[args.game]
    streams, ref = {}, None
    shared_actions = None

    for name in args.pipelines:
        config_name = PIPELINES[name].format(game_lower=slug)
        t0 = time.time()
        obs, rew, done, acts, n_actions = rollout(
            config_name, args.game, args.frames, args.seed, shared_actions)
        shared_actions = acts
        print(f"  {name:10s} {tuple(obs.shape)} {obs.dtype} "
              f"{obs.numel() * obs.element_size() / 1e6:7.0f} MB  "
              f"{time.time() - t0:5.0f}s  "
              f"episodes={int(done.sum())}  score={rew.sum():.0f}")

        # The whole design rests on the three rollouts being the same game.
        # Observation transforms cannot change reward or termination, so any
        # disagreement means the envs diverged (a stray RNG draw, a transform
        # that is not observation-only) and the frames do NOT line up.
        if ref is None:
            ref = (rew, done)
        else:
            if not np.array_equal(ref[0], rew) or not np.array_equal(ref[1], done):
                raise SystemExit(
                    f"pipeline {name!r} diverged from {args.pipelines[0]!r}: "
                    "reward/done streams differ, so the frames are not matched. "
                    "Do not use this probe set."
                )
        streams[name] = obs

    rewards, dones = ref
    g1,  complete = returns_to_go(rewards, dones, 1.0)
    g099, _       = returns_to_go(rewards, dones, 0.99)
    episodes = episode_index(dones)

    # Bioacoustics-style classification target: does a reward land within the
    # next H steps?  A linear probe's balanced accuracy on this is the direct
    # analogue of probing a frozen audio encoder for a call type -- it asks
    # whether the embedding carries an *event*, not a magnitude, and so it is
    # not dominated by the few huge-reward frames the way R^2 on return is.
    # The window must not reach across an episode boundary, so it is computed
    # per episode: a reward in the next game is not a reward this state leads
    # to.  `episodes` already carries the segmentation.
    H = args.reward_horizon
    hit = (rewards > 0)
    reward_soon = np.zeros(len(hit), dtype=bool)
    for ep in np.unique(episodes):
        rows = np.flatnonzero(episodes == ep)
        seg = hit[rows].astype(np.int64)
        # reward[i] is r_{i+1}, the payout for acting in s_i, so the window
        # "a reward follows s_i within H steps" is seg[i : i+H].
        cs = np.cumsum(np.r_[0, seg])
        lo = np.arange(len(seg))
        upper = np.minimum(lo + H, len(seg))
        reward_soon[rows] = (cs[upper] - cs[lo]) > 0

    payload = {
        "game":        args.game,
        "seed":        args.seed,
        "frames":      args.frames,
        "n_actions":   n_actions,
        "policy":      "uniform_random",
        "pipelines":   {k: v for k, v in streams.items()},
        "pipeline_config": {k: PIPELINES[k].format(game_lower=slug)
                            for k in streams},
        "action":      torch.from_numpy(shared_actions),
        "reward":      torch.from_numpy(rewards),
        "done":        torch.from_numpy(dones),
        "episode":     torch.from_numpy(episodes),
        "return_g1":   torch.from_numpy(g1),
        "return_g099": torch.from_numpy(g099),
        "complete":    torch.from_numpy(complete),
        "reward_soon": torch.from_numpy(reward_soon),
        "reward_horizon": H,
    }

    path = out_dir / f"probe_{args.game}.pt"
    torch.save(payload, path)
    size_mb = path.stat().st_size / 1e6

    print(f"\nwrote {path}  ({size_mb:.0f} MB)")
    print(f"  episodes           {int(dones.sum())}")
    print(f"  usable rows        {int(complete.sum())} / {args.frames}")
    print(f"  return (g=1)       mean {g1[complete].mean():.1f}  "
          f"sd {g1[complete].std():.1f}  max {g1[complete].max():.0f}")
    print(f"  reward within {H:>3}  {reward_soon[complete].mean():.3f} of rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
