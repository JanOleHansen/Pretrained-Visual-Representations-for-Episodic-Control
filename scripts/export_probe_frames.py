#!/usr/bin/env python
"""Export a probe set's RGB frames to a plain .npy the analysis venv can read.

The probe set is a ``torch.save`` bundle, and the figure venv (`.venv-umap`)
deliberately has no torch -- it holds umap-learn and scikit-learn only, kept
apart from the RL environment for the same reason `.venv-rliable` is.  Rather
than install a 2 GB dependency to read one uint8 array, the frames are written
once as a bare ``.npy`` that numpy can memory-map, so the plotting script never
loads the whole gigabyte at all.

    python scripts/export_probe_frames.py --game MsPacman

Writes ``probe_sets/frames_<game>.npy`` -- (T, 3, H, W) uint8, the mfec_rgb
stream, in the probe set's own row order so it indexes identically to
``probe_labels_<game>.npz``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="MsPacman")
    p.add_argument("--probe-dir", default="probe_sets")
    p.add_argument("--pipeline", default="mfec_rgb",
                   help="which observation stream to export (default mfec_rgb, "
                        "the one that looks like the game)")
    args = p.parse_args()

    src = Path(args.probe_dir) / f"probe_{args.game}.pt"
    if not src.exists():
        raise SystemExit(f"no {src}")

    payload = torch.load(src, map_location="cpu", weights_only=False)
    if args.pipeline not in payload["pipelines"]:
        raise SystemExit(f"{src} has no {args.pipeline!r} stream "
                         f"(has: {sorted(payload['pipelines'])})")

    frames = payload["pipelines"][args.pipeline].numpy()
    if frames.dtype != np.uint8:
        # Only the RGB stream is stored as bytes; the grayscale ones are float16
        # in [0, 1] and have to be scaled before they mean anything as an image.
        frames = (np.clip(frames.astype(np.float32), 0, 1) * 255).astype(np.uint8)

    dst = Path(args.probe_dir) / f"frames_{args.game}.npy"
    np.save(dst, frames)
    print(f"wrote {dst}  {frames.shape} {frames.dtype}  "
          f"{dst.stat().st_size / 1e6:.0f} MB")

    # The labels travel with the frames so the figure script can prove, without
    # torch, that this rollout is the one the cluster's embeddings were computed
    # on.  Cheap, and it is the whole basis for trusting the thumbnails.
    side = Path(args.probe_dir) / f"labels_{args.game}.npz"
    np.savez_compressed(
        side,
        **{k: payload[k].numpy() for k in
           ("action", "reward", "done", "episode",
            "return_g1", "return_g099", "complete", "reward_soon")})
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
