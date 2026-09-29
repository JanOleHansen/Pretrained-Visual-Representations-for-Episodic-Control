#!/usr/bin/env python3
"""How much of an embedding of an Atari frame is the same for every frame?

Replaces a footnote that quoted MAE at 0.66% and CLIP at 7.4% from two
different measurement scripts reading two different configuration files.  That
comparison was not admissible: the numbers were never produced under one
pipeline, and re-measuring all six arms uniformly moves both the values and
their ratio (11x -> 6x).  Everything here comes from ONE pass over the stored
matched-set embeddings, so the arms are comparable to each other.

For each encoder, ``probe_emb`` in the run archive holds its embedding of the
same 10,000 real Ms. Pac-Man frames.  Removing the component common to all of
them leaves the part a distance-based lookup can actually use; the fraction of
the original norm that survives is reported.  A small number means the encoder
maps every Atari frame to nearly the same vector, so an inverse-distance kernel
over that space is close to a flat average until training separates it.

    python make_collapse_numbers.py
"""
import os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
# ARM_ORDER's MFEC arms, in the order the thesis tables use.  `mae_smoke` and
# any other scratch dump in embeddings/ is deliberately not listed.
ARMS = [("random_projection", "RP (gray)"), ("random_projection_rgb", "RP (RGB)"),
        ("resnet", "ResNet"), ("dinov2", "DINOv2"), ("clip", "CLIP"), ("mae", "MAE")]
GAME = "MsPacman"


def residual_fraction(X):
    """Mean ||x - mean|| / mean ||x|| -- the share of the norm that is not shared."""
    X = np.asarray(X, dtype=np.float64)
    r = X - X.mean(axis=0)
    return float(np.linalg.norm(r, axis=1).mean() / np.linalg.norm(X, axis=1).mean())


def main():
    L = ["Embedding collapse on real Atari frames",
         f"  matched frame set, {GAME}, one row per MFEC arm",
         "  residual = mean ||x - mean(x)|| / mean ||x||, in percent.",
         "  Low means every frame embeds to nearly the same vector.",
         "  Measured in ONE pass over the stored probe_emb arrays, so the arms",
         "  are comparable to each other.", ""]
    vals = {}
    for enc, label in ARMS:
        fs = sorted(glob.glob(os.path.join(
            REPO, "embeddings", f"mfec_{GAME}_{enc}_seed*.npz")))
        if not fs:
            L.append(f"  {label:<11s} MISSING")
            continue
        per = []
        for f in fs:
            d = np.load(f, allow_pickle=True)
            per.append(residual_fraction(d["probe_emb"]))
        v = 100 * np.mean(per)
        vals[label] = v
        L.append(f"  {label:<11s} residual {v:6.2f}%   "
                 f"(sd {100*np.std(per, ddof=1) if len(per) > 1 else 0:.2f}, "
                 f"n={len(per)} seeds)")
    if "MAE" in vals and "CLIP" in vals:
        L += ["", f"  CLIP / MAE ratio: {vals['CLIP']/vals['MAE']:.1f}x"]
    p = os.path.join(HERE, "figures", "collapse_numbers.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote", p)


if __name__ == "__main__":
    main()
