#!/usr/bin/env python3
"""Has anything converged at 100k?  -- the control the cutoff-dependence claim needs.

Every number in the Results section is read off the LAST evaluation point, which
is only a statement about the encoders if the curves have leveled off by then.
They have not, and not uniformly: on Ms. Pac-Man MAE and ResNet are still
climbing steeply at 100k while CLIP and DINOv2 have flattened, so the final-step
ranking is partly a ranking of *when* an arm saturates.

This script writes figures/convergence_numbers.txt, which Section 5 and the
limitations quote.  It reads the same W&B cache as make_rliable_figures.py and
adds nothing to it.

    python make_convergence_numbers.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared run loader (make_figures.py) sits next to this script in
# analysis/; "DECK" is kept as the name for where it lives.
DECK = HERE
sys.path.insert(0, DECK)
import make_figures as mf                                       # noqa: E402

CACHE = os.environ.get("PVR_CACHE", os.path.join(HERE, "data", "runs"))
SEEDS = [42, 43, 44, 45, 46]


def curves(meta, algo, game, enc):
    """(n_seeds, n_evals) HNS over the budget, or None if any seed is short."""
    import pandas as pd
    lo, hi = mf.REF[game]
    out = []
    for sd in SEEDS:
        r = meta[(meta.algo == algo) & (meta.game == game) &
                 (meta.encoder == enc) & (meta.seed == sd)]
        if not len(r):
            return None
        h = pd.read_csv(os.path.join(CACHE, "wandb_cache",
                                     f"{r.iloc[0]['id']}.csv"))
        h = h[h["eval/return_mean"].notna()].sort_values("_step")
        out.append(((h["eval/return_mean"].to_numpy() - lo) / (hi - lo)))
    n = min(len(c) for c in out)
    return np.array([c[:n] for c in out])


def main():
    meta = mf.load(CACHE)
    L = ["Convergence at the 100k cutoff",
         "  mean human-normalized score (%) over 5 seeds at each evaluation point",
         "  'half' is the mid-budget evaluation; 'final' is the last one.",
         "  A large final-half gap means the arm was still improving when the",
         "  budget ran out, so its final score is a learning RATE, not a level.",
         ""]
    for algo in ["MFEC", "NEC"]:
        arms = [e for e in mf.ARM_ORDER
                if e in set(meta[meta.algo == algo].encoder)]
        for game in mf.GAME_ORDER:
            L.append(f"{algo} / {game}")
            rows = []
            for e in arms:
                C = curves(meta, algo, game, e)
                if C is None:
                    continue
                m = C.mean(axis=0) * 100
                half = m[len(m) // 2]
                rows.append((mf.ARM[e][0], m[0], half, m[-1], m[-1] - half))
            for nm, first, half, fin, d in rows:
                L.append(f"    {nm:<11s} eval1 {first:6.2f}   half {half:6.2f}"
                         f"   final {fin:6.2f}   (final-half {d:+6.2f})")
            if rows:
                by_half = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
                by_fin = [r[0] for r in sorted(rows, key=lambda r: -r[3])]
                L.append(f"    ordering at half : {' > '.join(by_half)}")
                L.append(f"    ordering at final: {' > '.join(by_fin)}")
                L.append(f"    same ordering: {by_half == by_fin}")
            L.append("")
    p = os.path.join(HERE, "figures", "convergence_numbers.txt")
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", p)


if __name__ == "__main__":
    main()
