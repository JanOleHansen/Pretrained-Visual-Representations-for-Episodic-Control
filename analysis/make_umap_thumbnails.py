#!/usr/bin/env python3
"""
UMAP of the matched frame set with the actual game frames drawn at their own
coordinates -- the figure that makes the "semantic clusters, value smeared
within them" claim checkable instead of asking the reader to take it on faith.

    python make_umap_thumbnails.py
    python make_umap_thumbnails.py --arms clip mae

Why this figure exists
----------------------
`make_embedding_figures.py` draws the same projections as a 6x3 grid of
anonymous point clouds.  That grid shows *that* the clusters exist; it cannot
show *what* they contain, so the central claim -- that an encoder groups frames
which look alike, and that frames which look alike are worth very different
returns -- is unverifiable from the picture.

Thumbnails fix that.  Each frame is drawn at its own UMAP position with a
border coloured by the return that followed it, so a cluster of visually
near-identical frames carrying a spread of border colours *is* the claim,
visible in one panel.  It is also the sharpest statement of the invariance
account: a contrastive encoder is trained to be invariant to exactly the
differences that decide the score.

The frames
----------
The probe set itself (~1.7 GB per game) stays on the cluster; this script
rebuilds it locally, which is sound only because the rollout is deterministic --
fixed seed, fixed action stream, `repeat_action_probability=0`, seeded
`NoopResetEnv`.  That is *verified rather than assumed*: `--verify` checks the
regenerated reward, done, episode and return arrays against the
`probe_labels_<game>.npz` that came back from the cluster with the embeddings.
If they disagree, the local frames are from a different rollout than the
embeddings were computed on, the thumbnails would be mislabelled, and the
script refuses to draw anything.

Output: figures/embedding_umap_thumbnails.pdf in both documents.
"""
import argparse, os, sys, warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared run loader (make_figures.py) sits next to this script in
# analysis/; "DECK" is kept as the name for where it lives.
DECK = HERE
sys.path.insert(0, DECK)
import make_figures as mf                      # noqa: E402
import make_embedding_figures as mef           # noqa: E402

ARM = mf.ARM
FIGDIRS = mef.FIGDIRS
EMB_DIR = mef.EMB_DIR
TEXTWIDTH_IN = mef.TEXTWIDTH_IN
REPO = os.path.join(HERE, "..")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load_frames(game, probe_dir):
    """Memory-map the exported RGB frames.

    Reads the ``.npy`` written by ``scripts/export_probe_frames.py`` rather than
    the ``.pt`` probe set directly: this venv has no torch on purpose, and
    mmap keeps a gigabyte of frames off the heap when only ~44 of them are
    ever drawn.
    """
    path = os.path.join(probe_dir, f"frames_{game}.npy")
    if not os.path.exists(path):
        raise SystemExit(
            f"no {path}.\nBuild it with the RL venv (this one has no torch):\n"
            f"  cd {REPO}\n"
            f"  .venv/bin/python scripts/build_probe_set.py --game {game} "
            f"--frames 10000 --pipelines mfec_rgb --out probe_sets\n"
            f"  .venv/bin/python scripts/export_probe_frames.py --game {game}")
    return np.load(path, mmap_mode="r")


def verify_against_labels(probe_dir, game, emb_dir):
    """Refuse to draw unless the local rollout IS the cluster's rollout.

    The embeddings were computed on the cluster's frames; these are local ones.
    If the two rollouts diverged -- a different ALE build, an unseeded draw --
    every thumbnail would sit at the coordinates of a *different* frame, and
    nothing about the resulting picture would look wrong.  Reward and episode
    structure are the fingerprint: they are independent of the observation
    pipeline, so agreement on them means the same game was played move for move.

    The local side is read from the sidecar the exporter writes next to the
    frames, so this check needs no torch either.
    """
    lab_path = os.path.join(emb_dir, f"probe_labels_{game}.npz")
    side_path = os.path.join(probe_dir, f"labels_{game}.npz")
    if not os.path.exists(lab_path):
        raise SystemExit(f"no {lab_path} to verify against")
    if not os.path.exists(side_path):
        raise SystemExit(
            f"no {side_path} -- re-run scripts/export_probe_frames.py, which "
            "writes it alongside the frames.")
    lab = np.load(lab_path, allow_pickle=True)
    loc = np.load(side_path, allow_pickle=True)

    checks = {name: (loc[name], lab[name])
              for name in ("reward", "done", "episode", "return_g1", "action")}
    bad = []
    for name, (local, cluster) in checks.items():
        if local.shape != cluster.shape:
            bad.append(f"{name}: shape {local.shape} vs {cluster.shape}")
        elif not np.allclose(local, cluster, rtol=0, atol=0):
            n = int((local != cluster).sum())
            bad.append(f"{name}: {n}/{len(local)} entries differ")
    if bad:
        raise SystemExit(
            "the local rollout is NOT the rollout the embeddings were computed "
            "on:\n  " + "\n  ".join(bad) +
            "\nThumbnails would be attached to the wrong frames. Refusing to "
            "draw.\nCopy probe_MsPacman.pt down from the cluster instead.")
    print(f"  verified: local rollout matches the cluster's exactly "
          f"({len(lab['reward'])} steps, {int(lab['done'].sum())} episodes)")
    return lab


def spread_indices(xy, n_thumbs, seed=0):
    """Pick well-separated points so thumbnails tile the layout, not pile up.

    k-means on the 2-D coordinates, then the real point closest to each
    centroid.  Picking at random instead would clump thumbnails wherever the
    density is highest, which is exactly where they occlude each other.
    """
    from sklearn.cluster import KMeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=n_thumbs, n_init=4, random_state=seed).fit(xy)
    out = []
    for c in km.cluster_centers_:
        d = ((xy - c) ** 2).sum(1)
        d[out] = np.inf              # no point used twice
        out.append(int(np.argmin(d)))
    return np.array(out)


#: Rows of the raw ALE frame to keep. The bottom of a Ms. Pac-Man frame is the
#: score/lives strip -- constant black plus digits, no game state -- and at
#: thumbnail size it eats a fifth of the height while carrying nothing the
#: reader is being asked to compare.
CROP_ROWS = {"MsPacman": (0, 172), "Qbert": (0, 195), "Frostbite": (0, 200)}


def panel(ax, emb2d, pct, frames, rows, n_thumbs, thumb_px, title, crop=None,
          zoom=0.11):
    """One arm: scatter coloured by return, with frames drawn on top."""
    scat = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=pct, cmap="viridis",
                      s=3.5, lw=0, alpha=0.55, rasterized=True)

    # Trim to the bulk before choosing thumbnails, so a couple of UMAP outliers
    # do not spend thumbnails on empty corners (same reasoning as the grid
    # figure's axis clip).
    lims = []
    for col in (0, 1):
        lo, hi = np.percentile(emb2d[:, col], [0.5, 99.5])
        pad = 0.08 * max(hi - lo, 1e-9)
        lims.append((lo - pad, hi + pad))
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])

    # Thumbnails are placed well inside the frame, not merely inside the axis
    # limits: an AnnotationBbox is anchored at its centre and is not clipped, so
    # a point at the very edge of the view puts half its image outside the panel
    # and over the neighbouring one.
    inner = []
    for col in (0, 1):
        lo, hi = np.percentile(emb2d[:, col], [5, 95])
        inner.append((lo, hi))
    inside = ((emb2d[:, 0] >= inner[0][0]) & (emb2d[:, 0] <= inner[0][1]) &
              (emb2d[:, 1] >= inner[1][0]) & (emb2d[:, 1] <= inner[1][1]))
    idx_in = np.flatnonzero(inside)
    pick = idx_in[spread_indices(emb2d[idx_in], min(n_thumbs, len(idx_in)))]

    cmap = plt.get_cmap("viridis")
    for i in pick:
        img = frames[rows[i]]                       # (3, H, W) uint8
        img = np.transpose(img, (1, 2, 0))
        if crop is not None:
            img = img[crop[0]:crop[1]]
        img = img[::max(1, img.shape[0] // thumb_px),
                  ::max(1, img.shape[1] // thumb_px)]
        box = OffsetImage(img, zoom=zoom, interpolation="nearest")
        # The border carries the frame's OWN return percentile -- that is what
        # makes "these look alike but are worth different amounts" legible.
        ab = AnnotationBbox(
            box, (emb2d[i, 0], emb2d[i, 1]), frameon=True, pad=0.10,
            bboxprops=dict(edgecolor=cmap(pct[i] / 100.0), linewidth=1.1))
        ax.add_artist(ab)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("0.75")
    ax.set_title(title, pad=4)
    return scat


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="MsPacman")
    p.add_argument("--arms", nargs="+", default=["clip", "random_projection"],
                   help="encoders to draw, left to right")
    p.add_argument("--seed", type=int, default=42, help="which run's encoder")
    p.add_argument("--emb-dir", default=EMB_DIR)
    p.add_argument("--probe-dir", default=os.path.join(REPO, "probe_sets"))
    p.add_argument("--rows", type=int, default=4000, help="points projected")
    p.add_argument("--thumbs", type=int, default=14)
    p.add_argument("--zoom", type=float, default=0.11,
                   help="thumbnail size; 0.11 is ~11%% of a panel's width")
    p.add_argument("--thumb-px", type=int, default=90)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.45,
                   help="UMAP min_dist. Higher spreads the layout out; the "
                        "default is deliberately looser than the grid figure's "
                        "0.1 because consecutive frames are near-duplicates and "
                        "a tight setting fuses them into a few unreadable dots")
    p.add_argument("--neighbours", type=int, default=7,
                   help="neighbours shown beside the query in the retrieval figure")
    p.add_argument("--skip-scatter", action="store_true",
                   help="only build the retrieval figure")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the rollout-identity check (do not use for the thesis)")
    args = p.parse_args()

    frames = load_frames(args.game, args.probe_dir)
    if not args.no_verify:
        verify_against_labels(args.probe_dir, args.game, args.emb_dir)
    lab = np.load(os.path.join(args.emb_dir, f"probe_labels_{args.game}.npz"),
                  allow_pickle=True)

    keep = lab["complete"].astype(bool)
    keep_rows = np.flatnonzero(keep)
    G = lab["return_g1"][keep]

    retrieval_figure(args, frames, lab, keep_rows,
                     "embedding_retrieval_neighbours")
    if args.skip_scatter:
        return 0

    fig, axes = plt.subplots(1, len(args.arms), squeeze=False,
                             figsize=(TEXTWIDTH_IN,
                                      TEXTWIDTH_IN / len(args.arms) + 0.75))
    scat = None
    for j, arm in enumerate(args.arms):
        name = f"mfec_{args.game}_{arm}_seed{args.seed}.npz"
        path = os.path.join(args.emb_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"no {path}")
        X = np.load(path, allow_pickle=True)["probe_emb"][keep]

        rng = np.random.default_rng(0)
        sel = (np.arange(len(X)) if len(X) <= args.rows
               else np.sort(rng.choice(len(X), args.rows, replace=False)))
        emb2d, _ = mef.umap_2d(X[sel], seed=0, n_neighbors=args.umap_neighbors,
                               min_dist=args.min_dist, max_rows=len(sel))
        v = G[sel]
        pct = (np.argsort(np.argsort(v)) / max(1, len(v) - 1)) * 100.0

        scat = panel(axes[0][j], emb2d, pct, frames, keep_rows[sel],
                     args.thumbs, args.thumb_px, ARM[arm][0],
                     crop=CROP_ROWS.get(args.game), zoom=args.zoom)
        print(f"  drew {ARM[arm][0]}: {len(sel)} points, {args.thumbs} frames")

    cbar = fig.colorbar(scat, ax=axes, orientation="horizontal",
                        fraction=0.035, pad=0.03, aspect=45)
    cbar.set_label("return-to-go percentile (frame border uses the same scale)",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)

    for d in FIGDIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "embedding_umap_thumbnails.pdf"))
    print("  wrote embedding_umap_thumbnails.pdf")
    return 0



# ---------------------------------------------------------------------------
# The retrieval figure -- what MFEC actually averages over
# ---------------------------------------------------------------------------
#
# The scatter above shows *where* frames land.  This shows what that costs.
# MFEC answers a query by averaging the stored returns of its k nearest keys
# (Eq. 2), so the honest question is not "are the clusters clean" but "do the
# frames a lookup actually returns carry the same value".  One row per encoder:
# a query frame and the neighbours it retrieves, each labelled with the return
# that really followed it.  A row of near-identical pictures with wildly
# different numbers under them is the invariance account, stated as plainly as
# it can be stated.

def retrieval_figure(args, frames, lab, keep_rows, out_name):
    from sklearn.neighbors import NearestNeighbors

    keep = lab["complete"].astype(bool)
    G = lab["return_g1"][keep]
    crop = CROP_ROWS.get(args.game)
    n_show = args.neighbours

    fig, axes = plt.subplots(len(args.arms), n_show + 1, squeeze=False,
                             figsize=(TEXTWIDTH_IN,
                                      1.15 * TEXTWIDTH_IN / (n_show + 1)
                                      * len(args.arms) + 0.5))

    for r, arm in enumerate(args.arms):
        X = np.load(os.path.join(args.emb_dir,
                                 f"mfec_{args.game}_{arm}_seed{args.seed}.npz"),
                    allow_pickle=True)["probe_emb"][keep]
        nn = NearestNeighbors(n_neighbors=n_show + 1).fit(X)
        dist, idx = nn.kneighbors(X)

        # Pick the query whose own neighbourhood disagrees most about value --
        # the worst case, chosen by a stated rule rather than by eye, and
        # reported with its numbers so the choice can be checked.
        spread = G[idx].max(1) - G[idx].min(1)
        q = int(np.argmax(spread))
        rows = idx[q]

        for c in range(n_show + 1):
            ax = axes[r][c]
            img = np.transpose(frames[keep_rows[rows[c]]], (1, 2, 0))
            if crop is not None:
                img = img[crop[0]:crop[1]]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color("0.2" if c == 0 else "0.75")
            ax.set_xlabel(f"{G[rows[c]]:.0f}", fontsize=7.5, labelpad=1.5)
            if c == 0:
                ax.set_ylabel(ARM[arm][0], fontsize=8.5, labelpad=3)
            # Column headings belong to the figure, not to each row: repeating
            # them under every arm implies the columns mean something different
            # per row, and they do not.
            if r == 0:
                ax.set_title("query" if c == 0 else f"nn {c}", fontsize=7.5,
                             pad=2, color="0.2" if c == 0 else "0.45")

        print(f"  {ARM[arm][0]:10s} widest neighbourhood: returns "
              f"{G[rows].min():.0f}-{G[rows].max():.0f} "
              f"(spread {spread[q]:.0f}), embedding distance to nn{n_show}: "
              f"{dist[q][-1]:.4f}")

    fig.suptitle("Frames retrieved together, and the return that actually "
                 "followed each", fontsize=9, y=1.005)
    for d in FIGDIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{out_name}.pdf"))
    plt.close(fig)
    print(f"  wrote {out_name}.pdf")

if __name__ == "__main__":
    raise SystemExit(main())
