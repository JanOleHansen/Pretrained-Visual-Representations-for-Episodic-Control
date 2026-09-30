#!/usr/bin/env python3
"""
Build the embedding-geometry figures and numbers: UMAP projections of the keys
and a decoding probe on the raw embeddings, per encoder and per game.

    python make_embedding_figures.py                        # everything
    python make_embedding_figures.py --skip-umap            # metrics only
    python make_embedding_figures.py --algos MFEC

Run it in the embedding environment (analysis/requirements-embedding.txt) --
umap-learn and scikit-learn are not in the main venv and are deliberately not added to it (see
make_rliable_figures.py for the same arrangement and why the environments are
kept apart).

What this answers
-----------------
RQ3's second half: *why* is an encoder good or bad?  Hit rate said what it is
not -- it is set by the game.  The positive account has to be a property of the
embeddings, so this asks what a decoder can read out of them, following
Schneider et al. (2023): judge an embedding by what can be recovered from it,
not by how the projection looks.  The projection is here too, but it is the
illustration, and the numbers are the claim.

Two sources, deliberately kept separate and reported side by side
-----------------------------------------------------------------
``memory``  the keys the run's episodic memory actually holds, with the value
            stored against each.  This is the object the algorithm retrieves
            from -- but each arm's memory holds the states *its own* policy
            visited, so a between-arm read confounds the encoder with the
            behaviour.
``probe``   every arm's encoder applied to one shared frame set
            (scripts/build_probe_set.py).  Same states for every arm, so phi is
            the only thing that varies.  This is the controlled comparison and
            the one the ranking claims are made on; its limitation is that the
            frames come from a uniform-random policy, not an on-policy
            distribution.

Neither is sufficient alone, which is why both are plotted.  Where they agree
the reading is safe; the Spearman correlation between the two rankings is
written into embedding_numbers.txt so that agreement is a reported number
rather than an impression.

Three things the probe controls for
-----------------------------------
1.  **Key width.**  CLIP has 512 dimensions against RP's 64, and a linear probe
    with more dimensions fits better for free.  Every probe is therefore run
    twice: at the encoder's native width, and again after PCA to 64 -- the width
    every NEC arm and both random projections already use.  The PCA is fitted
    inside the cross-validation fold, never on the full set.
2.  **Temporal leakage.**  Consecutive frames are near-identical and carry
    near-identical returns, so a random split over rows would put a frame's
    neighbour in the test set and inflate every R^2 in the study.  Splits are
    grouped by episode.
3.  **A number that means nothing.**  Every R^2 is reported against a
    shuffled-label null run through the identical pipeline.

Beyond the linear probe, ``R2 (kNN)`` fits nothing at all: it predicts a point's
value from its k nearest neighbours in the raw embedding space, at the k the
algorithm itself uses (MFEC k=11, NEC k=50).  That is the closest thing to
measuring the quantity episodic control actually depends on, because it is the
retrieval rule, not a readout that the agent never performs.

Output: figures/*.pdf and figures/*.tex in BOTH documents, plus
embedding_numbers.txt and embedding_metrics.csv, so captions and prose cannot
drift from the plots.
"""
import argparse, os, sys, time, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared run loader (make_figures.py) sits next to this script in
# analysis/; DECK is that directory.
DECK = HERE
sys.path.insert(0, DECK)
import make_figures as mf          # noqa: E402  (path must be set first)

ARM, COL, ARM_ORDER = mf.ARM, mf.COL, mf.ARM_ORDER
GAME_ORDER = mf.GAME_ORDER

FIGDIRS = [os.path.join(HERE, "figures"),
           os.path.join(HERE, "figures", "deck")]
REPO = os.path.join(HERE, "..")
#: The ~4 GB of per-run npz from scripts/extract_embeddings.py (gitignored).
EMB_DIR = os.environ.get("PVR_EMB", os.path.join(REPO, "embeddings"))
#: The small probe_labels_<game>.npz that ship with the repo, so the
#: figures-only path below needs nothing that is not committed.
LABEL_DIR = os.path.join(HERE, "data", "probe_labels")

#: Panel titles use the names the running text uses, as GAME_LABEL does in
#: make_thesis_figures.py, not the raw ALE ids.
_FIG_GAME = {"MsPacman": "Ms. Pac-Man", "Qbert": "Q*bert"}

TEXTWIDTH_IN = 430.00462 / 72.27   # \textwidth from main.log, pt -> inch

#: k for the retrieval probe -- each algorithm's OWN k, so the number describes
#: the lookup that algorithm performs.  MFEC: configs/algorithm/mfec_atari.yaml
#: ("The k-nearest-neighbour lookups used k = 11", Blundell et al. §4.1).
#: NEC: configs/algorithm/nec_atari.yaml (p = 50, Pritzel et al. §4).
ALGO_K = {"MFEC": 11, "NEC": 50}

#: Which return the probe targets, matching what each algorithm's memory stores.
ALGO_TARGET = {"MFEC": "return_g1", "NEC": "return_g099"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def study_run_names(cache=None):
    """The set of runs that count, read from the shared W&B cache.

    The cluster holds more than the study, and some of the extras are
    *indistinguishable by name pattern alone*: superseded MFEC runs whose game
    token is lowercase (`mfec_mspacman_dinov2_seed42`) carry an encoder that is
    in the arm vocabulary and a game that canonicalises to a real one, so they
    would silently land in the DINOv2/Ms. Pac-Man cell and give it ten seeds
    drawn from two different observation pipelines.

    `runs_meta.csv` is already this project's single definition of which runs
    count -- it is what every other figure is built from -- so it is the
    authority here too rather than a second, forkable rule.
    """
    import csv
    path = os.path.join(cache or mf.DEFAULT_CACHE, "runs_meta.csv")
    with open(path) as fh:
        return {r["name"] for r in csv.DictReader(fh)}


def load_labels(label_dir):
    """Read only the probe_labels_<game>.npz files from ``label_dir``."""
    import glob
    labels = {}
    for path in sorted(glob.glob(os.path.join(label_dir, "probe_labels_*.npz"))):
        game = os.path.basename(path)[len("probe_labels_"):-len(".npz")]
        labels[game] = dict(np.load(path, allow_pickle=True))
    return labels


def load_runs(emb_dir, algos, games, cache_filter=True):
    """Read every per-run npz written by scripts/extract_embeddings.py."""
    import glob
    runs, labels, unknown = [], {}, set()
    keep_names = study_run_names() if cache_filter else None
    off_grid = []
    for path in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))):
        base = os.path.basename(path)
        if base.startswith("probe_labels_"):
            game = base[len("probe_labels_"):-len(".npz")]
            labels[game] = dict(np.load(path, allow_pickle=True))
            continue
        name = base[:-len(".npz")]
        if keep_names is not None and name not in keep_names:
            off_grid.append(name)
            continue
        d = np.load(path, allow_pickle=True)
        algo, game = str(d["algo"]), str(d["game"])
        if algo not in algos or game not in games:
            continue
        # ARM is the shared arm vocabulary imported from the deck script, and
        # every figure indexes it for a label and a colour.  An npz from an arm
        # outside the ablation (a *_frozen_* control, say) would KeyError deep
        # inside plotting, so it is dropped here where the reason is visible.
        if str(d["encoder"]) not in ARM:
            unknown.add(str(d["encoder"]))
            continue
        runs.append({
            "algo": algo, "game": game,
            "encoder": str(d["encoder"]), "seed": int(d["seed"]),
            "name": base[:-len(".npz")], "npz": d,
        })
    if off_grid:
        print(f"skipped {len(off_grid)} run(s) absent from the W&B cache "
              f"(superseded or off-grid): {', '.join(sorted(off_grid))}")
    if unknown:
        print(f"skipped {len(unknown)} encoder(s) not in the arm vocabulary: "
              f"{', '.join(sorted(unknown))}")
    if not runs:
        raise SystemExit(
            f"no run npz files in {emb_dir}.\n"
            "Run scripts/extract_embeddings.py on the cluster and rsync the "
            "output here, or point PVR_EMB at it.")
    return runs, labels


def source_arrays(run, source, labels):
    """Return ``(X, y, groups)`` for one run and one data source.

    ``groups`` is an episode id on the probe set, so cross-validation can hold
    whole episodes out; it is None for the memory keys, which carry no episode
    identity (the memory stores deduplicated states, not a trajectory).
    """
    d = run["npz"]
    if source == "memory":
        return d["mem_keys"], d["mem_values"], None

    if "probe_emb" not in d.files:
        return None, None, None
    lab = labels.get(run["game"])
    if lab is None:
        return None, None, None

    X = d["probe_emb"]
    keep = lab["complete"].astype(bool)      # drop the truncated tail episode
    y = lab[ALGO_TARGET[run["algo"]]]
    if len(X) != len(keep):
        raise SystemExit(
            f"{run['name']}: {len(X)} probe rows against {len(keep)} labels — "
            "the npz and probe_labels come from different probe sets.")
    return X[keep], y[keep], lab["episode"][keep]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _splitter(groups, n_splits, seed):
    """Episode-grouped CV where episodes exist, shuffled KFold where they do not.

    The fallback is a real weakness, not a formality: without grouping, frame t
    and frame t+1 land on opposite sides of the split, and since they are nearly
    the same picture with nearly the same return the probe scores its own
    training data.  That inflates R^2 towards 1 and does it *quietly*.  Callers
    record which branch ran (``grouped`` in the metrics table) so an inflated
    number can never be quoted as a grouped one.
    """
    from sklearn.model_selection import GroupKFold, KFold
    if groups is None or len(np.unique(groups)) < n_splits:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None
    return GroupKFold(n_splits=n_splits), groups


def was_grouped(groups, n_splits=5):
    return groups is not None and len(np.unique(groups)) >= n_splits


def ridge_r2(X, y, groups, n_splits=5, pca_dim=None, shuffle_y=False, seed=0):
    """Held-out R^2 of a ridge decoder from embedding to return.

    Standardisation and the optional PCA are fitted **inside** each training
    fold, so no information about the held-out episodes reaches the model
    through either.  The ridge penalty is chosen by an inner CV on the training
    fold alone, which is what keeps the wide encoders from being scored on an
    over-fit they were handed for free.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    if shuffle_y:
        y = np.random.default_rng(seed).permutation(y)

    def build():
        steps = [StandardScaler()]
        if pca_dim is not None and X.shape[1] > pca_dim:
            steps.append(PCA(n_components=pca_dim, random_state=seed))
        steps.append(RidgeCV(alphas=np.logspace(-2, 6, 17)))
        return make_pipeline(*steps)

    cv, g = _splitter(groups, n_splits, seed)
    scores = []
    for tr, te in cv.split(X, y, g):
        if np.std(y[te]) == 0:
            continue
        model = build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
            scores.append(model.score(X[te], y[te]))
    return float(np.mean(scores)) if scores else np.nan


def knn_r2(X, y, groups, k, n_splits=5, seed=0):
    """Held-out R^2 of a k-NN regressor in the RAW embedding space.

    Deliberately unstandardised and unprojected: this is meant to score the
    geometry the algorithm queries, and neither MFEC's QEC nor NEC's DND
    rescales a key before measuring distance to it.
    """
    from sklearn.neighbors import KNeighborsRegressor

    y = np.asarray(y, dtype=np.float64)
    cv, g = _splitter(groups, n_splits, seed)
    scores = []
    for tr, te in cv.split(X, y, g):
        if np.std(y[te]) == 0 or len(tr) <= k:
            continue
        m = KNeighborsRegressor(n_neighbors=k).fit(X[tr], y[tr])
        scores.append(m.score(X[te], y[te]))
    return float(np.mean(scores)) if scores else np.nan


def value_coherence(X, y, k, max_rows=6000, seed=0):
    """Mean |Δvalue| among k nearest neighbours, over the same for random pairs.

    Fits nothing, so it cannot be inflated by a decoder's capacity.  Below 1
    means neighbours in embedding space are worth more similar returns than two
    states drawn at random -- i.e. retrieval returns states of comparable value,
    which is the entire premise of episodic control.  At 1 the geometry carries
    no value information at all.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X[idx], np.asarray(y)[idx]
    y = np.asarray(y, dtype=np.float64)
    if len(X) <= k + 1 or np.std(y) == 0:
        return np.nan

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    local = np.abs(y[idx[:, 1:]] - y[:, None]).mean()

    a = rng.integers(0, len(y), 200_000)
    b = rng.integers(0, len(y), 200_000)
    globally = np.abs(y[a] - y[b]).mean()
    return float(local / globally) if globally > 0 else np.nan


def clf_balanced_accuracy(X, yb, groups, n_splits=5, seed=0):
    """Balanced accuracy of a linear probe for "a reward follows within H steps".

    The bioacoustics move: score a frozen encoder by what a *linear* readout can
    classify from it.  Balanced accuracy rather than accuracy because the
    positive class is not 50/50, and unlike R^2 on return it is not dominated by
    the handful of very-high-return frames.  0.5 is chance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    yb = np.asarray(yb).astype(int)
    cv, g = _splitter(groups, n_splits, seed)
    scores = []
    for tr, te in cv.split(X, yb, g):
        if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], yb[tr])
            scores.append(balanced_accuracy_score(yb[te], m.predict(X[te])))
    return float(np.mean(scores)) if scores else np.nan


def compute_metrics(runs, labels, sources, args):
    """One row per (run, source); cached to embedding_metrics.csv.

    On the ``probe`` source the five seeds of a frozen MFEC arm share one
    encoder -- the weights are fixed, only the memory differs -- so all five
    rows are the *same computation* on the *same* matrix.  They are computed
    once and copied, which cuts the MFEC probe pass by roughly 4x and, more
    importantly, keeps the fact visible: ``phi_distinct`` records how many
    genuinely different encoders a cell's mean is over, so a zero error bar is
    read as "no seed variation by construction" rather than as a suspiciously
    tight measurement.  The random projections and every NEC arm keep five
    distinct encoders and are not deduplicated.
    """
    rows = []
    total = len(runs) * len(sources)
    n = 0
    phi_cache: dict[str, dict] = {}
    for run in runs:
        for source in sources:
            n += 1
            X, y, groups = source_arrays(run, source, labels)
            t0 = time.time()
            if X is None:
                print(f"[{n:3d}/{total}] {run['name']:46s} {source:6s} "
                      f"-- no data, skipped")
                continue
            sel = None
            if args.max_rows and len(X) > args.max_rows:
                sel = np.random.default_rng(0).choice(
                    len(X), args.max_rows, replace=False)
                X, y = X[sel], np.asarray(y)[sel]
                groups = None if groups is None else np.asarray(groups)[sel]

            k = ALGO_K[run["algo"]]
            sig = (str(run["npz"]["probe_signature"])
                   if source == "probe" and "probe_signature" in run["npz"].files
                   else None)
            cached = phi_cache.get(sig) if sig else None

            if cached is not None:
                metrics = dict(cached)
                note = "(shared phi)"
            else:
                metrics = {
                    "r2_linear":       ridge_r2(X, y, groups),
                    "r2_linear_pca64": ridge_r2(X, y, groups, pca_dim=64),
                    "r2_linear_null":  ridge_r2(X, y, groups, shuffle_y=True),
                    "r2_knn":          knn_r2(X, y, groups, k),
                    "coherence":       value_coherence(X, y, k),
                }
                if source == "probe":
                    lab = labels[run["game"]]
                    keep = lab["complete"].astype(bool)
                    yb = lab["reward_soon"][keep]
                    if sel is not None:
                        yb = yb[sel]
                    metrics["bal_acc"] = clf_balanced_accuracy(X, yb, groups)
                if sig:
                    phi_cache[sig] = metrics
                note = ""

            row = {
                "algo": run["algo"], "game": run["game"],
                "encoder": run["encoder"], "seed": run["seed"],
                "source": source, "d": X.shape[1], "n": len(X),
                "grouped": was_grouped(groups),
                "n_episodes": (0 if groups is None
                               else int(len(np.unique(groups)))),
                "phi_shared": cached is not None,
                **metrics,
            }
            rows.append(row)
            print(f"[{n:3d}/{total}] {run['name']:46s} {source:6s} "
                  f"d={row['d']:4d} n={row['n']:6d} "
                  f"R2lin={row['r2_linear']:+.3f} R2knn={row['r2_knn']:+.3f} "
                  f"coh={row['coherence']:.3f}  ({time.time()-t0:.0f}s) {note}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def umap_2d(X, seed=0, n_neighbors=30, min_dist=0.1, max_rows=6000):
    """Project to 2-D with the metric the algorithm itself retrieves under.

    Euclidean, not cosine and not a standardised space: MFEC's QEC and NEC's DND
    both take raw L2 distances between keys, so a projection under any other
    metric would draw a neighbourhood structure the agent never sees.  (For the
    L2-normalised arms -- CLIP under ``clip_normalize`` and every NEC key --
    Euclidean and cosine order neighbours identically, so this costs nothing
    there and is the honest choice everywhere else.)
    """
    import umap

    rng = np.random.default_rng(seed)
    idx = (np.arange(len(X)) if len(X) <= max_rows
           else np.sort(rng.choice(len(X), max_rows, replace=False)))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="euclidean",
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(X[idx]), idx


def figure_umap(runs, labels, source, algo, games, args):
    """Grid of UMAP panels: one row per encoder, one column per game."""
    encs = [e for e in ARM_ORDER
            if any(r["encoder"] == e and r["algo"] == algo for r in runs)]
    if not encs:
        return None
    games = [g for g in games
             if any(r["game"] == g and r["algo"] == algo for r in runs)]

    fig, axes = plt.subplots(
        len(encs), len(games), squeeze=False,
        figsize=(TEXTWIDTH_IN, 1.02 * TEXTWIDTH_IN / len(games) * len(encs)))

    scat = None
    for i, enc in enumerate(encs):
        for j, game in enumerate(games):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(0.4)
                ax.spines[side].set_color("0.75")

            match = [r for r in runs if r["algo"] == algo
                     and r["encoder"] == enc and r["game"] == game
                     and r["seed"] == args.umap_seed]
            if not match:
                ax.text(0.5, 0.5, "--", ha="center", va="center",
                        color="0.6", transform=ax.transAxes)
                continue

            X, y, _ = source_arrays(match[0], source, labels)
            if X is None or len(X) < 50:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        color="0.6", transform=ax.transAxes)
                continue

            emb, idx = umap_2d(X, seed=0, n_neighbors=args.umap_neighbors,
                               max_rows=args.umap_rows)
            v = np.asarray(y, dtype=np.float64)[idx]
            # Colour by within-panel percentile.  Returns are heavy-tailed --
            # a handful of frames carry most of the score -- so a linear colour
            # scale paints 95% of every panel the same shade and shows nothing.
            pct = (np.argsort(np.argsort(v)) / max(1, len(v) - 1)) * 100.0
            # Marks sized so the panel reads at the same ink density whatever
            # the memory happens to hold: a Q*bert MFEC table is a fraction of
            # a Frostbite one, and a fixed size draws the small ones as dust.
            size = float(np.clip(3000.0 / len(v), 0.9, 9.0))
            scat = ax.scatter(emb[:, 0], emb[:, 1], c=pct, cmap="viridis",
                              s=size, lw=0, alpha=0.75, rasterized=True)

            # Clip the view to the bulk.  UMAP routinely strands a handful of
            # points far from everything else; on a shared auto-scaled axis
            # those few decide the range and squeeze the other 99% into a blob
            # a few pixels across, which is a picture of the outliers rather
            # than of the embedding.  The points are still drawn -- only the
            # window is trimmed -- and the same rule is applied to every panel
            # so panels remain comparable.
            for axis, col in ((ax.set_xlim, 0), (ax.set_ylim, 1)):
                lo, hi = np.percentile(emb[:, col], [0.5, 99.5])
                pad = 0.06 * max(hi - lo, 1e-9)
                axis(lo - pad, hi + pad)

            if i == 0:
                ax.set_title(_FIG_GAME.get(game, game), pad=4)
            if j == 0:
                # Plain ink, not ARM[enc][1]: the row label is the panel's only
                # identifier, and RP (RGB)'s #9A9A9A is unreadable as text on
                # white.  Nothing else in the panel competes for the identity,
                # so the colour buys nothing and costs legibility.
                ax.set_ylabel(ARM[enc][0], rotation=90, labelpad=6,
                              fontsize=8.5)

    if scat is not None:
        cbar = fig.colorbar(scat, ax=axes, orientation="horizontal",
                            fraction=0.025, pad=0.03, aspect=50)
        cbar.set_label("return-to-go percentile within panel", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_linewidth(0.4)

    suffix = "" if algo == "MFEC" else "_nec"
    return fig, f"embedding_umap_{source}{suffix}"


# ---------------------------------------------------------------------------
# Metric figures
# ---------------------------------------------------------------------------

def figure_probe(df, algo, source, games):
    """Two rows -- linear probe R^2 and k-NN R^2 -- one column per game.

    The two bars per encoder are the same hue with different texture: hue is
    identity (which encoder), texture is the series (native width vs PCA-64), so
    neither is carried by colour alone.
    """
    d = df[(df.algo == algo) & (df.source == source)]
    if d.empty:
        return None
    games = [g for g in games if g in set(d.game)]
    encs = [e for e in ARM_ORDER if e in set(d.encoder)]

    # sharey per ROW, not globally: the three games must be read off one scale
    # or the panels cannot be compared, but the linear and k-NN probes are
    # different quantities and pinning them to a common axis would waste half
    # the height of whichever is smaller.
    fig, axes = plt.subplots(2, len(games), squeeze=False, sharex=True,
                             sharey="row", figsize=(TEXTWIDTH_IN, 4.3))

    panels = [("r2_linear", "r2_linear_pca64", "linear probe  $R^2$"),
              ("r2_knn", None, "$k$-NN probe  $R^2$")]

    for row, (metric, metric2, ylabel) in enumerate(panels):
        for j, game in enumerate(games):
            ax = axes[row][j]
            sub = d[d.game == game]
            x = np.arange(len(encs))
            w = 0.38 if metric2 else 0.62

            for off, key, hatch, alpha in (
                    (-w / 2 if metric2 else 0.0, metric, None, 1.0),
                    (w / 2, metric2, "////", 0.45)):
                if key is None:
                    continue
                mean = [sub[sub.encoder == e][key].mean() for e in encs]
                sd = [sub[sub.encoder == e][key].std(ddof=1) for e in encs]
                # The PCA-64 series is separated from its native-width twin by
                # BOTH texture and fill opacity.  Hatch alone is invisible on
                # RP (gray)'s near-black #111111, which is exactly the arm the
                # width control exists to compare against.
                ax.bar(x + off, mean, w, yerr=sd, capsize=1.6,
                       color=[ARM[e][1] for e in encs], alpha=alpha,
                       edgecolor="k", lw=0.4, hatch=hatch,
                       error_kw=dict(lw=0.6, ecolor="0.3"))

            null = sub["r2_linear_null"].mean()
            ax.axhline(null, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.axhline(0, color="0.8", lw=0.5, zorder=0)
            ax.set_xlim(-0.65, len(encs) - 0.35)
            ax.set_xticks(x)
            ax.set_xticklabels([ARM[e][0] for e in encs], rotation=30,
                               ha="right")
            if row == 0:
                ax.set_title(_FIG_GAME.get(game, game), pad=4)
            if j == 0:
                ax.set_ylabel(ylabel)

    # Figure-level legend under the panels rather than inside one of them: the
    # bars run to the top of the axes whenever an encoder decodes well, so an
    # in-panel legend lands on the tallest bar in the figure.
    handles = [Patch(facecolor="0.55", edgecolor="k", lw=0.4,
                     label="native width"),
               Patch(facecolor="0.55", edgecolor="k", lw=0.4, hatch="////",
                     alpha=0.45, label="PCA to 64 dims"),
               Line2D([], [], color="0.35", lw=0.7, ls=(0, (3, 2)),
                      label="shuffled-label null")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    suffix = "" if algo == "MFEC" else "_nec"
    return fig, f"embedding_probe_{source}{suffix}"


def figure_coherence(df, algo, source, games):
    """Neighbourhood value coherence -- one panel, games grouped by encoder.

    Filtered by algorithm, not just by source: ``resnet``, ``dinov2``, ``clip``
    and ``mae`` each name an arm in BOTH grids, and they are not the same object
    -- MFEC's are frozen at their native width, NEC's are fine-tuned to d=64.
    Pooling them would average two different encoders into one bar.
    """
    d = df[(df.source == source) & (df.algo == algo)]
    if d.empty:
        return None
    encs = [e for e in ARM_ORDER if e in set(d.encoder)]
    games = [g for g in games if g in set(d.game)]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.5))
    x = np.arange(len(encs))
    w = 0.8 / len(games)
    for j, game in enumerate(games):
        sub = d[d.game == game]
        mean = [sub[sub.encoder == e]["coherence"].mean() for e in encs]
        sd = [sub[sub.encoder == e]["coherence"].std(ddof=1) for e in encs]
        ax.bar(x + (j - (len(games) - 1) / 2) * w, mean, w * 0.92, yerr=sd,
               capsize=1.6, color=mf.GAME_COLOR[game], edgecolor="k", lw=0.4,
               label=game, error_kw=dict(lw=0.6, ecolor="0.3"))

    ax.axhline(1.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
    ax.annotate("no value structure", xy=(len(encs) - 0.5, 1.0),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM[e][0] for e in encs], rotation=30, ha="right")
    ax.set_ylabel(r"$|\Delta G|$ among $k$-NN  /  random pairs")
    # Below the axes, like figure_probe: the bars run from zero, so any in-panel
    # legend placement covers data for some arm.
    ax.legend(frameon=False, ncol=len(games), loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    suffix = "" if algo == "MFEC" else "_nec"
    return fig, f"embedding_coherence_{source}{suffix}"


# ---------------------------------------------------------------------------
# Table and numbers
# ---------------------------------------------------------------------------

# Display names matching the running text, as GAME_LABEL in make_thesis_figures.py.
_TABLE_GAME = {"MsPacman": r"Ms.~Pac-Man", "Qbert": "Q*bert"}


def latex_table(df, source, games, name):
    """Rows = encoder; column groups = game; R^2 linear / kNN, mean over seeds.

    Written into both documents' figures/ for the same reason the PDFs are:
    a table the prose quotes must not be able to drift from the plot beside it.
    """
    games = [g for g in games if g in set(df.game)]
    # Three columns per game: the two probes and the coherence statistic, which
    # reads better as a column beside the probes it is compared to.
    lines = [r"% generated by make_embedding_figures.py -- do not edit",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "rrr" * len(games) + "}",
             r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{_TABLE_GAME.get(g, g)}}}" for g in games)
             + r" \\",
             "".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}"
                     for i in range(len(games))),
             "Encoder & " + " & ".join(
                 [r"$R^2_{\text{lin}}$ & $R^2_{k}$ & $c$"] * len(games))
             + r" \\",
             r"\midrule"]

    # Best and second best per column, marked bold / underlined.  Marked WITHIN
    # an algorithm block, never across: an MFEC probe value and a NEC one are
    # not measured against the same target (realized return vs. bootstrapped
    # N-step estimate), so a mark spanning both would compare two quantities.
    # Direction differs by column: R^2 is better high, the coherence c is
    # better LOW (c = 1 means neighbors are worth no more similar returns than
    # strangers), so c is ranked ascending.
    def _ranked(d, encs, g, col, ascending):
        vals = []
        for e in encs:
            s = d[(d.encoder == e) & (d.game == g)]
            if not s.empty:
                vals.append((float(s[col].mean()), e))
        vals.sort(key=lambda t: t[0], reverse=not ascending)
        return [e for _, e in vals[:2]]

    def _fmt(v, e, top2):
        t = f"{v:.3f}"
        if not top2:
            return rf"${t}$"
        if e == top2[0]:
            return rf"$\mathbf{{{t}}}$"
        if len(top2) > 1 and e == top2[1]:
            return rf"$\underline{{{t}}}$"
        return rf"${t}$"

    for algo in ["MFEC", "NEC"]:
        d = df[(df.algo == algo) & (df.source == source)]
        if d.empty:
            continue
        encs = [x for x in ARM_ORDER if x in set(d.encoder)]
        top = {(g, col): _ranked(d, encs, g, col, asc)
               for g in games
               for col, asc in (("r2_linear", False), ("r2_knn", False),
                                ("coherence", True))}
        lines.append(rf"\multicolumn{{{1+3*len(games)}}}{{l}}{{\textit{{{algo}}}}} \\")
        for e in encs:
            cells = []
            for g in games:
                s = d[(d.encoder == e) & (d.game == g)]
                if s.empty:
                    cells += ["--", "--", "--"]
                else:
                    cells += [
                        _fmt(s.r2_linear.mean(), e, top[(g, "r2_linear")]),
                        _fmt(s.r2_knn.mean(), e, top[(g, "r2_knn")]),
                        _fmt(s.coherence.mean(), e, top[(g, "coherence")]),
                    ]
            lines.append(f"{ARM[e][0]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for d in FIGDIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"  wrote {name}.tex")


def target_difficulty(labels):
    """How much of the probe target is unpredictable once episodes are held out.

    The probe's folds are grouped by episode, so any part of the return that is
    an episode-level offset -- the luck of what a random policy happened to
    collect in that run -- cannot be recovered from a frame, by any encoder.
    Where that share is large, a held-out R^2 near zero (or below it) is the
    *correct* answer rather than an encoder failing, and the ranking between
    arms is correspondingly compressed.

    Reported so the Frostbite column can be read for what it is.  It is a
    property of the game and the behaviour policy, not of any embedding, so it
    is computed once per game from the labels alone.
    """
    out = {}
    for game, lab in labels.items():
        keep = lab["complete"].astype(bool)
        G, ep = lab["return_g1"][keep], lab["episode"][keep]
        uniq = np.unique(ep)
        means = np.array([G[ep == e].mean() for e in uniq])
        ns = np.array([(ep == e).sum() for e in uniq])
        between = np.average((means - G.mean()) ** 2, weights=ns)
        out[game] = {
            "n": int(keep.sum()), "episodes": len(uniq),
            "mean": float(G.mean()), "sd": float(G.std()),
            "between_share": float(between / G.var()) if G.var() > 0 else np.nan,
            "reward_soon_rate": float(lab["reward_soon"][keep].mean()),
        }
    return out


def write_numbers(df, path, args, labels=None):
    """Every number the prose is allowed to quote, with its seed spread."""
    out = ["Embedding-geometry analysis -- generated by make_embedding_figures.py",
           f"  probe target: MFEC {ALGO_TARGET['MFEC']}, NEC {ALGO_TARGET['NEC']}",
           f"  k: MFEC {ALGO_K['MFEC']}, NEC {ALGO_K['NEC']}",
           f"  rows capped at {args.max_rows} per run", ""]

    # An ungrouped split leaks a frame's neighbour into the test fold and
    # inflates R^2 towards 1.  If any row ran that way it goes at the TOP of the
    # file, not in a footnote -- the numbers below it are not comparable to the
    # grouped ones and must not be quoted as though they were.
    if "grouped" in df and not df["grouped"].all():
        bad = df[~df["grouped"].astype(bool)]
        out += ["!! WARNING -- ungrouped cross-validation in "
                f"{len(bad)} of {len(df)} rows.",
                "!! Their R^2 is inflated by temporal leakage between adjacent "
                "frames.",
                "!! Affected: "
                + ", ".join(sorted({f"{r.algo}/{r.game}/{r.encoder}/{r.source}"
                                    for r in bad.itertuples()})),
                "!! The memory source has no episode ids by construction; a "
                "probe row here",
                "!! means the probe set held fewer episodes than CV folds -- "
                "rebuild it longer.",
                ""]

    if labels:
        out.append("=== probe target difficulty (per game, encoder-independent) "
                   + "=" * 8)
        out.append("    'between' is the share of return variance that is an "
                   "episode-level offset;")
        out.append("    folds are grouped by episode, so that share is "
                   "unpredictable from a frame by")
        out.append("    ANY encoder, and a near-zero held-out R^2 there is the "
                   "correct answer.")
        for game, s in target_difficulty(labels).items():
            out.append(f"    {game:<10s} n={s['n']:5d} episodes={s['episodes']:3d} "
                       f"G mean={s['mean']:7.1f} sd={s['sd']:6.1f}  "
                       f"between={s['between_share']:.3f}  "
                       f"reward_soon rate={s['reward_soon_rate']:.3f}")
        out.append("")

    for source in sorted(set(df.source)):
        out.append(f"=== source: {source} " + "=" * 40)
        d = df[df.source == source]
        for algo in ["MFEC", "NEC"]:
            da = d[d.algo == algo]
            if da.empty:
                continue
            out.append(f"\n-- {algo}")
            for game in [g for g in GAME_ORDER if g in set(da.game)]:
                out.append(f"  {game}")
                sub = da[da.game == game]
                for e in [x for x in ARM_ORDER if x in set(sub.encoder)]:
                    s = sub[sub.encoder == e]
                    # phi=1 means the five seeds share one frozen encoder, so
                    # the +-0.000 beside it is structural, not a measurement.
                    n_phi = (int((~s["phi_shared"].astype(bool)).sum())
                             if "phi_shared" in s else len(s))
                    out.append(
                        f"    {ARM[e][0]:<12s} d={int(s.d.iloc[0]):4d} "
                        f"n={int(s.n.mean()):6d} seeds={len(s)} "
                        f"phi={n_phi}  "
                        f"R2lin {s.r2_linear.mean():+.3f}+-{s.r2_linear.std(ddof=1):.3f}  "
                        f"R2lin@64 {s.r2_linear_pca64.mean():+.3f}  "
                        f"R2knn {s.r2_knn.mean():+.3f}+-{s.r2_knn.std(ddof=1):.3f}  "
                        f"null {s.r2_linear_null.mean():+.3f}  "
                        f"coh {s.coherence.mean():.3f}"
                        + (f"  balacc {s.bal_acc.mean():.3f}"
                           if "bal_acc" in s and s.bal_acc.notna().any() else ""))

    # Do the two data sources tell the same story?  If they do not, every
    # ranking claim has to be qualified by which set it was measured on -- so
    # this is reported rather than assumed.
    if {"probe", "memory"} <= set(df.source):
        out.append("\n=== agreement between sources " + "=" * 28)
        for algo in ["MFEC", "NEC"]:
            for metric in ["r2_linear", "r2_knn", "coherence"]:
                a = (df[(df.algo == algo) & (df.source == "probe")]
                     .groupby(["game", "encoder"])[metric].mean())
                b = (df[(df.algo == algo) & (df.source == "memory")]
                     .groupby(["game", "encoder"])[metric].mean())
                common = a.index.intersection(b.index)
                if len(common) < 4:
                    continue
                rho = pd.Series(a[common]).corr(pd.Series(b[common]),
                                                method="spearman")
                out.append(f"  {algo:4s} {metric:12s} Spearman rho = {rho:+.3f} "
                           f"over {len(common)} (game, encoder) cells")

    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[:4]))


# ---------------------------------------------------------------------------

def save(fig_and_name):
    if fig_and_name is None:
        return
    fig, name = fig_and_name
    for d in FIGDIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", default=EMB_DIR)
    p.add_argument("--games", nargs="+", default=GAME_ORDER)
    p.add_argument("--algos", nargs="+", default=["MFEC", "NEC"])
    p.add_argument("--sources", nargs="+", default=["probe", "memory"],
                   choices=["probe", "memory"])
    p.add_argument("--max-rows", type=int, default=8000,
                   help="cap on rows fed to the probes (default 8000)")
    p.add_argument("--umap-rows", type=int, default=6000)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-seed", type=int, default=42,
                   help="which seed's run is projected (default 42)")
    p.add_argument("--skip-umap", action="store_true")
    p.add_argument("--skip-metrics", action="store_true",
                   help="reuse embedding_metrics.csv instead of recomputing")
    p.add_argument("--no-cache-filter", action="store_true",
                   help="do NOT restrict to runs present in the W&B cache; "
                        "only for analysing a run outside the study grid")
    args = p.parse_args()

    csv_path = os.path.join(HERE, "data", "embedding_metrics.csv")
    if args.skip_metrics and args.skip_umap and os.path.exists(csv_path):
        # Figures-only path: the bundled metrics and probe labels are all the
        # tables, probe/coherence figures and numbers need, so the embeddings
        # themselves are not required.
        runs, labels = [], load_labels(LABEL_DIR)
        print(f"figures-only: {len(labels)} probe label set(s) from {LABEL_DIR}")
    else:
        runs, labels = load_runs(args.emb_dir, set(args.algos), set(args.games),
                                 cache_filter=not args.no_cache_filter)
        print(f"{len(runs)} run(s), {len(labels)} probe label set(s) "
              f"from {args.emb_dir}")

    if args.skip_metrics and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"reusing {csv_path} ({len(df)} rows)")
    else:
        df = compute_metrics(runs, labels, args.sources, args)
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path} ({len(df)} rows)")

    for source in args.sources:
        for algo in args.algos:
            save(figure_probe(df, algo, source, args.games))
            save(figure_coherence(df, algo, source, args.games))
        latex_table(df, source, args.games, f"embedding_table_{source}")

    if not args.skip_umap:
        for source in args.sources:
            for algo in args.algos:
                save(figure_umap(runs, labels, source, algo, args.games, args))

    write_numbers(df, os.path.join(HERE, "figures", "embedding_numbers.txt"), args, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
