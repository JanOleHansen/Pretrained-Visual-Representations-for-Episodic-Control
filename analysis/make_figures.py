#!/usr/bin/env python3
"""
Regenerate every results figure in the defence deck from the cached W&B runs.

    python make_figures.py --refresh     # re-pull from W&B, then plot
    python make_figures.py               # plot from the local cache

Cache layout:  <cache>/runs_meta.csv  +  <cache>/wandb_cache/<run_id>.csv
Output:        figures/*.pdf   (vector, picked up directly by Beamer)
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ENTITY_PROJECT = "LatentLab/torchrl-hydra-template"

# --- MFEC validity filter -------------------------------------------------
# MFEC's bit-exact key lookup is unreadable when phi runs on TF32 convolutions
# (see pin_fp32_conv_precision in src/encoders/factory.py).  The failure
# signature is a memory hit rate of identically 0.000, because an unreadable
# memory never hits.  A run is therefore invalid iff its memory was never
# readable; the rule is applied uniformly to every arm, with no encoder
# exemptions.  Section 5.1 of the thesis states this criterion and its effect.
TF32_FIX = "2026-08-19T08:52:59Z"   # time FP32 was pinned; informational only
HIT_RATE = "sum::eval/memory_hit_rate"
QEC_SIZE = "sum::train/qec_size"

# A plot needs enough seeds to be worth showing.  Cells thinner than this are
# dropped from the *figures* (the tables still list them, marked with an
# asterisk); if nothing survives, the figure is not written at all and the
# frame falls back to its PLACEHOLDER box.
MIN_SEEDS = 3
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures", "deck")   # defence-deck sizing
DEFAULT_CACHE = os.environ.get("PVR_CACHE", os.path.join(HERE, "data", "runs"))

# Colours mirror the \definecolor block in main.tex so slides and figures agree.
COL = {"baseline": "#7F7F7F", "supervised": "#C05746", "selfsup": "#2E7D5B",
       "contrastive": "#2F6690", "recon": "#8E6C9B",
       # The two random-projection controls need to be told apart from each
       # other AND from the neutral grey used for shape-only legend keys.
       "rp_gray": "#111111", "rp_rgb": "#9A9A9A"}

# encoder -> (display label, colour, linestyle)
ARM = {
    "random_projection":     ("RP (gray)",   COL["rp_gray"],     (0, (1, 1.4))),
    "random_projection_rgb": ("RP (RGB)",    COL["rp_rgb"],      "-"),
    "none":                  ("ConvNet",     COL["baseline"],    "-"),
    "resnet":                ("ResNet",      COL["supervised"],  "-"),
    "dinov2":                ("DINOv2",      COL["selfsup"],     "-"),
    "clip":                  ("CLIP",        COL["contrastive"], "-"),
    "mae":                   ("MAE",         COL["recon"],       "-"),
    "vae":                   ("VAE",         COL["recon"],       (0, (4, 2))),
}

# The frozen-NEC control (RQ4).  Each shares its backbone's hue -- hue is the
# backbone, linestyle is whether phi was trained -- so a frozen/fine-tuned pair
# reads as one entity in two states rather than as two unrelated arms.
#
# DELIBERATELY NOT IN ARM_ORDER.  Every existing figure selects its arms with
# `[e for e in ARM_ORDER if e in set(s.encoder)]`, so adding these there would
# silently inject four extra arms into the RQ1/RQ2 figures and tables, which
# answer a different question.  The RQ4 figures carry their own ordering
# (FROZEN_PAIRS below); this dict exists only so labels and colours resolve.
ARM.update({
    "resnet_frozen": ("ResNet (frozen)", COL["supervised"],  (0, (4, 2))),
    "dinov2_frozen": ("DINOv2 (frozen)", COL["selfsup"],     (0, (4, 2))),
    "clip_frozen":   ("CLIP (frozen)",   COL["contrastive"], (0, (4, 2))),
    "mae_frozen":    ("MAE (frozen)",    COL["recon"],       (0, (4, 2))),
})

#: (fine-tuned, frozen) NEC arms, in the order RQ4 reports them.  The pairing is
#: the unit of analysis: configs/experiment/nec/<game>_<arm>_frozen.yaml differs
#: from <game>_<arm>.yaml in `freeze_backbone` and nothing else, so the delta
#: within a pair isolates fine-tuning with the algorithm, the backbone, the key
#: width and every hyperparameter held fixed.
FROZEN_PAIRS = [("resnet", "resnet_frozen"), ("dinov2", "dinov2_frozen"),
                ("clip", "clip_frozen"), ("mae", "mae_frozen")]

ARM_ORDER = ["random_projection", "random_projection_rgb", "none",
             "resnet", "dinov2", "clip", "mae", "vae"]
GAME_ORDER = ["MsPacman", "Qbert", "Frostbite"]
# One colour per game, used wherever games are grouped (RQ3 retrieval).
GAME_COLOR = {"MsPacman": "#4C72B0", "Qbert": "#DD8452", "Frostbite": "#55A868"}

# Atari-100k random / human reference scores (Wang et al. 2016 conventions).
REF = {"MsPacman": (307.3, 6951.6), "Qbert": (163.9, 13455.0), "Frostbite": (65.2, 4334.7)}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _ksteps(x, _pos=None):
    """Axis labels as 20k / 100k rather than 0.2 ... 1.0 against a '1e5' offset.

    The offset form makes the reader multiply two numbers to recover a step
    count, and the exponent sits detached in the corner where it is easy to
    miss entirely."""
    if x == 0:
        return "0"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:g}k"
    return f"{x:g}"


# --------------------------------------------------------------------- data
def refresh(cache):
    """Re-pull configs, summaries and histories for every MFEC/NEC run."""
    import wandb
    hist_dir = os.path.join(cache, "wandb_cache")
    os.makedirs(hist_dir, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs = list(api.runs(ENTITY_PROJECT,
                         filters={"config.algorithm._target_":
                                  {"$regex": "(MFEC|NEC)Algorithm"}}, per_page=500))

    def g(c, *path, default=None):
        cur = c
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    meta = []
    for i, r in enumerate(runs):
        c = r.config
        row = dict(
            id=r.id, name=r.name, state=r.state, created=r.created_at,
            algo=str(g(c, "algorithm", "_target_", default="?")).split(".")[-1].replace("Algorithm", ""),
            game=str(g(c, "environment", "name", default="?")).replace("ALE/", "").replace("-v5", ""),
            encoder=str(g(c, "algorithm", "encoder_type",
                          default=g(c, "run", "encoder", default="?"))),
            seed=g(c, "trainer", "seed"), total_frames=g(c, "trainer", "total_frames"),
            runtime_s=r.summary.get("_runtime"), num_envs=g(c, "trainer", "num_envs"))
        for k in r.summary.keys():
            if not k.startswith("_"):
                v = r.summary[k]
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row["sum::" + k] = v
        meta.append(row)
        # Re-download a cached history whenever it is SHORTER than the run.
        # A plain os.path.exists() check is not enough: a history cached while
        # the run was still in flight is a partial history, and because the run
        # later reaches "finished" nothing would ever invalidate it.  That is
        # not a cosmetic staleness -- curves() truncates a whole arm to its
        # shortest seed, so one mid-flight cache entry cuts the arm's line off
        # at the step it happened to be cached at.  (Measured: MFEC/MsPacman/
        # rp_gray seed 43 cached at 4 eval points ending 1190, live 10 points
        # ending 2730 -- the entire RP-gray baseline plotted to 40k, not 100k.)
        hp = os.path.join(hist_dir, f"{r.id}.csv")
        want = r.summary.get("_step")
        stale_hist = True
        if os.path.exists(hp):
            try:
                have = pd.read_csv(hp)["_step"].max()
                stale_hist = want is not None and have < want
            except Exception:
                stale_hist = True
        if stale_hist:
            pd.DataFrame(r.history(samples=2000, pandas=False)).to_csv(hp, index=False)
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(runs)}")
    pd.DataFrame(meta).to_csv(os.path.join(cache, "runs_meta.csv"), index=False)
    print(f"refreshed {len(meta)} runs into {cache}")


def load(cache, include_pre_fix=False):
    """Finished runs only, one run per (algo, game, encoder, seed) -- latest wins.

    MFEC runs whose episodic memory was never readable (the measured TF32
    failure signature: a hit rate of identically zero over a non-empty memory)
    are dropped, uniformly across arms.  `include_pre_fix` keeps even those.
    See the TF32 block at the top of this file for why this is measured rather
    than dated.
    """
    m = pd.read_csv(os.path.join(cache, "runs_meta.csv"))
    m = m[m.state == "finished"].copy()
    if not include_pre_fix:
        stored = m[QEC_SIZE] > 0 if QEC_SIZE in m.columns else True
        broken = (m.algo == "MFEC") & (m[HIT_RATE] == 0) & stored
        if broken.any():
            print(f"dropping {int(broken.sum())} MFEC run(s) with an unreadable "
                  f"memory (hit rate 0.000 over a non-empty buffer)")
        m = m[~broken]
    m = (m.sort_values("created")
           .drop_duplicates(["algo", "game", "encoder", "seed"], keep="last"))
    return m


def curves(cache, meta, metric="eval/return_mean"):
    """(algo, game, encoder) -> (steps, per-seed matrix) on a common step grid."""
    out = {}
    for key, grp in meta.groupby(["algo", "game", "encoder"]):
        series = []
        for _, r in grp.iterrows():
            p = os.path.join(cache, "wandb_cache", f"{r['id']}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            if metric not in d.columns:
                continue
            d = d.dropna(subset=[metric])
            if len(d):
                series.append(d[["_step", metric]].rename(columns={metric: r["seed"]}))
        if not series:
            continue
        n = min(len(s) for s in series)
        steps = series[0]["_step"].values[:n]
        mat = np.vstack([s[s.columns[1]].values[:n] for s in series])
        out[key] = (steps, mat)
    return out


def thick_cells(meta, algo):
    """Rows of `meta` whose (game, encoder) cell has at least MIN_SEEDS seeds."""
    s = meta[meta.algo == algo]
    n = s.groupby(["game", "encoder"]).size()
    keep = {k for k, v in n.items() if v >= MIN_SEEDS}
    return s[[(g, e) in keep for g, e in zip(s.game, s.encoder)]]


def band(ax, steps, mat, label, color, ls):
    """Mean line with a standard-error band across seeds."""
    mu = mat.mean(0)
    ax.plot(steps, mu, color=color, ls=ls, lw=1.6, label=label, zorder=3)
    if mat.shape[0] > 1:
        se = mat.std(0, ddof=1) / np.sqrt(mat.shape[0])
        ax.fill_between(steps, mu - se, mu + se, color=color, alpha=0.15, lw=0, zorder=2)


# ------------------------------------------------------------------- figures
def fig_curves(cv, meta, algo, fname, title):
    cv = {k: v for k, v in cv.items()
          if k[0] != algo or v[1].shape[0] >= MIN_SEEDS}
    games = [g for g in GAME_ORDER
             if sum(1 for a, gm, _ in cv if a == algo and gm == g) >= 2]
    if not games:
        print(f"  skipped {fname}: no game has {MIN_SEEDS}+ seeds on 2+ encoders")
        return
    fig, axes = plt.subplots(1, len(games), figsize=(3.5 * len(games), 2.5), squeeze=False)
    seen = {}
    for ax, gm in zip(axes[0], games):
        for enc in ARM_ORDER:
            if (algo, gm, enc) not in cv:
                continue
            steps, mat = cv[(algo, gm, enc)]
            lab, col, ls = ARM[enc]
            # NEC's from-scratch ConvNet *is* the "learned keys" of RQ2; say so
            # in the legend rather than expecting the audience to infer it.
            if algo == "NEC" and enc == "none":
                lab = "ConvNet (learned)"
            lab_n = f"{lab} (n={mat.shape[0]})" if mat.shape[0] < 5 else lab
            band(ax, steps, mat, lab_n, col, ls)
            seen.setdefault(lab, (col, ls))
        rnd = REF.get(gm, (None, None))[0]
        if rnd is not None:
            ax.axhline(rnd, color="k", lw=0.7, ls=":", alpha=0.6, zorder=1)
            ax.annotate("random", (steps[0], rnd), fontsize=6.5, va="bottom",
                        ha="left", alpha=0.6)
        ax.set_title(gm)
        ax.set_xlabel("agent steps")
        ax.xaxis.set_major_formatter(FuncFormatter(_ksteps))
        ax.legend(frameon=False, loc="upper left", handlelength=1.6,
                  labelspacing=0.25, borderpad=0.2)
    axes[0][0].set_ylabel("evaluation return")
    fig.suptitle(title, y=1.04, fontsize=10)
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_hns(meta, fname):
    """Human-normalised final score per arm, per game, MFEC + NEC side by side."""
    col = "sum::eval/return_mean"
    if not len(thick_cells(meta, "MFEC")) and not len(thick_cells(meta, "NEC")):
        print(f"  skipped {fname}: no cell has {MIN_SEEDS}+ seeds")
        return
    # sharey is deliberate.  On independent axes MFEC's ~50% and NEC's ~11%
    # each fill their own panel and the slide reads as "the two agents perform
    # about the same" -- the opposite of what the numbers say, and the kind of
    # thing a committee reads off a figure in five seconds without checking the
    # tick labels.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), squeeze=False, sharey=True)
    for ax, algo in zip(axes[0], ["MFEC", "NEC"]):
        s = thick_cells(meta, algo)
        if not len(s):
            ax.set_visible(False)
            continue
        encs = [e for e in ARM_ORDER if e in set(s.encoder)]
        games = [g for g in GAME_ORDER if g in set(s.game)]
        w = 0.8 / max(len(games), 1)
        for j, gm in enumerate(games):
            rnd, hum = REF[gm]
            vals, errs = [], []
            for e in encs:
                v = s[(s.encoder == e) & (s.game == gm)][col].dropna()
                hns = (v - rnd) / (hum - rnd) * 100 if len(v) else pd.Series(dtype=float)
                vals.append(hns.mean() if len(hns) else np.nan)
                errs.append(hns.std(ddof=1) / np.sqrt(len(hns)) if len(hns) > 1 else 0.0)
            x = np.arange(len(encs)) + j * w - 0.4 + w / 2
            ax.bar(x, vals, w * 0.92, yerr=errs, capsize=2, label=gm,
                   color=GAME_COLOR.get(gm, "0.5"), edgecolor="k", lw=0.4,
                   error_kw=dict(lw=0.7))
        ax.set_xticks(np.arange(len(encs)))
        ax.set_xticklabels([ARM[e][0] for e in encs], rotation=30, ha="right")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(algo)
        ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
    axes[0][0].set_ylabel("human-normalised\nscore (%)")
    # Headroom for the legend.  The axes are shared, so the tallest bar in
    # EITHER panel sets the top -- and it is MFEC/MAE, directly under the
    # upper-right legend, whose "Frostbite" key lands on the bar without this.
    top = max((ax.get_ylim()[1] for ax in axes[0] if ax.get_visible()), default=1)
    axes[0][0].set_ylim(top=top * 1.16)
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_retrieval(cache, meta, fname):
    """RQ3: retrieval geometry.  Hit rate is reported per game -- pooling hides
    that it is strongly game-dependent (Frostbite ~0.05 vs. Qbert ~0.28)."""
    # Thin cells are SHOWN here, hatched and labelled with their n, rather than
    # dropped: a silently absent bar (CLIP/Qbert, n=2) reads as a broken
    # pipeline to an audience counting legend entries.
    s = meta[meta.algo == "MFEC"]
    if not len(s):
        print(f"  skipped {fname}: no MFEC runs")
        return
    encs = [e for e in ARM_ORDER if e in set(s.encoder)]
    games = [g for g in GAME_ORDER if g in set(s.game)]
    hit = "sum::eval/memory_hit_rate"

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1.15]))

    # left: hit rate per encoder, grouped by game
    ax = axes[0][0]
    w = 0.8 / max(len(games), 1)
    for j, gm in enumerate(games):
        vals, errs = [], []
        ns = []
        for e in encs:
            v = s[(s.encoder == e) & (s.game == gm)][hit].dropna()
            vals.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            ns.append(len(v))
        x = np.arange(len(encs)) + j * w - 0.4 + w / 2
        bars = ax.bar(x, vals, w * 0.92, yerr=errs, capsize=2, label=gm,
                      color=GAME_COLOR[gm], edgecolor="k", lw=0.4,
                      error_kw=dict(lw=0.7))
        for b, v, k in zip(bars, vals, ns):
            if 0 < k < MIN_SEEDS and not np.isnan(v):
                b.set_hatch("////")
                ax.annotate(f"n={k}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=6, color="k")
    ax.set_xticks(np.arange(len(encs)))
    ax.set_xticklabels([ARM[e][0] for e in encs], rotation=30, ha="right")
    ax.set_ylabel("memory hit rate")
    ax.set_title("Neighbour retrieval")
    ax.legend(frameon=False, ncol=3, fontsize=7)

    # right: hit rate vs. score, one point per (encoder, game)
    ax = axes[0][1]
    mk = {"MsPacman": "o", "Qbert": "s", "Frostbite": "^"}
    for e in encs:
        for gm in games:
            d = s[(s.encoder == e) & (s.game == gm)]
            if not len(d):
                continue
            x = d[hit].dropna().mean()
            rnd, hum = REF[gm]
            y = ((d["sum::eval/return_mean"].dropna() - rnd) / (hum - rnd) * 100).mean()
            if np.isnan(x) or np.isnan(y):
                continue
            ax.scatter(x, y, s=34, color=ARM[e][1], edgecolor="k", lw=0.4,
                       marker=mk[gm], zorder=3)
    ax.set_xlabel("memory hit rate  (how OFTEN the memory answers)")
    ax.set_ylabel("human-normalised\nscore (%)  (how WELL)")
    ax.set_title("Retrieving often $\\neq$ retrieving well")
    handles = [plt.Line2D([], [], marker=m, ls="", color="0.75", mec="k",
                          mew=0.4, label=g) for g, m in mk.items() if g in games]
    handles += [plt.Line2D([], [], marker="o", ls="", color=ARM[e][1], mec="k",
                           mew=0.4, label=ARM[e][0]) for e in encs]
    # Outside the axes: at projection size an in-plot legend either covers
    # points or forces a font nobody can read.
    ax.legend(handles=handles, frameon=False, fontsize=6, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              handletextpad=0.3, labelspacing=0.35)
    # Marker = game, colour = encoder.  Vertical spread at fixed x is the
    # point: same retrieval frequency, wildly different score.
    # The reading instructions live on the slide, not in the axes -- at
    # projection size an in-plot paragraph is unreadable and collides with data.
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


# Encoder specifications.  These are analytic, not measured: wall-clock from
# W&B is unusable here because the runs shared GPUs, so a slower arm may simply
# have had a busier device.  Key width d is what drives episodic-memory cost.
ENCODER_SPEC = {
    "random_projection":     ("--",                       "64",  "gray 84x84"),
    "random_projection_rgb": ("--",                       "64",  "RGB 84x84"),
    "resnet":                ("ResNet-18 (IN-1k, sup.)",  "512", "RGB 224x224"),
    "dinov2":                ("ViT-S/14 (DINOv2)",        "384", "RGB 224x224"),
    "clip":                  ("ViT-B/32 (CLIP)",          "512", "RGB 224x224"),
    "mae":                   ("ViT-B/16 (MAE)",           "768", "RGB 224x224"),
}


def fig_cost_perf(meta, fname):
    """RQ1 as a cost-benefit question: what does a bigger key actually buy?

    Two panels on the same log-memory x-axis.  Left pools all three games and
    draws the Pareto frontier, which makes the dominated arm visible -- DINOv2
    costs 6x a random projection and scores below it.  Right is Q*bert alone,
    where the frontier inverts and the cheapest arm is also the best; that is
    the reversal restated as a purchasing decision rather than a score.
    """
    s = thick_cells(meta, "MFEC")
    encs = [e for e in ARM_ORDER if e in set(s.encoder) and e in ENCODER_SPEC]
    if not encs:
        print(f"  skipped {fname}: no MFEC cells")
        return

    def hns(e, games):
        out = []
        for g in games:
            lo, hi = REF[g]
            v = s[(s.encoder == e) & (s.game == g)]["sum::eval/return_mean"]
            out += list((v.dropna() - lo) / (hi - lo) * 100)
        return float(np.mean(out)) if out else np.nan

    games = [g for g in GAME_ORDER if g in set(s.game)]
    panels = [("All three games (pooled)", games),
              ("Q*bert alone --- the order flips", ["Qbert"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.9), squeeze=False, sharex=True)
    for ax, (title, gs) in zip(axes[0], panels):
        pts = []
        for e in encs:
            mb = int(ENCODER_SPEC[e][1]) * 4 * 100_000 / 1e6   # float32 keys
            y = hns(e, gs)
            if not np.isnan(y):
                pts.append((mb, y, e))
        # Pareto frontier: cheapest-first, keep each new best score.
        front, best = [], -np.inf
        for mb, y, e in sorted(pts):
            if y > best:
                front.append((mb, y)); best = y
        if len(front) > 1:
            fx, fy = zip(*front)
            ax.step(fx, fy, where="post", color="k", lw=0.9, ls="--",
                    alpha=0.45, zorder=1)
        for mb, y, e in pts:
            lab, col, _ = ARM[e]
            dominated = all(not (o[0] < mb and o[1] >= y) for o in pts)
            ax.scatter(mb, y, s=70, color=col, edgecolor="k",
                       lw=0.9 if dominated else 1.6, zorder=3,
                       marker="o" if dominated else "X")
            # RP (gray) and RP (RGB) share an x; stagger them vertically.
            dy = 9 if e != "random_projection_rgb" else -15
            ax.annotate(lab, (mb, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=6.5)
        ax.set_xscale("log")
        ticks = sorted({int(ENCODER_SPEC[e][1]) * 4 * 100_000 // 10**6
                        for e in encs})
        ax.set_xticks(ticks, minor=False)
        ax.set_xticklabels([f"{t}" for t in ticks], fontsize=7)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("episodic memory per action (MB, log scale)")
        ax.set_title(title, fontsize=8.5, loc="left", fontweight="bold")
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_axisbelow(True)
        ax.margins(y=0.28)
    axes[0][0].set_ylabel("human-normalised score (%)")
    axes[0][1].annotate("X = pays more, scores less\nthan a cheaper encoder",
                        xy=(0.98, 0.04), xycoords="axes fraction", fontsize=6,
                        ha="right", va="bottom", alpha=0.8)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def table_cost(meta, fname, compact=False):
    """Analytic encoder cost: key width and the memory it implies per action.

    `compact` drops the parenthetical in the backbone name and shortens the
    header.  On a slide this table shares the frame with the results table and
    the full form overruns the column -- and the parenthetical is redundant
    there anyway, because the Arm column already names the family.
    """
    s = meta[meta.algo == "MFEC"]
    encs = [e for e in ARM_ORDER if e in set(s.encoder)]
    buf = 100_000  # algorithm.buffer_size, identical across arms
    head = r"Arm & Backbone & $d$ & Mem/act. \\" if compact else \
           r"Arm & Backbone & $d$ & Memory/action \\"
    lines = [r"% auto-generated by make_figures.py -- do not edit by hand",
             r"\begin{tabular}{llrr}", r"\toprule", head, r"\midrule"]
    for e in encs:
        bb, d, _ = ENCODER_SPEC[e]
        if compact:
            bb = bb.split(" (")[0]
        mb = buf * int(d) * 4 / 1024**2   # float32 keys
        lines.append(rf"{ARM[e][0]} & {bb} & {d} & {mb:.0f}\,MB \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(FIGDIR, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", fname)


def table_results(meta, fname):
    """LaTeX booktabs table of final scores; \\input straight into a frame."""
    col = "sum::eval/return_mean"
    lines = [r"% auto-generated by make_figures.py -- do not edit by hand",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Agent & Encoder & " + " & ".join(GAME_ORDER) + r" \\", r"\midrule"]
    for algo in ["MFEC", "NEC"]:
        s = meta[meta.algo == algo]
        if not len(s):
            continue
        encs = [e for e in ARM_ORDER if e in set(s.encoder)]
        for i, e in enumerate(encs):
            cells = []
            for gm in GAME_ORDER:
                v = s[(s.encoder == e) & (s.game == gm)][col].dropna()
                if not len(v):
                    cells.append(r"--")
                else:
                    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                    star = "" if len(v) >= 5 else r"\textsuperscript{*}"
                    cells.append(rf"${v.mean():.0f}\pm{se:.0f}${star}")
            head = rf"\multirow{{{len(encs)}}}{{*}}{{{algo}}}" if i == 0 else ""
            lines.append(f"{head} & {ARM[e][0]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]
    with open(os.path.join(FIGDIR, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", fname)


def coverage(meta):
    print("\nseed coverage (finished runs, deduplicated):")
    for algo in ["MFEC", "NEC"]:
        s = meta[meta.algo == algo]
        if not len(s):
            continue
        print(f"  {algo}: " + ", ".join(
            f"{ARM[e][0]}/{gm}={len(s[(s.encoder==e)&(s.game==gm)])}"
            for e in ARM_ORDER if e in set(s.encoder) for gm in GAME_ORDER
            if len(s[(s.encoder == e) & (s.game == gm)]) < 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull from W&B first")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--include-pre-fix", action="store_true",
                    help="keep MFEC runs from before the TF32 fix (not valid measurements)")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    # Clear generated artefacts first: a figure that can no longer be built --
    # because its runs were filtered out -- must vanish rather than linger as a
    # stale file the deck would happily \includegraphics.  The frames fall back
    # to a visible PLACEHOLDER box when the file is absent.
    # Only this script's own outputs are cleared.  The rliable figures live in
    # the same directory but are produced by make_rliable_figures.py (separate
    # venv), so a blanket fig_*.pdf wipe would delete them.
    OWNED = {"fig_mfec_curves", "fig_nec_curves", "fig_hns", "fig_retrieval",
             "fig_cost_perf",
             "tab_results", "tab_cost"}
    for f in os.listdir(FIGDIR):
        if f.rsplit(".", 1)[0] in OWNED and f.endswith((".pdf", ".tex", ".png")):
            os.remove(os.path.join(FIGDIR, f))
    if a.refresh:
        refresh(a.cache)
    if not os.path.exists(os.path.join(a.cache, "runs_meta.csv")):
        sys.exit(f"no cache at {a.cache}; run once with --refresh")
    meta = load(a.cache, a.include_pre_fix)
    print(f"{len(meta)} finished runs after dedup")
    cv = curves(a.cache, meta)
    fig_curves(cv, meta, "MFEC", "fig_mfec_curves.pdf",
               "MFEC: pretrained keys vs. random projection")
    fig_curves(cv, meta, "NEC", "fig_nec_curves.pdf", None)
    fig_hns(meta, "fig_hns.pdf")
    fig_retrieval(a.cache, meta, "fig_retrieval.pdf")
    fig_cost_perf(meta, "fig_cost_perf.pdf")
    table_cost(meta, "tab_cost.tex", compact=True)
    table_results(meta, "tab_results.tex")
    coverage(meta)


if __name__ == "__main__":
    main()
