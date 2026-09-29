#!/usr/bin/env python3
"""
Stratified-bootstrap evaluation of the encoder arms, per Agarwal et al.,
"Deep Reinforcement Learning at the Edge of the Statistical Precipice"
(NeurIPS 2021) -- https://github.com/google-research/rliable

    python make_rliable_figures.py
    python make_rliable_figures.py --algos NEC

Both encoder ablations are bootstrapped, each against its own control (see
BASELINES): MFEC's arms against the matched-observation random projection
(RQ1), NEC's against its jointly trained ConvNet (RQ2).  MFEC writes the
unsuffixed figure names; NEC writes the same three figures with a `_nec`
suffix, so the two sets never overwrite each other.

WHY THIS RUNS IN ITS OWN VIRTUALENV
-----------------------------------
rliable pins `arch<8.0`, and arch 7.x cannot import under pandas 3 (it calls
`pandas.util._decorators.deprecate_kwarg` with the pre-3.0 signature).  arch 8
fixes that but changes the bootstrap kwargs rliable subclasses.  There is no
version triple that satisfies both, so rliable lives in `.venv-rliable`
(pandas 2.3.x) and the rest of the figure pipeline keeps the main venv.

WHAT THIS ADDS OVER THE ±s.e. BARS
----------------------------------
Five seeds is far too few for a normal-theory standard error to mean what it
looks like; that is precisely the failure the rliable paper documents.  These
figures replace point estimates with stratified-bootstrap interval estimates,
and add a probability-of-improvement statistic that answers RQ1 directly
instead of by eyeballing whether two error bars overlap.

WHAT THE INTERVALS RESAMPLE -- AND WHAT THEY DO NOT
---------------------------------------------------
rliable's `get_interval_estimates` takes `task_bootstrap=False` by DEFAULT, and
nothing here overrides it.  So every interval in these figures resamples SEEDS,
stratified within each game; the set of games is held fixed.  In Agarwal et
al.'s own words that "captures the statistical uncertainty ... if the
experiment is repeated using a different set of runs (e.g., changing seeds) on
the same set of tasks" -- it is NOT a statement about a larger population of
games.  Describe it that way in prose: with three games, `task_bootstrap=True`
produces intervals so wide that nothing separates from anything (measured: the
NEC DINOv2-vs-ConvNet pooled lower bound falls from 0.51 to ~0.34), which is
the honest consequence of three tasks and the reason the per-game panels carry
the argument instead of the pooled one.

The rectangular-matrix requirement below is therefore NOT because tasks are
resampled -- it is because the aggregate metrics operate on a full
(seeds x games) array and a NaN poisons it.

READ THE PER-GAME PANEL, NOT ONLY THE AGGREGATE
-----------------------------------------------
This matters twice over.  The headline MFEC finding is a *reversal*: pretrained keys win on Ms. Pac-Man
and Frostbite and lose on Q*bert.  Any statistic aggregated over the three
games averages that reversal away -- and with only three tasks the aggregate is
thin to begin with (rliable's own study uses 26).  So the aggregate panels are
reported alongside, never instead of, the per-game probability of improvement.

NEC concentrates the same problem into one game.  Its pretrained arms separate
from the ConvNet on Frostbite alone, and they do it *bimodally*: per arm, some
seeds stay pinned at the ConvNet's ~220 return and the rest jump to ~1000-1300,
so the mean lands between two modes that no run occupies.  That is exactly the
regime an IQM and a probability of improvement are built for and a +-s.e. bar
is not, which is why NEC is bootstrapped here rather than read off the bars.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rliable import library as rly, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared run loader (make_figures.py) sits next to this script in
# analysis/; "DECK" is kept as the name for where it lives.
DECK = HERE
sys.path.insert(0, DECK)
import make_figures as mf          # noqa: E402

CACHE = os.environ.get("PVR_CACHE", os.path.join(HERE, "data", "runs"))
SEEDS = [42, 43, 44, 45, 46]
# The control each algorithm's arms are measured against.  They are different
# questions, so they need different controls: RQ1 asks whether pretrained keys
# beat a *matched-observation random projection* inside MFEC, while RQ2 asks
# whether they beat NEC's own jointly trained ConvNet ("none").
BASELINES = {"MFEC": "random_projection_rgb", "NEC": "none"}

# MFEC keeps the unsuffixed filenames the deck and the thesis already
# \includegraphics; NEC's figures are new files, so nothing that exists moves.
SUFFIX = {"MFEC": "", "NEC": "_nec"}

# Display names for the per-game panels.  The W&B keys stay as they are
# everywhere else; only the printed panel titles change, so that these figures
# say "Ms. Pac-Man" like the prose and every other figure rather than showing
# the raw run key.
GAME_LABEL = {"MsPacman": "Ms. Pac-Man", "Qbert": "Q*bert",
              "Frostbite": "Frostbite"}


def glabel(g):
    return GAME_LABEL.get(g, g)

# One output profile per document, so each gets figures at its own text width.
# pw/ph size the performance profile, which is the one square-ish figure here:
# the thesis runs it at full text width like every other thesis figure, the
# deck keeps it narrow so it sits on a slide beside a bullet list.
# `term` is the word the axis label uses for one encoder configuration.  The
# deck stopped calling those "arms" after the 2026-08-28 supervisor pass; the
# thesis prose still does, and a figure must match the document it sits in.
TARGETS = {
    "thesis": dict(dir=os.path.join(HERE, "figures"), width=5.95, fs=9,
                   pw=1.0, ph=2.7, term="arm"),
    "deck":   dict(dir=os.path.join(HERE, "figures", "deck"), width=7.6,  fs=8,
                   pw=0.62, ph=2.3, term="encoder"),
}


def score_matrices(meta, algo="MFEC"):
    """arm -> (num_seeds, num_games) human-normalized scores, complete cells only.

    rliable needs a rectangular matrix: the aggregate metrics reduce over a full
    (seeds x games) array and a single NaN poisons the whole estimate.  (Not,
    as this used to say, because games are resampled -- task_bootstrap is off;
    see the module docstring.)  An arm short of a seed anywhere is reported and
    dropped rather than silently padded.
    """
    s = meta[meta.algo == algo]
    out, dropped = {}, []
    for e in mf.ARM_ORDER:
        if e not in set(s.encoder):
            continue
        M = np.full((len(SEEDS), len(mf.GAME_ORDER)), np.nan)
        for j, g in enumerate(mf.GAME_ORDER):
            lo, hi = mf.REF[g]
            for i, sd in enumerate(SEEDS):
                v = s[(s.encoder == e) & (s.game == g) &
                      (s.seed == sd)]["sum::eval/return_mean"]
                if len(v):
                    M[i, j] = (v.iloc[0] - lo) / (hi - lo)
        if np.isnan(M).any():
            dropped.append((mf.ARM[e][0], int(np.isnan(M).sum())))
        else:
            out[e] = M
    return out, dropped


def score_matrices_for(meta, algo, arms):
    """``score_matrices`` over an explicit arm list rather than ``ARM_ORDER``.

    RQ4's arms are the frozen controls, which are deliberately absent from
    ``ARM_ORDER`` so that they cannot leak into the RQ1/RQ2 figures (see the
    note beside ``ARM`` in make_figures.py).  The matrix construction and the
    rectangularity requirement are identical; only the iteration order differs.
    """
    s = meta[meta.algo == algo]
    out, dropped = {}, []
    for e in arms:
        if e not in set(s.encoder):
            continue
        M = np.full((len(SEEDS), len(mf.GAME_ORDER)), np.nan)
        for j, g in enumerate(mf.GAME_ORDER):
            lo, hi = mf.REF[g]
            for i, sd in enumerate(SEEDS):
                v = s[(s.encoder == e) & (s.game == g) &
                      (s.seed == sd)]["sum::eval/return_mean"]
                if len(v):
                    M[i, j] = (v.iloc[0] - lo) / (hi - lo)
        if np.isnan(M).any():
            dropped.append((mf.ARM[e][0], int(np.isnan(M).sum())))
        else:
            out[e] = M
    return out, dropped


def est_poi_paired(mats, reps, pairs):
    """P(fine-tuned > its OWN frozen counterpart), per pair and per panel.

    Distinct from ``est_poi``, which compares every arm against one shared
    control.  RQ4's comparison is *within* a backbone: the two configs differ in
    ``freeze_backbone`` and in nothing else, so each arm is its own control and
    the statistic isolates fine-tuning with the algorithm, the backbone, the key
    width and every hyperparameter held fixed.

    This is NOT a substitute for comparing the frozen arms against the ConvNet
    -- an earlier version of this note claimed that comparison would "re-answer
    RQ2 with extra steps", which was wrong.  RQ2 as run compares *fine-tuned*
    arms against the ConvNet, so it cannot say whether a Frostbite separation
    comes from fine-tuning or from the pretrained features.  Only frozen-vs-
    ConvNet decides that, and it is computed separately (``--frozen-vs-base``,
    on by default).  It finds a counterexample: frozen ResNet-18 beats the
    ConvNet on two games.
    """
    out = {}
    usable = [(ft, fr) for ft, fr in pairs if ft in mats and fr in mats]
    if not usable:
        return None
    for panel in ["All games"] + list(mf.GAME_ORDER):
        sel = (lambda M: M) if panel == "All games" else \
              (lambda M, j=mf.GAME_ORDER.index(panel): M[:, [j]])
        prs = {ft: (sel(mats[ft]), sel(mats[fr])) for ft, fr in usable}
        out[panel] = rly.get_interval_estimates(
            prs, metrics.probability_of_improvement, reps=reps)
    return out


def fig_rq4(mats, tgt, est, pairs):
    """P(fine-tuned > frozen) for each backbone, aggregated and per game."""
    if est is None:
        print("  skipped fig_rliable_rq4.pdf: no complete frozen/fine-tuned pair")
        return
    arms = [ft for ft, fr in pairs if ft in mats and fr in mats]
    panels = list(est)
    fig, axes = plt.subplots(1, len(panels), figsize=(tgt["width"], 1.95),
                             squeeze=False, sharex=True)
    for ax, panel in zip(axes[0], panels):
        pts, cis = est[panel]
        _hbars(ax, arms,
               {e: float(np.ravel(pts[e])[0]) for e in arms},
               {e: float(np.ravel(cis[e])[0]) for e in arms},
               {e: float(np.ravel(cis[e])[1]) for e in arms},
               "", glabel(panel), ref=0.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
    axes[0][0].set_xlabel("P(NEC fine-tuned $>$ NEC frozen)")
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.tight_layout(w_pad=0.8)
    p = os.path.join(tgt["dir"], "fig_rliable_rq4.pdf")
    fig.savefig(p); plt.close(fig)
    print("  wrote", os.path.basename(p))


def dump_rq4_numbers(mats, est, pairs, dropped, reps, tgt):
    """The RQ4 numbers the prose quotes, beside the raw per-cell means."""
    L = ["RQ4 -- NEC, fine-tuned vs frozen (freeze_backbone the only difference)",
         f"  {reps} bootstrap resamples, {len(SEEDS)} seeds x "
         f"{len(mf.GAME_ORDER)} games", ""]
    for nm, n in dropped:
        L.append(f"  EXCLUDED {nm}: {n} missing run(s)")
    if est:
        L.append("P(fine-tuned > frozen), [95% CI]:")
        for panel in est:
            pts, cis = est[panel]
            L.append(f"  {panel}")
            for ft, fr in pairs:
                if ft not in pts:
                    continue
                p = float(np.ravel(pts[ft])[0])
                lo, hi = (float(x) for x in np.ravel(cis[ft])[:2])
                L.append(f"    {mf.ARM[ft][0]:<16s} {p:.3f}  [{lo:.3f}, {hi:.3f}]")
    L.append("\nhuman-normalized score per cell (%), mean +- sd over seeds:")
    for ft, fr in pairs:
        if ft not in mats or fr not in mats:
            continue
        L.append(f"  {mf.ARM[ft][0]}")
        for j, g in enumerate(mf.GAME_ORDER):
            a, b = mats[ft][:, j] * 100, mats[fr][:, j] * 100
            L.append(f"    {g:<10s} fine-tuned {a.mean():6.2f}+-{a.std(ddof=1):5.2f}"
                     f"   frozen {b.mean():6.2f}+-{b.std(ddof=1):5.2f}"
                     f"   delta {a.mean()-b.mean():+6.2f}")
        d = (mats[ft] - mats[fr]).mean() * 100
        L.append(f"    mean delta over games: {d:+.2f}")
    p = os.path.join(tgt["dir"], "rliable_numbers_rq4.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("  wrote", os.path.basename(p))


def frozen_vs_base(meta, reps, targets, dropped_note=True):
    """P(frozen pretrained arm > from-scratch ConvNet), pooled and per game.

    The control RQ2 is missing.  Every NEC arm in the RQ2 grid is fine-tuned, so
    a separation there confounds "the pretrained features are better keys" with
    "a pretrained initialization is easier to optimize".  The frozen arms hold
    the backbone fixed against the same ConvNet baseline and separate the two.

    Reported alongside RQ4 rather than folded into it: RQ4's control is per-arm
    (each backbone against itself), this one is shared (every arm against the
    ConvNet), so the two cannot share a figure without relabelling the axis.
    """
    arms = [fr for _, fr in mf.FROZEN_PAIRS] + ["none"]
    mats, dropped = score_matrices_for(meta, "NEC", arms)
    print(f"\n=== frozen-vs-ConvNet: {len(mats)} complete arms "
          f"(4 frozen + control)")
    for nm, n in dropped:
        print(f"  EXCLUDED {nm}: {n} missing run(s) -- rliable needs a full matrix")
    if "none" not in mats:
        print("  skipped: ConvNet control incomplete")
        return
    print("bootstrapping ...", flush=True)
    e_agg = est_aggregate(mats, reps)
    e_poi = est_poi(mats, reps, "none")
    for name in targets:
        tgt = TARGETS[name]
        os.makedirs(tgt["dir"], exist_ok=True)
        _style(tgt["fs"])
        print(f"[{name}]")
        fig_aggregate(mats, tgt, e_agg, "_frozen", "NEC (frozen backbone)")
        fig_poi(mats, tgt, e_poi, "none", "_frozen", "NEC frozen")
    dump_numbers(mats, e_agg, e_poi, dropped, reps, "NEC (frozen)", "none",
                 "_frozen")


def _style(fs):
    plt.rcParams.update({
        "font.family": "serif", "font.size": fs, "axes.titlesize": fs + 0.5,
        "axes.labelsize": fs, "legend.fontsize": fs - 1,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def _hbars(ax, arms, point, lo, hi, xlabel, title, ref=None):
    """Point estimate with a bootstrap interval, one row per arm."""
    y = np.arange(len(arms))[::-1]
    for yi, e in zip(y, arms):
        c = mf.ARM[e][1]
        ax.plot([lo[e], hi[e]], [yi, yi], color=c, lw=2.6, solid_capstyle="round",
                alpha=0.55, zorder=2)
        ax.plot([point[e]], [yi], marker="o", ms=5, color=c, mec="k", mew=0.5,
                zorder=3)
    if ref is not None:
        ax.axvline(ref, color="k", lw=0.8, ls=":", alpha=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([mf.ARM[e][0] for e in arms])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.25)
    ax.set_axisbelow(True)


AGG_NAMES = ["Median", "IQM", "Mean", "Optimality gap"]


def est_aggregate(mats, reps):
    agg = lambda x: np.array([metrics.aggregate_median(x), metrics.aggregate_iqm(x),
                              metrics.aggregate_mean(x),
                              metrics.aggregate_optimality_gap(x)])
    return rly.get_interval_estimates(mats, agg, reps=reps)


def fig_aggregate(mats, tgt, est, suffix, label=""):
    """Median / IQM / Mean / Optimality gap with 95% stratified-bootstrap CIs.

    `label` names the algorithm/grid the panel belongs to.  It is drawn on the
    figure itself rather than left to the caption: these figures repeat once
    per grid (MFEC, NEC, frozen NEC) with identical axes, so without it a
    reader landing on one cannot tell which of the three it is.
    """
    arms = list(mats)
    names = AGG_NAMES
    pts, cis = est
    fig, axes = plt.subplots(1, 4, figsize=(tgt["width"], 1.95), squeeze=False)
    for i, (ax, nm) in enumerate(zip(axes[0], names)):
        _hbars(ax, arms,
               {e: pts[e][i] for e in arms},
               {e: cis[e][0, i] for e in arms},
               {e: cis[e][1, i] for e in arms},
               "", nm)
        if i:
            ax.set_yticklabels([])
    # One shared label: at this width four per-axis labels collide, and all
    # four panels are in the same units anyway (the optimality gap is the mean
    # shortfall below human, so it is an HNS distance).
    fig.tight_layout(w_pad=0.8)
    fig.supxlabel("human-normalized score" + (f"  \u2013  {label}" if label else ""),
                  y=-0.01)
    p = os.path.join(tgt["dir"], f"fig_rliable_aggregate{suffix}.pdf")
    fig.savefig(p); plt.close(fig)
    print("  wrote", os.path.basename(p))


def est_profile(mats, reps):
    hi = max(m.max() for m in mats.values())
    tau = np.linspace(0.0, hi * 1.02, 41)
    dist, dist_ci = rly.create_performance_profile(mats, tau, reps=reps)
    return tau, dist, dist_ci


def fig_profile(mats, tgt, est, suffix, label=""):
    """Performance profile: fraction of (seed, game) runs scoring above tau."""
    tau, dist, dist_ci = est
    fig, ax = plt.subplots(figsize=(tgt["width"] * tgt["pw"], tgt["ph"]))
    for e in mats:
        lab, c, ls = mf.ARM[e]
        ax.plot(tau, dist[e], color=c, ls=ls, lw=1.6, label=lab, zorder=3)
        ax.fill_between(tau, dist_ci[e][0], dist_ci[e][1], color=c, alpha=0.13,
                        lw=0, zorder=2)
    ax.set_xlabel(r"human-normalized score $\tau$")
    ax.set_ylabel(r"fraction of runs $>\tau$")
    ax.set_title("Performance profile" + (f"  \u2013  {label}" if label else ""),
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, handlelength=1.7, labelspacing=0.25)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    p = os.path.join(tgt["dir"], f"fig_rliable_profile{suffix}.pdf")
    fig.savefig(p); plt.close(fig)
    print("  wrote", os.path.basename(p))


def est_poi(mats, reps, baseline):
    """P(arm > control) with CIs, once for all games and once per game."""
    if baseline not in mats:
        return None
    arms = [e for e in mats if e != baseline]
    out = {}
    for panel in ["All games"] + list(mf.GAME_ORDER):
        sel = (lambda M: M) if panel == "All games" else \
              (lambda M, j=mf.GAME_ORDER.index(panel): M[:, [j]])
        pairs = {e: (sel(mats[e]), sel(mats[baseline])) for e in arms}
        out[panel] = rly.get_interval_estimates(
            pairs, metrics.probability_of_improvement, reps=reps)
    return out


def fig_poi(mats, tgt, est, baseline, suffix, label=""):
    """P(arm > control), aggregated and per game.

    The per-game panels are the point of this figure: the aggregate hides the
    Q*bert reversal, which is the finding.
    """
    if est is None:
        print(f"  skipped fig_rliable_poi{suffix}.pdf: control arm incomplete")
        return
    arms = [e for e in mats if e != baseline]
    panels = list(est)
    fig, axes = plt.subplots(1, len(panels), figsize=(tgt["width"], 2.15),
                             squeeze=False, sharex=True)
    for ax, panel in zip(axes[0], panels):
        pts, cis = est[panel]
        _hbars(ax, arms,
               {e: float(np.ravel(pts[e])[0]) for e in arms},
               {e: float(np.ravel(cis[e])[0]) for e in arms},
               {e: float(np.ravel(cis[e])[1]) for e in arms},
               "", glabel(panel), ref=0.5)
        # A hair of margin past 0 and 1: NEC/DINOv2/Frostbite is P=1.00 exactly
        # (all 25 seed pairs win), and on a hard [0,1] axis its marker is
        # sliced in half by the spine and reads as a plotting glitch.
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
    axes[0][0].set_xlabel(
        f"P({label + ' ' if label else ''}{tgt['term']} $>$ {mf.ARM[baseline][0]})")
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.tight_layout(w_pad=0.8)
    p = os.path.join(tgt["dir"], f"fig_rliable_poi{suffix}.pdf")
    fig.savefig(p); plt.close(fig)
    print("  wrote", os.path.basename(p))


def dump_numbers(mats, e_agg, e_poi, dropped, reps, algo, baseline, suffix):
    """Write the figures' numbers to a text file.

    The thesis and the deck quote these in prose; regenerating the figures
    without regenerating the quoted numbers is how a caption drifts away from
    the plot it describes, so both come out of the same run.
    """
    pts, cis = e_agg
    L = [f"{algo}: stratified bootstrap, {reps} resamples, 95% CI",
         f"arms: {len(mats)} x {len(SEEDS)} seeds x {len(mf.GAME_ORDER)} games"]
    for nm, n in dropped:
        L.append(f"EXCLUDED {nm}: {n} missing run(s)")
    L += ["", "aggregate"]
    for i, nm in enumerate(AGG_NAMES):
        for e in mats:
            L.append(f"  {nm:14s} {mf.ARM[e][0]:10s} {pts[e][i]:.3f} "
                     f"[{cis[e][0, i]:.3f}, {cis[e][1, i]:.3f}]")
    if e_poi:
        L += ["", f"probability of improvement over {mf.ARM[baseline][0]}"]
        for panel, (p, c) in e_poi.items():
            for e in p:
                L.append(f"  {panel:11s} {mf.ARM[e][0]:10s} "
                         f"{float(np.ravel(p[e])[0]):.2f} "
                         f"[{float(np.ravel(c[e])[0]):.2f}, "
                         f"{float(np.ravel(c[e])[1]):.2f}]")
    for name in TARGETS:
        with open(os.path.join(TARGETS[name]["dir"],
                               f"rliable_numbers{suffix}.txt"), "w") as f:
            f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    # rliable's own Atari study uses 50k, but that is sized for 26-task
    # matrices; here each bootstrap draws from 5 seeds x 3 games and the
    # intervals are stable well before 20k.  probability_of_improvement costs
    # ~2 ms per rep per pair and the POI figure needs 16 of them, so 50k would
    # be ~27 min for no visible change in the CIs.
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the stratified bootstrap.  Fixed by default "
                         "so that reruns reproduce the same intervals -- rliable "
                         "draws from np.random, so an unseeded run shifts every CI "
                         "by ~0.01-0.08 and silently desynchronises the numbers "
                         "quoted in the thesis prose from the plotted ones.")
    ap.add_argument("--reps", type=int, default=20_000,
                    help="bootstrap resamples per estimate")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--algos", nargs="+", default=list(BASELINES),
                    choices=list(BASELINES),
                    help="which algorithm's encoder ablation to bootstrap")
    ap.add_argument("--rq4", action="store_true", default=True,
                    help="also bootstrap the NEC frozen/fine-tuned pairs (RQ4)")
    ap.add_argument("--no-rq4", dest="rq4", action="store_false")
    ap.add_argument("--frozen-vs-base", action="store_true", default=True,
                    help="also bootstrap the frozen NEC arms against the "
                         "from-scratch ConvNet -- the control that separates "
                         "'pretrained features help' from 'fine-tuning helps'")
    ap.add_argument("--no-frozen-vs-base", dest="frozen_vs_base",
                    action="store_false")
    a = ap.parse_args()
    np.random.seed(a.seed)

    meta = mf.load(a.cache)
    for algo in a.algos:
        baseline, suffix = BASELINES[algo], SUFFIX[algo]
        mats, dropped = score_matrices(meta, algo)
        print(f"\n=== {algo}: {len(mats)} complete arms x {len(SEEDS)} seeds x "
              f"{len(mf.GAME_ORDER)} games, control = {mf.ARM[baseline][0]}")
        for nm, n in dropped:
            print(f"  EXCLUDED {nm}: {n} missing run(s) -- rliable needs a full matrix")
        if not mats:
            print(f"  skipped {algo}: no complete arm")
            continue

        # Bootstrap once, draw twice: the estimates do not depend on page width,
        # and sharing them guarantees the deck and the thesis quote identical
        # numbers.
        print("bootstrapping ...", flush=True)
        e_agg = est_aggregate(mats, a.reps)
        e_prof = est_profile(mats, max(a.reps // 5, 2000))
        e_poi = est_poi(mats, a.reps, baseline)

        for name in a.targets:
            tgt = TARGETS[name]
            os.makedirs(tgt["dir"], exist_ok=True)
            _style(tgt["fs"])
            print(f"[{name}]")
            fig_aggregate(mats, tgt, e_agg, suffix, algo)
            fig_profile(mats, tgt, e_prof, suffix, algo)
            fig_poi(mats, tgt, e_poi, baseline, suffix, algo)

        dump_numbers(mats, e_agg, e_poi, dropped, a.reps, algo, baseline, suffix)

    # RQ4 is a NEC-only, paired comparison and does not belong in the loop
    # above: its arms are not in ARM_ORDER and its control is per-arm rather
    # than shared, so it gets its own matrices and its own figure.
    if a.rq4 and "NEC" in a.algos:
        arms = [e for pair in mf.FROZEN_PAIRS for e in pair]
        mats4, dropped4 = score_matrices_for(meta, "NEC", arms)
        print(f"\n=== RQ4: {len(mats4)} complete arms "
              f"({len(mf.FROZEN_PAIRS)} frozen/fine-tuned pairs)")
        for nm, n in dropped4:
            print(f"  EXCLUDED {nm}: {n} missing run(s) -- rliable needs a full matrix")
        print("bootstrapping ...", flush=True)
        e_rq4 = est_poi_paired(mats4, a.reps, mf.FROZEN_PAIRS)
        for name in a.targets:
            tgt = TARGETS[name]
            os.makedirs(tgt["dir"], exist_ok=True)
            _style(tgt["fs"])
            print(f"[{name}]")
            fig_rq4(mats4, tgt, e_rq4, mf.FROZEN_PAIRS)
        dump_rq4_numbers(mats4, e_rq4, mf.FROZEN_PAIRS, dropped4, a.reps,
                         TARGETS[a.targets[0]])

    if a.frozen_vs_base and "NEC" in a.algos:
        frozen_vs_base(meta, a.reps, a.targets)


if __name__ == "__main__":
    main()
