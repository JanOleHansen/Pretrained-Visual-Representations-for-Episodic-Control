#!/usr/bin/env python3
"""
Build every results figure and table for the thesis from the cached W&B runs.

    python make_thesis_figures.py --refresh      # re-pull from W&B, then plot
    python make_thesis_figures.py                # plot from the local cache
    python make_thesis_figures.py --games MsPacman Qbert Frostbite

Output: figures/*.pdf and figures/*.tex, sized for this document (single-column
`article`, \textwidth = 430pt = 5.95in) -- NOT the defense deck, whose figures
are laid out for a 16:9 frame and come out visibly oversized here.

Data loading, the TF32 validity filter and the arm/color vocabulary are
imported from the deck's make_figures.py so there is exactly one definition of
"which runs count".  Do not fork them.
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared run loader (make_figures.py) sits next to this script in
# analysis/; "DECK" is kept as the name for where it lives.
DECK = HERE
sys.path.insert(0, DECK)
import make_figures as mf          # noqa: E402  (path must be set first)

FIGDIR = os.path.join(HERE, "figures")

# Display names for the games.  The W&B keys stay as they are; only the
# printed labels change, so that figures and tables match the prose
# ("Ms. Pac-Man", "Q*bert") instead of showing the raw run keys.
GAME_LABEL = {"MsPacman": "Ms. Pac-Man", "Qbert": "Q*bert",
              "Frostbite": "Frostbite"}
def glabel(g):
    return GAME_LABEL.get(g, g)
CACHE = os.environ.get("PVR_CACHE", os.path.join(HERE, "data", "runs"))

TEXTWIDTH_IN = 430.00462 / 72.27   # \textwidth from main.log, pt -> inch
ARM, COL, REF = mf.ARM, mf.COL, mf.REF
ARM_ORDER = mf.ARM_ORDER
MIN_SEEDS = mf.MIN_SEEDS

# 10pt serif against the document's 12pt body: figure text should read a shade
# smaller than surrounding prose, not larger, once the PDF is placed at
# width=\linewidth (i.e. at 1:1, since the figures are authored at \textwidth).
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
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


def hns(vals, game):
    """Human-normalized score in %, per Wang et al. (2016) conventions."""
    rnd, hum = REF[game]
    return (vals - rnd) / (hum - rnd) * 100.0


def cells(meta, algo, games):
    """Rows for `algo` restricted to `games`, keeping only thick-enough cells."""
    s = meta[(meta.algo == algo) & (meta.game.isin(games))]
    n = s.groupby(["game", "encoder"]).size()
    keep = {k for k, v in n.items() if v >= MIN_SEEDS}
    return s[[(g, e) in keep for g, e in zip(s.game, s.encoder)]]


def agg(s, col, enc, game, transform=None):
    """mean, standard error and n for one (encoder, game) cell."""
    v = s[(s.encoder == enc) & (s.game == game)][col].dropna()
    if not len(v):
        return np.nan, 0.0, 0
    if transform is not None:
        v = transform(v, game)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), se, len(v)


# ------------------------------------------------------------------ figures
# LAYOUT: panels are stacked vertically, each spanning the full \textwidth,
# rather than sitting side by side.  At 430pt a two-up panel is ~2.9in wide and
# six encoder arms crowd into it; one panel per row doubles the horizontal
# resolution, which is where the curves actually separate.  \textheight is
# 556pt (7.7in), so a two-row stack must stay under ~5.8in to leave room for
# the caption on the same page.
PANEL_H = 2.45          # inch, per stacked panel


def panel_h(n):
    """Per-panel height that keeps an n-panel stack and its caption on one page.

    \textheight is 7.70in and a five-line caption costs roughly 1.1in, so the
    figure itself has to stay under ~6.45in.  Two games fit at the full
    PANEL_H; three do not, and LaTeX answers an oversized float by pushing it
    to a page of its own away from the text that refers to it.
    """
    return min(PANEL_H, 6.45 / max(n, 1))


def fig_curves(cv, meta, algo, games, fname, title=None):
    """Learning curves, one full-width panel per game, mean +/- s.e. over seeds."""
    games = [g for g in games
             if sum(1 for a, gm, e in cv
                    if a == algo and gm == g and cv[(a, gm, e)][1].shape[0] >= MIN_SEEDS) >= 2]
    if not games:
        print(f"  skipped {fname}: no game has {MIN_SEEDS}+ seeds on 2+ arms")
        return
    # One ROW of panels, not a column.  Stacked full-width panels cost a whole
    # page and repeat the legend once per game; side by side they cost a third
    # of that and the legend is drawn once for the figure.  The y-axes stay
    # independent -- the three games differ by a factor of four in return, so a
    # shared y would flatten Frostbite into the axis -- and sharex is dropped
    # with them because each panel now carries its own visible x-axis.
    fig, axes = plt.subplots(1, len(games), squeeze=False,
                             figsize=(TEXTWIDTH_IN, 2.15))
    for ax, gm in zip(axes[0], games):
        for enc in ARM_ORDER:
            if (algo, gm, enc) not in cv:
                continue
            steps, mat = cv[(algo, gm, enc)]
            if mat.shape[0] < MIN_SEEDS:
                continue
            lab, col, ls = ARM[enc]
            lab = f"{lab} (n={mat.shape[0]})" if mat.shape[0] < 5 else lab
            mf.band(ax, steps, mat, lab, col, ls)
        rnd = REF[gm][0]
        ax.axhline(rnd, color="k", lw=0.7, ls=":", alpha=0.6, zorder=1)
        ax.annotate("random play", (ax.get_xlim()[0], rnd), fontsize=6.5,
                    va="bottom", ha="left", alpha=0.6)
        ax.set_title(glabel(gm), loc="left", fontweight="bold")
        # No upper headroom any more: it existed to keep an in-panel legend off
        # the fastest arm, and the legend now sits under the figure.
        ax.set_xlabel("agent steps")
        ax.xaxis.set_major_formatter(FuncFormatter(_ksteps))
    axes[0][0].set_ylabel("evaluation return")

    # One legend for the figure, under the row.  Taken from the first panel:
    # every panel plots the same arms in ARM_ORDER, so the handles are
    # identical and the repetition bought nothing.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=len(l),
               bbox_to_anchor=(0.5, 0.045), handlelength=1.7,
               columnspacing=1.3, borderpad=0.2)
    if title:
        fig.suptitle(title, y=0.995, fontsize=10)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def _grouped_bars(ax, s, encs, games, col, transform=None, ylabel=None):
    """Grouped bars: one cluster per GAME, one colored bar per encoder.

    The games form the x axis and the encoders are the compared series, because
    the comparison this thesis makes is between encoders *within* a game.  The
    earlier arrangement (a cluster per encoder, games distinguished by grey
    fill) put the quantity of interest across clusters and coded it only by
    lightness, which also made the figure read as monochrome beside the colored
    learning curves.  Encoders keep the same color vocabulary they carry in
    every other figure, so an arm is the same color throughout the document.

    The encoder axis stays FIXED within a cluster: an arm absent from `s`
    leaves a gap rather than shifting the others, so stacked panels line up and
    an arm can be compared straight down the page.
    """
    w = 0.8 / max(len(encs), 1)
    for j, e in enumerate(encs):
        vals, errs = [], []
        for gm in games:
            m, se, _ = agg(s, col, e, gm, transform)
            vals.append(m)
            errs.append(se)
        x = np.arange(len(games)) + j * w - 0.4 + w / 2
        ax.bar(x, vals, w * 0.92, yerr=errs, capsize=2, label=ARM[e][0],
               color=ARM[e][1], edgecolor="k", lw=0.4,
               error_kw=dict(lw=0.7))
    ax.set_xticks(np.arange(len(games)))
    ax.set_xticklabels([glabel(g) for g in games])
    if ylabel:
        ax.set_ylabel(ylabel)


def _arm_legend(fig, encs, y=0.055, ncol=None):
    """One shared encoder key under the figure, in the document's arm colors."""
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=7,
                                   color=ARM[e][1], mec="k", mew=0.4,
                                   label=ARM[e][0]) for e in encs],
               frameon=False, fontsize=7.5, loc="upper center",
               ncol=ncol or min(len(encs), 6), bbox_to_anchor=(0.5, y),
               handletextpad=0.35, columnspacing=1.1)


def fig_hns(meta, games, fname):
    """Thesis 4.3.1 -- human-normalized final score per arm, MFEC over NEC."""
    panels = [a for a in ("MFEC", "NEC") if len(cells(meta, a, games))]
    if not panels:
        print(f"  skipped {fname}: no cell has {MIN_SEEDS}+ seeds")
        return
    # One arm axis shared by both panels (their union, in the canonical order),
    # so DINOv2 sits directly above DINOv2 and the agents can be read as a
    # column.  sharey for the same reason it was shared side by side: on
    # independent axes MFEC's ~50% and NEC's ~11% each fill their own panel and
    # the figure reads as "the two agents perform about the same".
    shown = set()
    for a in panels:
        shown |= set(cells(meta, a, games).encoder)
    encs = [e for e in ARM_ORDER if e in shown]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, sharey=True,
                             squeeze=False,
                             figsize=(TEXTWIDTH_IN, 2.25 * len(panels)))
    for ax, algo in zip(axes[:, 0], panels):
        s = cells(meta, algo, games)
        gms = [g for g in games if g in set(s.game)]
        _grouped_bars(ax, s, encs, gms, "sum::eval/return_mean", transform=hns)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(algo, loc="left", fontweight="bold")
        ax.set_ylabel("human-normalized\nscore (%)")
    top = max(ax.get_ylim()[1] for ax in axes[:, 0])
    axes[0, 0].set_ylim(0, top * 1.16)
    # The encoder key goes under the figure rather than inside the top panel:
    # with the encoders now the bar series there are six entries, which no
    # longer fit beside the bars without covering them.
    _arm_legend(fig, encs, y=0.055)
    fig.tight_layout(h_pad=1.0, rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_retrieval(meta, games, fname):
    """Thesis 4.3.2 -- does the embedding support meaningful retrieval?

    Top: exact-match hit rate (MFEC only -- NEC's DND has no exact path).
    Bottom: hit rate against score, one point per (encoder, game).
    """
    s = cells(meta, "MFEC", games)
    if not len(s):
        print(f"  skipped {fname}: no MFEC cell has {MIN_SEEDS}+ seeds")
        return
    encs = [e for e in ARM_ORDER if e in set(s.encoder)]
    gms = [g for g in games if g in set(s.game)]
    hit = "sum::eval/memory_hit_rate"

    # Side by side rather than stacked: the two panels answer one question in
    # two steps and the stacked version cost most of a page.  The game key stays
    # inside the bar panel; the encoder key moves under the figure, where it
    # serves both panels instead of being redrawn in the scatter.
    fig, axes = plt.subplots(1, 2, squeeze=False, figsize=(TEXTWIDTH_IN, 2.5))
    ax = axes[0, 0]
    _grouped_bars(ax, s, encs, gms, hit, ylabel="memory hit rate")
    ax.set_title("Neighbor retrieval (MFEC)", loc="left", fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    # No in-panel legend any more: with the encoders carrying the bar colors,
    # the shared key under the figure already names them for both panels.

    ax = axes[0, 1]
    mk = {"MsPacman": "o", "Qbert": "s", "Frostbite": "^"}
    for e in encs:
        for gm in gms:
            x, _, n = agg(s, hit, e, gm)
            y, _, _ = agg(s, "sum::eval/return_mean", e, gm, transform=hns)
            if n == 0 or np.isnan(x) or np.isnan(y):
                continue
            ax.scatter(x, y, s=42, color=ARM[e][1], edgecolor="k", lw=0.4,
                       marker=mk[gm], zorder=3)
    ax.set_xlabel("memory hit rate")
    ax.set_ylabel("human-normalized score (%)")
    ax.set_title("Retrieval vs. performance (MFEC)", loc="left",
                 fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    # Marker shape carries the game, fill color the encoder -- the same color
    # vocabulary the learning-curve figure uses.  BOTH keys are needed and both
    # move under the figure: the bar panel's legend distinguishes the games by
    # bar shade, which says nothing about the scatter's marker shapes.
    fig.legend(handles=[plt.Line2D([], [], marker=m, ls="", color="0.35",
                                   mec="k", mew=0.4, label=glabel(g))
                        for g, m in mk.items() if g in gms]
                       + [plt.Line2D([], [], marker="o", ls="", color=ARM[e][1],
                                     mec="k", mew=0.4, label=ARM[e][0])
                          for e in encs],
               frameon=False, fontsize=7.5, loc="upper center",
               ncol=(len(gms) + len(encs) + 1) // 2, bbox_to_anchor=(0.5, 0.10),
               handletextpad=0.35, columnspacing=1.0)
    fig.tight_layout(w_pad=1.6, rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


def _agg_curve(cv, algo, encoder, games, grid):
    """One aggregated learning curve: mean and s.e. over seeds of the
    per-seed mean human-normalized score across `games`.

    Averaging is done per seed and then across seeds, not pooled: a seed is the
    unit of variation here, so the band is the spread of whole experiments
    rather than of individual (seed, game) cells.  Every arm is interpolated
    onto one shared step grid so the five curves are read off the same x axis.
    """
    per_game = []
    for gm in games:
        key = (algo, gm, encoder)
        if key not in cv:
            return None
        steps, mat = cv[key]
        if len(steps) < 2:
            return None
        h = hns(mat, gm)                      # (seeds, steps), elementwise
        per_game.append(np.vstack([np.interp(grid, steps, h[i])
                                   for i in range(h.shape[0])]))
    n = min(p.shape[0] for p in per_game)
    stack = np.stack([p[:n] for p in per_game], axis=0)   # (games, seeds, grid)
    per_seed = stack.mean(axis=0)                          # (seeds, grid)
    se = (per_seed.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(grid)
    return per_seed.mean(axis=0), se


#: The five arms the summary figure carries, in legend order.
#: (algo, encoder, display label, linestyle).  The selection rule is stated in
#: the caption and is not free: the two baselines are each algorithm's own
#: control, and the two "best" arms are the ones the bootstrap of
#: Section 5.3.2/5.3.4 ranks highest on the IQM inside their own grid -- MAE in
#: MFEC, frozen ResNet-18 among the frozen NEC arms -- with fine-tuned DINOv2
#: added because it is the only fine-tuned arm whose pooled interval clears its
#: control, i.e. the arm RQ4 resolves.
SUMMARY_ARMS = [
    ("MFEC", "random_projection",     "MFEC  RP (gray) \u2013 baseline",      (0, (1, 1.4))),
    ("MFEC", "mae",                   "MFEC  MAE \u2013 best frozen key",     "-"),
    ("NEC",  "none",                  "NEC  ConvNet \u2013 baseline",         (0, (5, 2))),
    ("NEC",  "resnet_frozen",         "NEC  ResNet (frozen) \u2013 best frozen key", (0, (4, 2))),
    ("NEC",  "dinov2",                "NEC  DINOv2 (fine-tuned)",         "-"),
]


def fig_summary(cv, games, fname):
    """The one-figure summary of the thesis: aggregated learning curves.

    Everything the three grids establish, on one axis -- both baselines, the
    best frozen key under each algorithm, and what fine-tuning adds.  Scores
    are human-normalized before averaging, which is what makes three games of
    very different raw scales addable at all.
    """
    grid = np.linspace(0, 100_000, 51)
    drawn = []
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.1))
    for algo, enc, lab, ls in SUMMARY_ARMS:
        r = _agg_curve(cv, algo, enc, games, grid)
        if r is None:
            print(f"  summary: no complete curve for {algo}/{enc} -- skipped")
            continue
        m, se = r
        c = ARM[enc][1]
        ax.plot(grid, m, color=c, ls=ls, lw=1.8, label=lab, zorder=3)
        ax.fill_between(grid, m - se, m + se, color=c, alpha=0.13, lw=0, zorder=2)
        drawn.append(lab)
    if not drawn:
        print(f"  skipped {fname}: no arm had a complete curve")
        plt.close(fig)
        return
    ax.axhline(0, color="k", lw=0.7)
    ax.xaxis.set_major_formatter(FuncFormatter(_ksteps))
    ax.set_xlabel("environment steps")
    ax.set_ylabel("human-normalized score (%)\nmean over " f"{len(games)} games")
    ax.set_xlim(0, 100_000)
    ax.legend(frameon=False, loc="upper left", handlelength=2.4,
              labelspacing=0.3, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print("  wrote", fname)


# -------------------------------------------------------------------- tables
def _tex(lines, fname):
    with open(os.path.join(FIGDIR, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", fname)


def table_results(meta, games, fname):
    """Thesis 4.3 -- final evaluation return, mean +/- s.e. over seeds.

    No \\multirow: main.tex does not load that package.  The agent column is
    printed once per block and the blocks are separated by \\midrule.
    """
    col = "sum::eval/return_mean"
    spec = "ll" + "r" * len(games)
    lines = [r"% auto-generated by make_thesis_figures.py -- do not edit by hand",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             r"Agent & Encoder & " + " & ".join(glabel(g) for g in games) + r" \\",
             r"\midrule"]
    for algo in ("MFEC", "NEC"):
        s = meta[(meta.algo == algo) & (meta.game.isin(games))]
        if not len(s):
            continue
        encs = [e for e in ARM_ORDER if e in set(s.encoder)]
        # Best and second best are marked WITHIN an agent's block, never across
        # the two: Section 5.3.4 and the limitations both state that the
        # MFEC--NEC comparison is uncontrolled, so a mark spanning both blocks
        # would assert exactly the ranking the thesis declines to make.
        best = {}
        for gm in games:
            ranked = sorted(
                ((agg(s, col, e, gm)[0], e) for e in encs
                 if agg(s, col, e, gm)[2] > 0),
                key=lambda t: t[0], reverse=True)
            best[gm] = [e for _, e in ranked[:2]]
        for i, e in enumerate(encs):
            row = []
            for gm in games:
                m, se, n = agg(s, col, e, gm)
                if n == 0:
                    row.append("--")
                    continue
                star = "" if n >= 5 else r"\textsuperscript{*}"
                rank = best[gm].index(e) if e in best[gm] else None
                if rank == 0:
                    cell = rf"$\mathbf{{{m:.0f}}} \pm {se:.0f}$"
                elif rank == 1:
                    cell = rf"$\underline{{{m:.0f}}} \pm {se:.0f}$"
                else:
                    cell = rf"${m:.0f} \pm {se:.0f}$"
                row.append(cell + star)
            head = algo if i == 0 else ""
            lines.append(f"{head} & {ARM[e][0]} & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    _tex(lines, fname)


def table_cost(meta, games, fname):
    """Thesis 4.3.3 -- computational cost of the encoder.

    Key width d and the episodic memory it implies are analytic.  Peak GPU
    memory is measured.  Wall-clock is deliberately NOT reported: the runs
    shared GPUs, so elapsed time reflects device contention as much as the arm.
    """
    s = meta[(meta.algo == "MFEC") & (meta.game.isin(games))]
    encs = [e for e in ARM_ORDER if e in set(s.encoder) and e in mf.ENCODER_SPEC]
    buf = 100_000
    lines = [r"% auto-generated by make_thesis_figures.py -- do not edit by hand",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Arm & Backbone & $d$ & Keys & Peak GPU \\", r"\midrule"]
    for e in encs:
        bb, d, _ = mf.ENCODER_SPEC[e]
        keys_mb = buf * int(d) * 4 / 1024 ** 2
        g = s[s.encoder == e]["sum::sys/gpu_mem_peak_gb"].dropna()
        gpu = rf"{g.mean():.1f}\,GB" if len(g) else "--"
        lines.append(rf"{ARM[e][0]} & {bb} & {d} & {keys_mb:.0f}\,MB & {gpu} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex(lines, fname)


def coverage(meta, games):
    print("\nseed coverage (finished, deduplicated, TF32-valid):")
    for algo in ("MFEC", "NEC"):
        s = meta[(meta.algo == algo) & (meta.game.isin(games))]
        if not len(s):
            continue
        print(f"  {algo}")
        for e in ARM_ORDER:
            if e not in set(s.encoder):
                continue
            row = []
            for gm in games:
                n = len(s[(s.encoder == e) & (s.game == gm)])
                flag = "" if n >= 5 else ("  <-- THIN" if n < MIN_SEEDS else "  <-- thin")
                row.append(f"{gm}={n}{flag}")
            print(f"    {ARM[e][0]:<12} " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull from W&B first")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--games", nargs="+", default=list(mf.GAME_ORDER),
                    help="games to include (default: all three, as in the thesis)")
    ap.add_argument("--include-pre-fix", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    # Only this script's own outputs are cleared.  The rliable figures live in
    # the same directory but are produced by make_rliable_figures.py (separate
    # venv), and a blanket fig_*.pdf wipe silently deleted them.
    OWNED = {"fig_mfec_curves", "fig_nec_curves", "fig_hns", "fig_retrieval",
             "fig_summary", "tab_results", "tab_cost"}
    for f in os.listdir(FIGDIR):
        if f.rsplit(".", 1)[0] in OWNED and f.endswith((".pdf", ".tex")):
            os.remove(os.path.join(FIGDIR, f))
    if a.refresh:
        mf.refresh(a.cache)
    meta = mf.load(a.cache, a.include_pre_fix)
    games = [g for g in mf.GAME_ORDER if g in a.games]
    print(f"{len(meta)} valid runs; games: {', '.join(games)}")
    cv = mf.curves(a.cache, meta)
    fig_summary(cv, games, "fig_summary.pdf")
    fig_curves(cv, meta, "MFEC", games, "fig_mfec_curves.pdf")
    fig_curves(cv, meta, "NEC", games, "fig_nec_curves.pdf")
    fig_hns(meta, games, "fig_hns.pdf")
    fig_retrieval(meta, games, "fig_retrieval.pdf")
    table_results(meta, games, "tab_results.tex")
    table_cost(meta, games, "tab_cost.tex")
    coverage(meta, games)


if __name__ == "__main__":
    main()
