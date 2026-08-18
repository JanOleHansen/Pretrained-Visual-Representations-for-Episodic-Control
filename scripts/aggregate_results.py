#!/usr/bin/env python
"""Turn a set of finished W&B runs into the paper's cross-game results.

The one post-processing step the codebase cannot do run-by-run: aggregate
human-normalised scores *across games* into the headline Atari 100k numbers
(mean / median / IQM) with confidence intervals, plus the per-encoder cost
table. Everything else (``eval/hns``, ``eval/value_return_corr``,
``sys/gpu_mem_peak_gb``, ``time/elapsed_min``) is already logged per run; this
script only pools them.

It is deliberately robust to runs produced *before* ``eval/hns`` existed: when a
run has no ``eval/hns`` in its summary it recomputes one from the final
``eval/return_mean`` and the game's baseline (``src/utils/atari_scores.py``), so
your existing runs need no re-running.

Confidence intervals use a **stratified bootstrap** (resample seeds within each
game, per Agarwal et al. 2021, "Deep RL at the Edge of the Statistical
Precipice") implemented in NumPy — no ``rliable`` dependency.

Usage
-----
    python scripts/aggregate_results.py --entity <you> --project <proj>
    python scripts/aggregate_results.py --project <proj> --plot --curves

Outputs (under ``--output-dir``, default ``results/``):
    per_run.csv          one row per run: algorithm, encoder, game, seed, hns, ...
    aggregate.csv        one row per (algorithm, encoder): mean/median/iqm + 95% CI
    aggregate.tex        the same as a LaTeX table, ready to \\input
    cost.csv             per (algorithm, encoder): wall-clock and peak GPU memory
    curves.csv           (with --curves) eval/hns vs step, mean+sem over seeds
    aggregate.png        (with --plot) IQM per encoder with 95% CI bars
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import the shared baseline table so HNS here is computed with the exact same
# constants the trainer uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.atari_scores import human_normalized_score, resolve_game  # noqa: E402


# --------------------------------------------------------------------------- #
# Pulling runs
# --------------------------------------------------------------------------- #
def _nested(config: dict, *path: str, default=None):
    """Read ``config['a']['b']`` (nested) or ``config['a/b']`` (flattened)."""
    node = config
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            node = None
            break
    if node is not None:
        return node
    return config.get("/".join(path), default)


class RunRecord:
    """The handful of fields the aggregation needs from one W&B run."""

    __slots__ = ("algorithm", "encoder", "game", "seed", "hns", "ret",
                 "corr", "elapsed_min", "gpu_gb", "name")

    def __init__(self, run) -> None:
        cfg = dict(run.config)
        summary = dict(run.summary)

        self.name = run.name
        self.algorithm = _nested(cfg, "run", "algorithm") or _nested(cfg, "algorithm") or "?"
        self.encoder = _nested(cfg, "run", "encoder") or "?"
        raw_game = _nested(cfg, "run", "game") or _nested(cfg, "game") \
            or _nested(cfg, "environment", "name")
        self.game = resolve_game(raw_game if raw_game is None else str(raw_game)) or str(raw_game)
        seed = _nested(cfg, "run", "seed")
        if seed is None:
            seed = _nested(cfg, "trainer", "seed")
        self.seed = int(seed) if seed is not None else -1

        self.ret = _finite(summary.get("eval/return_mean"))
        # Prefer the logged HNS; fall back to recomputing it (older runs).
        hns = _finite(summary.get("eval/hns"))
        if hns is None and self.ret is not None:
            hns = human_normalized_score(self.game, self.ret)
        self.hns = hns

        self.corr = _finite(summary.get("eval/value_return_corr"))
        self.elapsed_min = _finite(summary.get("time/elapsed_min"))
        self.gpu_gb = _finite(summary.get("sys/gpu_mem_peak_gb"))


def _finite(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def fetch_runs(entity: str | None, project: str, wandb_filter: str | None) -> list[RunRecord]:
    import json
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = json.loads(wandb_filter) if wandb_filter else None
    records = []
    for run in api.runs(path, filters=filters):
        if run.state == "running":
            continue
        rec = RunRecord(run)
        if rec.hns is not None:
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Aggregation statistics
# --------------------------------------------------------------------------- #
def _iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the values within [25th, 75th] percentile."""
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [25, 75])
    middle = values[(values >= lo) & (values <= hi)]
    return float(middle.mean()) if middle.size else float(values.mean())


def _point_estimates(by_game: dict[str, list[float]]) -> dict[str, float]:
    """mean / median (over per-game means) and IQM (over pooled runs)."""
    per_game_mean = np.array([np.mean(v) for v in by_game.values()])
    pooled = np.array([x for v in by_game.values() for x in v])
    return {
        "mean": float(per_game_mean.mean()),
        "median": float(np.median(per_game_mean)),
        "iqm": _iqm(pooled),
    }


def _stratified_bootstrap(
    by_game: dict[str, list[float]], n_boot: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """95% CIs by resampling seeds *within* each game with replacement."""
    games = list(by_game)
    arrays = {g: np.asarray(by_game[g], dtype=float) for g in games}
    stats = {"mean": [], "median": [], "iqm": []}
    for _ in range(n_boot):
        resampled = {
            g: arrays[g][rng.integers(0, arrays[g].size, arrays[g].size)]
            for g in games
        }
        est = _point_estimates(resampled)
        for k in stats:
            stats[k].append(est[k])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in stats.items()
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _arm_key(rec: RunRecord) -> tuple[str, str]:
    return (rec.algorithm, rec.encoder)


def write_per_run(records: list[RunRecord], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "encoder", "game", "seed", "hns",
                    "return_mean", "value_return_corr", "elapsed_min", "gpu_mem_peak_gb", "run"])
        for r in sorted(records, key=lambda r: (r.algorithm, r.encoder, r.game, r.seed)):
            w.writerow([r.algorithm, r.encoder, r.game, r.seed,
                        _fmt(r.hns), _fmt(r.ret), _fmt(r.corr),
                        _fmt(r.elapsed_min), _fmt(r.gpu_gb), r.name])


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.6g}"


def aggregate(records: list[RunRecord], n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    arms: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    corr_arms: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        arms[_arm_key(r)][r.game].append(r.hns)
        if r.corr is not None:
            corr_arms[_arm_key(r)].append(r.corr)

    rows = []
    for key in sorted(arms):
        by_game = arms[key]
        point = _point_estimates(by_game)
        ci = _stratified_bootstrap(by_game, n_boot, rng)
        n_runs = sum(len(v) for v in by_game.values())
        corr = corr_arms.get(key, [])
        rows.append({
            "algorithm": key[0], "encoder": key[1],
            "games": len(by_game), "runs": n_runs,
            "mean": point["mean"], "mean_lo": ci["mean"][0], "mean_hi": ci["mean"][1],
            "median": point["median"], "median_lo": ci["median"][0], "median_hi": ci["median"][1],
            "iqm": point["iqm"], "iqm_lo": ci["iqm"][0], "iqm_hi": ci["iqm"][1],
            "corr_mean": float(np.mean(corr)) if corr else float("nan"),
        })
    return rows


def write_aggregate_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def write_aggregate_tex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Algorithm & Encoder & Mean HNS & Median HNS & IQM HNS & kNN corr \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['algorithm']} & {r['encoder']} & "
            f"{r['mean']:.3f} [{r['mean_lo']:.3f}, {r['mean_hi']:.3f}] & "
            f"{r['median']:.3f} [{r['median_lo']:.3f}, {r['median_hi']:.3f}] & "
            f"{r['iqm']:.3f} [{r['iqm_lo']:.3f}, {r['iqm_hi']:.3f}] & "
            + (f"{r['corr_mean']:.3f}" if np.isfinite(r['corr_mean']) else "--")
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def write_cost_csv(records: list[RunRecord], path: Path) -> None:
    arms: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"elapsed_min": [], "gpu_gb": []}
    )
    for r in records:
        if r.elapsed_min is not None:
            arms[_arm_key(r)]["elapsed_min"].append(r.elapsed_min)
        if r.gpu_gb is not None:
            arms[_arm_key(r)]["gpu_gb"].append(r.gpu_gb)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "encoder", "elapsed_min_mean", "gpu_mem_peak_gb_mean", "runs"])
        for key in sorted(arms):
            e, g = arms[key]["elapsed_min"], arms[key]["gpu_gb"]
            w.writerow([key[0], key[1],
                        _fmt(float(np.mean(e)) if e else None),
                        _fmt(float(np.mean(g)) if g else None),
                        max(len(e), len(g))])


def write_curves(entity, project, wandb_filter, path: Path) -> None:
    """Per (algorithm, encoder, game): eval/hns vs step, mean+sem over seeds."""
    import json
    import wandb

    api = wandb.Api()
    p = f"{entity}/{project}" if entity else project
    filters = json.loads(wandb_filter) if wandb_filter else None
    # step -> arm -> list of hns across seeds
    curve: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in api.runs(p, filters=filters):
        if run.state == "running":
            continue
        rec = RunRecord(run)
        ref_game = rec.game
        for row in run.scan_history(keys=["_step", "eval/hns", "eval/return_mean"]):
            step = row.get("_step")
            hns = row.get("eval/hns")
            if hns is None and row.get("eval/return_mean") is not None:
                hns = human_normalized_score(ref_game, row["eval/return_mean"])
            if step is not None and hns is not None and np.isfinite(hns):
                curve[(rec.algorithm, rec.encoder, ref_game)][int(step)].append(float(hns))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "encoder", "game", "step", "hns_mean", "hns_sem", "n_seeds"])
        for (algo, enc, game), steps in sorted(curve.items()):
            for step in sorted(steps):
                vals = np.asarray(steps[step], dtype=float)
                sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                w.writerow([algo, enc, game, step, f"{vals.mean():.6g}", f"{sem:.6g}", vals.size])


def plot_aggregate(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping --plot", file=sys.stderr)
        return
    labels = [f"{r['algorithm']}\n{r['encoder']}" for r in rows]
    iqm = [r["iqm"] for r in rows]
    lo = [r["iqm"] - r["iqm_lo"] for r in rows]
    hi = [r["iqm_hi"] - r["iqm"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4))
    ax.bar(range(len(rows)), iqm, yerr=[lo, hi], capsize=4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("IQM human-normalised score")
    ax.set_title("Aggregate over games (95% stratified-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=None, help="W&B entity (default: your login)")
    ap.add_argument("--project", required=True, help="W&B project name")
    ap.add_argument("--filter", default=None,
                    help='W&B run filter as JSON, e.g. \'{"tags": "final"}\'')
    ap.add_argument("--output-dir", default="results", type=Path)
    ap.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--curves", action="store_true", help="also export learning curves")
    ap.add_argument("--plot", action="store_true", help="also render aggregate.png")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching runs from {args.entity or '<default>'}/{args.project} ...")
    records = fetch_runs(args.entity, args.project, args.filter)
    if not records:
        print("no finished runs with a usable score found; check --project/--filter",
              file=sys.stderr)
        sys.exit(1)
    print(f"  {len(records)} runs across "
          f"{len({r.game for r in records})} games, "
          f"{len({_arm_key(r) for r in records})} (algorithm, encoder) arms")

    write_per_run(records, args.output_dir / "per_run.csv")
    rows = aggregate(records, args.bootstrap, args.seed)
    write_aggregate_csv(rows, args.output_dir / "aggregate.csv")
    write_aggregate_tex(rows, args.output_dir / "aggregate.tex")
    write_cost_csv(records, args.output_dir / "cost.csv")
    if args.curves:
        write_curves(args.entity, args.project, args.filter, args.output_dir / "curves.csv")
    if args.plot:
        plot_aggregate(rows, args.output_dir / "aggregate.png")

    print(f"\nwrote {args.output_dir}/: per_run.csv, aggregate.csv, aggregate.tex, cost.csv"
          + (", curves.csv" if args.curves else "")
          + (", aggregate.png" if args.plot else ""))
    print("\naggregate (IQM human-normalised score, 95% CI):")
    for r in rows:
        print(f"  {r['algorithm']:>5} {r['encoder']:<10} "
              f"IQM={r['iqm']:.3f} [{r['iqm_lo']:.3f}, {r['iqm_hi']:.3f}]  "
              f"median={r['median']:.3f}  mean={r['mean']:.3f}  "
              f"(n={r['runs']} over {r['games']} games)")


if __name__ == "__main__":
    main()
