#!/usr/bin/env bash
# Run the whole embedding extraction on the cluster, in the order that fails
# cheapest first.  Everything it writes lands under $OUT, and only $OUT/*.npz
# needs to come home -- the probe frame sets stay here (several GB) and the
# checkpoints never move (~120 GB).
#
#   ssh amur.ins.informatik.uni-kiel.de
#   cd <this repo>
#   bash scripts/extract_all.sh
#
# Re-running is safe and cheap: every stage skips work whose output already
# exists, so an interrupted run is resumed by invoking it again.  Add
# --overwrite to extract_embeddings.py by hand if you really want a rebuild.
set -euo pipefail

RUNS="${RUNS:-/datahome/rschwinger/episodicrl/jhansen/logs/train/runs}"
OUT="${OUT:-$PWD/embeddings}"
PROBES="${PROBES:-$PWD/probe_sets}"
FRAMES="${FRAMES:-10000}"
PY="${PY:-python}"
GAMES=(MsPacman Qbert Frostbite)

echo "runs   : $RUNS"
echo "probes : $PROBES"
echo "out    : $OUT"
echo

# ---------------------------------------------------------------------------
# 0. Does the grid actually exist?  Answer this before spending an hour on it.
# ---------------------------------------------------------------------------
echo "== checkpoint coverage =="
missing=0
found=0
for d in "$RUNS"/{mfec,nec}_*_seed*; do
    [ -d "$d" ] || continue
    if [ -f "$d/checkpoints/last.pt" ]; then
        found=$((found + 1))
    else
        echo "  MISSING last.pt: $(basename "$d")"
        missing=$((missing + 1))
    fi
done
echo "  $found run(s) with last.pt, $missing without"
if [ "$found" -eq 0 ]; then
    echo "Nothing to extract -- check RUNS." >&2
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# 1. The matched probe sets.  CPU only, no GPU needed; ~10 min per game.
# ---------------------------------------------------------------------------
echo "== probe sets =="
for g in "${GAMES[@]}"; do
    if [ -f "$PROBES/probe_$g.pt" ]; then
        echo "  $g: exists, skipping"
    else
        $PY scripts/build_probe_set.py --game "$g" --frames "$FRAMES" --out "$PROBES"
    fi
done
echo

# ---------------------------------------------------------------------------
# 2. MFEC first.  Its encoders are frozen and shared across seeds, so the probe
#    cache means only a handful of ViT passes actually run -- if something is
#    wrong with the environment, this is where it surfaces in minutes rather
#    than at the end of the NEC pass.
# ---------------------------------------------------------------------------
echo "== MFEC =="
$PY scripts/extract_embeddings.py --run-root "$RUNS" --probe-root "$PROBES" \
    --out "$OUT" --only mfec_

# ---------------------------------------------------------------------------
# 3. NEC.  Every embedding network is fine-tuned by its own run, so there is
#    nothing to share and this is one forward pass per run.
# ---------------------------------------------------------------------------
echo
echo "== NEC =="
$PY scripts/extract_embeddings.py --run-root "$RUNS" --probe-root "$PROBES" \
    --out "$OUT" --only nec_

echo
echo "== done =="
du -sh "$OUT"
echo "Bring home with:"
echo "  rsync -avz --exclude _probe_cache \\"
echo "    amur.ins.informatik.uni-kiel.de:$OUT/ <local>/embeddings/"
