#!/usr/bin/env bash
# Thesis follow-up measurements that need the cluster.
#
# Everything here is either a diagnostic that needs the *training hardware*
# (cuDNN/TF32 behaviour is GPU-specific, so measuring it on a laptop answers a
# different question) or an extraction that needs the checkpoints, which never
# leave the cluster.  Nothing here trains anything.
#
#   ssh amur.ins.informatik.uni-kiel.de
#   cd <this repo>
#   bash scripts/thesis_followups.sh
#
# Total runtime: roughly 30-60 min, dominated by stage 2.
# Re-running is safe: every stage skips work whose output already exists.
#
# What comes home afterwards (small):
#   results/encoder_tolerance.txt      <- stage 1, a few KB
#   embeddings/nec_*_frozen_*.npz      <- stage 2, ~25 MB per run
#
set -euo pipefail

RUNS="${RUNS:-/datahome/rschwinger/episodicrl/jhansen/logs/train/runs}"
OUT="${OUT:-$PWD/embeddings}"
PROBES="${PROBES:-$PWD/probe_sets}"
RESULTS="${RESULTS:-$PWD/results}"
PY="${PY:-python}"

mkdir -p "$RESULTS"
echo "runs    : $RUNS"
echo "probes  : $PROBES"
echo "out     : $OUT"
echo "results : $RESULTS"
echo

# ---------------------------------------------------------------------------
# Stage 1 -- per-backbone key stability and near-exact tolerance.
#
# Thesis Section 7.3 ("Encoder-internal validity") concedes that the
# 3e-5 relative tolerance was validated on 400 Ms. Pac-Man frames with
# ResNet-18 and *assumed* to transfer to the three transformer arms.  This
# stage measures it for every arm instead of assuming it.
#
# --device cuda is mandatory: the quantity at issue is whether phi returns
# identical bits at different batch shapes on the training hardware, which is a
# property of cuDNN algorithm selection and TF32, not of the maths.
# ---------------------------------------------------------------------------
echo "== stage 1: encoder diagnostics (GPU) =="
if [ -s "$RESULTS/encoder_tolerance.txt" ]; then
    echo "  exists, skipping -- delete it to re-measure"
else
    # --frames 2000 rather than the 150 default: the thesis quotes a
    # nearest-distinct-frame distance, and that estimate is only as good as the
    # number of pairs behind it.
    $PY scripts/encoder_diagnostics.py \
        --frames 2000 \
        --device cuda \
        --resnet --clip --mae \
        --dinov2-weights "${DINOV2_WEIGHTS:-/datahome/rschwinger/models/dinov2_vits14_pretrain.pth}" \
        2>&1 | tee "$RESULTS/encoder_tolerance.txt"
    echo "  -> $RESULTS/encoder_tolerance.txt"
fi
echo

# ---------------------------------------------------------------------------
# Stage 2 -- embeddings for the FROZEN NEC arms.
#
# Section 8.3 calls re-running the Section 6 geometry analysis on a frozen and
# a fine-tuned checkpoint of the same backbone "the most direct continuation of
# this thesis", and says the runs already exist and only the analysis is
# missing.  That is true, but the frozen-NEC embeddings were never extracted --
# embeddings/ currently holds mfec_* and fine-tuned nec_* only.  This stage
# produces the missing half.
#
# If your frozen runs are named differently, fix the --only pattern; the script
# prints what it matched before doing any work.
# ---------------------------------------------------------------------------
echo "== stage 2: frozen-NEC embeddings (GPU) =="
echo "  frozen run dirs found:"
ls -d "$RUNS"/nec_*frozen*_seed* 2>/dev/null | sed 's|.*/|    |' || {
    echo "    none matched 'nec_*frozen*_seed*'"
    echo "    -> list \$RUNS and adjust the --only pattern below, then re-run"
}
echo
$PY scripts/extract_embeddings.py \
    --run-root "$RUNS" \
    --probe-root "$PROBES" \
    --out "$OUT" \
    --only 'nec_*frozen*' \
    || echo "  extraction reported a problem -- see above"
echo

echo "== done =="
echo "Copy home:"
echo "  scp <user>@amur.ins.informatik.uni-kiel.de:$RESULTS/encoder_tolerance.txt ."
echo "  scp '<user>@amur.ins.informatik.uni-kiel.de:$OUT/nec_*frozen*.npz' ./embeddings/"
