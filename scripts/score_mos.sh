#!/bin/bash
# ============================================================
# Stage 3a: neural MOS scoring (DNSMOS Pro + UTMOS) over wav folders (SLURM, 1 GPU).
# Writes mos_scores.txt inside each wav folder (per-sample + mean).
#
# Submit:   WAV_DIRS="/path/a /path/b" sbatch scripts/score_mos.sh
# Override: DNSMOS_VARIANT=NISQA OVERWRITE=true WAV_DIRS="..." sbatch scripts/score_mos.sh
# Baselines (MOS ceiling/floor over the full test split, no WAV_DIRS needed):
#   ROLE=healthy OUT_DIR=/some/dir sbatch scripts/score_mos.sh     # ceiling (healthy target)
#   ROLE=source  OUT_DIR=/some/dir sbatch scripts/score_mos.sh     # floor  (pathological input)
# ============================================================
#SBATCH --job-name=melvc_mos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/melvc_mos_%j.log
#SBATCH --error=/tmp/melvc_mos_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_mos_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

# Space-separated list of wav folders to score. Default = the first vocoded folder.
WAV_DIRS=${WAV_DIRS:-/data/fs201163/eo49197/DumpedMels/classic_ogm_ckpt500/classic_ogm_ckpt500__tf__av__p2h_ck500_gbestmel}
DNSMOS_VARIANT=${DNSMOS_VARIANT:-BVCC}
ROLE=${ROLE:-pred}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"

if [ "$ROLE" = "pred" ]; then
    echo "[mos] role=pred variant=$DNSMOS_VARIANT  dirs=$WAV_DIRS"
    python scripts/score_mos.py $WAV_DIRS --dnsmos-variant "$DNSMOS_VARIANT" $OW_FLAG
else
    # Baseline (ceiling/floor): no WAV_DIRS; writes mos_<role>.txt into OUT_DIR.
    OUT_DIR=${OUT_DIR:-/data/fs201163/eo49197/DumpedMels_run3/mos_baseline}
    MANIFEST_FLAG=""; [ -n "${MANIFEST:-}" ] && MANIFEST_FLAG="--manifest $MANIFEST"
    echo "[mos] role=$ROLE variant=$DNSMOS_VARIANT  out_dir=$OUT_DIR"
    python scripts/score_mos.py --role "$ROLE" --out-dir "$OUT_DIR" \
        --dnsmos-variant "$DNSMOS_VARIANT" $MANIFEST_FLAG $OW_FLAG
fi

echo "ALL DONE"
