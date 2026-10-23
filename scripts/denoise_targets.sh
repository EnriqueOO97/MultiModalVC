#!/bin/bash
# ============================================================
# Denoise the TARGET wavs of a manifest (SLURM, 1 GPU).
#
# Writes '<stem>_denoise.wav' NEXT TO each original. Nothing is overwritten and
# nothing is deleted -- listen to both, then decide whether to keep or discard.
#
#   sbatch scripts/denoise_targets.sh                       # test split (default)
#   MANIFEST=/path.tsv sbatch scripts/denoise_targets.sh    # another manifest
#   LIMIT=20 sbatch scripts/denoise_targets.sh              # first 20 only, to sample
#   OVERWRITE=true sbatch scripts/denoise_targets.sh        # redo existing _denoise files
#
# Re-running is safe: files that already have a _denoise output are skipped.
# ============================================================
#SBATCH --job-name=melvc_denoise
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/melvc_denoise_%j.log
#SBATCH --error=/tmp/melvc_denoise_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_denoise_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-trim/testPATH-HE.tsv}
COL=${COL:-2}
SUFFIX=${SUFFIX:-_denoise}
LIMIT=${LIMIT:-0}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"
DRY_FLAG=""; [ "${DRY_RUN:-}" = "true" ] && DRY_FLAG="--dry-run"

echo "[denoise] manifest=$MANIFEST col=$COL suffix=$SUFFIX limit=$LIMIT overwrite=${OVERWRITE:-false}"
python scripts/denoise_targets.py \
    --manifest "$MANIFEST" --col "$COL" --suffix "$SUFFIX" --limit "$LIMIT" \
    $OW_FLAG $DRY_FLAG

echo "ALL DONE"
