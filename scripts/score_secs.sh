#!/bin/bash
# ============================================================
# Stage 4: speaker-similarity (SECS) scoring with ECAPA-TDNN (SLURM, 1 GPU).
# Writes secs_scores.txt inside each prediction wav folder (per-sample cos_target +
# cos_source, MEAN/STD/N, and per-speaker / per-condition breakdowns).
#
# Submit:   WAV_DIRS="/path/a /path/b" sbatch scripts/score_secs.sh
# Override: MANIFEST=/path.tsv OVERWRITE=true NO_SOURCE=true WAV_DIRS="..." sbatch scripts/score_secs.sh
# ============================================================
#SBATCH --job-name=melvc_secs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/melvc_secs_%j.log
#SBATCH --error=/tmp/melvc_secs_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_secs_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

WAV_DIRS=${WAV_DIRS:?set WAV_DIRS="/dir1 /dir2 ..."}
MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-trim/testPATH-HE.tsv}
BATCH_SIZE=${BATCH_SIZE:-16}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"
NS_FLAG=""; [ "${NO_SOURCE:-}" = "true" ] && NS_FLAG="--no-source"

echo "[secs] manifest=$MANIFEST  dirs=$WAV_DIRS"

python scripts/score_secs.py $WAV_DIRS \
    --manifest "$MANIFEST" \
    --batch-size "$BATCH_SIZE" \
    $OW_FLAG $NS_FLAG

echo "ALL DONE"
