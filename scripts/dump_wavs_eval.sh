#!/bin/bash
# ============================================================
# Stage 2: vocode dumped mel .npy folders -> 16 kHz mono wavs (SLURM, single GPU).
# Runs scripts/dump_wavs_eval.py over the jobs in a JSON config (BigVGAN generator +
# list of npy folders). Wavs land beside their npy folder, inside the same father.
#
# Submit:   sbatch scripts/dump_wavs_eval.sh
# Override: CONFIG=/path.json BATCH_SIZE=64 sbatch scripts/dump_wavs_eval.sh
# ONE GPU (the light vocoder); batched inference.
# ============================================================
#SBATCH --job-name=melvc_vocode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/melvc_vocode_%j.log
#SBATCH --error=/tmp/melvc_vocode_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_vocode_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

VOC_CONFIG=${CONFIG:-$ROOT/scripts/vocoder_jobs.json}
BATCH_SIZE=${BATCH_SIZE:-32}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"

echo "[vocode] config=$VOC_CONFIG  batch_size=$BATCH_SIZE  OVERWRITE=${OVERWRITE:-false}"

python scripts/dump_wavs_eval.py \
    --config "$VOC_CONFIG" \
    --batch-size "$BATCH_SIZE" \
    $OW_FLAG

echo "ALL DONE"
