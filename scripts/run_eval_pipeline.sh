#!/bin/bash
# ============================================================
# Eval pipeline wrapper (SLURM, 1 GPU):
#   Stage 1 dump mels | Stage 2 vocode | Stage 3 MOS | Stage 4 SECS | Stage 5 ASR (WER/CER).
# The spec (checkpoints, modes, vocoders, manifest, out-root) is INLINE in
# scripts/run_eval_pipeline.py -- edit there to change the run.
#
# Full pipeline:   sbatch scripts/run_eval_pipeline.sh                 # stages 1-4
# Subset:          STAGE=2,3,4 sbatch scripts/run_eval_pipeline.sh
#                  STAGE=1 sbatch scripts/run_eval_pipeline.sh
# Rescore/redump:  OVERWRITE=true sbatch scripts/run_eval_pipeline.sh
# ============================================================
#SBATCH --job-name=melvc_eval_pipe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/melvc_eval_pipe_%j.log
#SBATCH --error=/tmp/melvc_eval_pipe_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_eval_pipe_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

STAGE=${STAGE:-all}
BATCH_SIZE=${BATCH_SIZE:-32}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-48}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"

echo "[eval-pipe] stage=$STAGE batch=$BATCH_SIZE score_batch=$SCORE_BATCH_SIZE out_root=${OUT_ROOT:-<default>} modality=${MODALITY:-<default>}"
python scripts/run_eval_pipeline.py --stage "$STAGE" --batch-size "$BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" $OW_FLAG

echo "ALL DONE"
