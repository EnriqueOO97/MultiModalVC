#!/bin/bash
# ============================================================
# MelVC evaluation (SLURM, single GPU).
# Runs scripts/evaluate_melvc.py over the CHECKPOINTS list inside that file,
# computing mel-domain metrics (L1, SSIM, truthful MCD, cos, LSD, RMSE, SC) on
# the validation manifest, clean + babble-0dB by default.
#
# Submit:   sbatch scripts/evaluate_melvc.sh
# Override: SNR=-5 NUM_WORKERS=12 sbatch scripts/evaluate_melvc.sh
# All four in one job:  CONDITIONS=clean,0,-5,-10 sbatch scripts/evaluate_melvc.sh
# Batched, multi-worker pipeline (same as training validation) -> full split in
# minutes per checkpoint. ONE GPU; the CPUs are for the dataloader workers.
# ============================================================
#SBATCH --job-name=melvc_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/melvc_eval_%j.log
#SBATCH --error=/tmp/melvc_eval_%j.log
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

# Mirror the training scripts: redirect to SHARED storage so the log is
# tail-able live (the #SBATCH path above is node-local private /tmp on this
# cluster, unreadable from the login node).
mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_eval_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Optional overrides (unset -> script defaults: clean + babble_0dB, full split).
# CONDITIONS runs clean + any noisy levels in ONE job, e.g. CONDITIONS=clean,0,-5,-10
ARGS=""
[ -n "$CONDITIONS" ]  && ARGS="$ARGS --conditions $CONDITIONS"
[ -n "$SNR" ]         && ARGS="$ARGS --snr $SNR"
[ -n "$SNRLIST" ]     && ARGS="$ARGS --snr-list=$SNRLIST"
[ -n "$NUM_WORKERS" ] && ARGS="$ARGS --num-workers $NUM_WORKERS"
[ -n "$MANIFEST" ]    && ARGS="$ARGS --manifest $MANIFEST"
[ -n "$OUT" ]         && ARGS="$ARGS --out $OUT"
[ -n "$OUT_PERUTT" ]  && ARGS="$ARGS --out-perutt $OUT_PERUTT"

python scripts/evaluate_melvc.py $ARGS
