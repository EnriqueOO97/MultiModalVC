#!/bin/bash
# ============================================================
# Diagnostic: how much content did the QuickVC synths lose?
# Transcribes each original LRS3 wav (reference) and its QuickVC synth (hypothesis)
# and reports WER/CER between them. Read-only; writes one TSV if OUT is set.
#
#   sbatch scripts/score_synth_content.sh
#   MODEL=openai/whisper-large-v3 LIMIT=100 sbatch scripts/score_synth_content.sh
# ============================================================
#SBATCH --job-name=synth_content
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/synth_content_%j.log
#SBATCH --error=/tmp/synth_content_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"
mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/synth_content_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-Trim-Balanced/1000newLRS3.tsv}
MODEL=${MODEL:-openai/whisper-small}
LANGUAGE=${LANGUAGE:-en}
BATCH_SIZE=${BATCH_SIZE:-16}
LIMIT=${LIMIT:-0}
OUT=${OUT:-$ROOT/exp/eval-logs/synth_content_${SLURM_JOBID}.tsv}

echo "[synth-content] manifest=$MANIFEST model=$MODEL lang=$LANGUAGE limit=$LIMIT"
python scripts/score_synth_content.py \
    --manifest "$MANIFEST" --model "$MODEL" --language "$LANGUAGE" \
    --batch-size "$BATCH_SIZE" --limit "$LIMIT" --out "$OUT"

echo "ALL DONE"
