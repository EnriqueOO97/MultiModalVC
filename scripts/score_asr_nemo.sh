#!/bin/bash
# ============================================================
# Stage 5 (NeMo): ASR WER/CER with an NVIDIA NeMo German model (SLURM, 1 GPU).
# Needs `nemo_toolkit[asr]` installed in torchEnv.
# role=pred  : writes asr_scores.txt inside each prediction wav folder.
# role=healthy|source : writes asr_<role>.txt into OUT_DIR (baseline over the manifest).
#
# Ceiling:   ROLE=healthy OUT_DIR=/some/dir sbatch scripts/score_asr_nemo.sh
# Preds:     WAV_DIRS="/dir1 /dir2 ..." sbatch scripts/score_asr_nemo.sh
# Override:  MODEL=nvidia/stt_de_conformer_ctc_large BATCH_SIZE=16 OVERWRITE=true ...
# ============================================================
#SBATCH --job-name=melvc_asr_nemo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=2:00:00
#SBATCH --output=/tmp/melvc_asr_nemo_%j.log
#SBATCH --error=/tmp/melvc_asr_nemo_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_asr_nemo_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

ROLE=${ROLE:-pred}
MODEL=${MODEL:-nvidia/stt_de_conformer_ctc_large}
MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-trim/testPATH-HE.tsv}
TRANSCRIPTS=${TRANSCRIPTS:-/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv}
BATCH_SIZE=${BATCH_SIZE:-16}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"

echo "[asr-nemo] role=$ROLE  model=$MODEL"

if [ "$ROLE" = "pred" ]; then
    : "${WAV_DIRS:?set WAV_DIRS=\"/dir1 /dir2 ...\" for role=pred}"
    python scripts/score_asr_nemo.py $WAV_DIRS \
        --role pred --model "$MODEL" --manifest "$MANIFEST" --transcripts "$TRANSCRIPTS" \
        --batch-size "$BATCH_SIZE" $OW_FLAG
else
    : "${OUT_DIR:?set OUT_DIR=/dir for role=$ROLE (where asr_$ROLE.txt is written)}"
    python scripts/score_asr_nemo.py \
        --role "$ROLE" --model "$MODEL" --out-dir "$OUT_DIR" --manifest "$MANIFEST" \
        --transcripts "$TRANSCRIPTS" --batch-size "$BATCH_SIZE" $OW_FLAG
fi

echo "ALL DONE"
