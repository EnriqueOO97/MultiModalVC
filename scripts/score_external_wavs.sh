#!/bin/bash
# ============================================================
# Score an EXTERNAL model's wavs with stages 3-5 (SLURM, 1 GPU):
#   Stage 3 MOS (DNSMOS + UTMOS) | Stage 4 SECS | Stage 5 ASR (WER/CER).
#
# The external model's outputs are mapped onto this repo's uid scheme via a
# symlink tree, then the SAME scorers the main pipeline uses are invoked.
# run_eval_pipeline.py is not involved and not modified.
#
#   WAV_DIR=/path/to/model/outputs \
#   MAP=/path/to/convert_eval.txt \
#   OUT_ROOT=/data/fs201163/eo49197/DumpedMels_<model>_test \
#   TAG=<model> sbatch scripts/score_external_wavs.sh
#
# Options: STAGE=3,4,5  BASELINES=true  ASR_MODEL=openai/whisper-small
#          LANGUAGE=de  OVERWRITE=true
# ============================================================
#SBATCH --job-name=melvc_ext_score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/melvc_ext_score_%j.log
#SBATCH --error=/tmp/melvc_ext_score_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_ext_score_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export TOKENIZERS_PARALLELISM=false

: "${WAV_DIR:?set WAV_DIR=/path/to/external/wavs}"
: "${MAP:?set MAP=/path/to/mapping (title|src|tgt or name<TAB>uid)}"
: "${OUT_ROOT:?set OUT_ROOT=/path/for/results}"

TAG=${TAG:-external}
STAGE=${STAGE:-all}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-48}
ASR_MODEL=${ASR_MODEL:-openai/whisper-large-v3}
LANGUAGE=${LANGUAGE:-de}
MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-trim/testPATH-HE.tsv}
TRANSCRIPTS=${TRANSCRIPTS:-/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv}
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"
BL_FLAG=""; [ "${BASELINES:-}" = "true" ] && BL_FLAG="--baselines"

echo "[ext-score] tag=$TAG stage=$STAGE wav_dir=$WAV_DIR map=$MAP out_root=$OUT_ROOT asr=$ASR_MODEL language=$LANGUAGE baselines=${BASELINES:-false}"
python scripts/score_external_wavs.py \
    --wav-dir "$WAV_DIR" --map "$MAP" --out-root "$OUT_ROOT" --tag "$TAG" \
    --manifest "$MANIFEST" --transcripts "$TRANSCRIPTS" \
    --asr-model "$ASR_MODEL" --language "$LANGUAGE" \
    --stage "$STAGE" --score-batch-size "$SCORE_BATCH_SIZE" $BL_FLAG $OW_FLAG

echo "ALL DONE"
