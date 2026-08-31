#!/bin/bash
# ============================================================
# Dump GENUINE-INFERENCE predicted mels for the EVAL pipeline (SLURM, single GPU).
# Runs scripts/dump_mels_eval.py over ONE manifest for every (checkpoint, mode,
# modality) in the JSON config, writing <out_root>/<tag>__<mode>__<modality>/<uid>.npy.
#
# Submit:   sbatch scripts/dump_mels_eval.sh
# Override: CONFIG=/path.json MANIFEST=/path.tsv OUT_ROOT=/path NUM_WORKERS=12 \
#             sbatch scripts/dump_mels_eval.sh
# ONE GPU (the frozen acoustic model); CPUs drive the dataloader workers.
# ============================================================
#SBATCH --job-name=melvc_dumpeval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/melvc_dumpeval_%j.log
#SBATCH --error=/tmp/melvc_dumpeval_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

# Log to SHARED storage so it is tail-able live from the login node.
mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_dumpeval_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
# Whisper/Q-Former architecture is built from the local HF cache; weights come from
# the checkpoint. Force offline so a slow/unreachable HF host can't stall the job.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Dedicated var names (not generic DATA) so a var exported in the login shell can't
# leak in via SLURM and override these.
EVAL_CONFIG=${CONFIG:-$ROOT/scripts/eval_checkpoints.json}
EVAL_MANIFEST=${MANIFEST:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT/testPATH-HE.tsv}
EVAL_OUT_ROOT=${OUT_ROOT:-/data/fs201163/eo49197/DumpedMels}
NUM_WORKERS=${NUM_WORKERS:-12}

# OVERWRITE=true -> re-dump existing .npy (default: SKIP already-written mels).
OW_FLAG=""; [ "${OVERWRITE:-}" = "true" ] && OW_FLAG="--overwrite"

echo "[dump-eval] config=$EVAL_CONFIG"
echo "[dump-eval] manifest=$EVAL_MANIFEST"
echo "[dump-eval] out_root=$EVAL_OUT_ROOT  workers=$NUM_WORKERS  OVERWRITE=${OVERWRITE:-false}"

python scripts/dump_mels_eval.py \
    --config "$EVAL_CONFIG" \
    --manifest "$EVAL_MANIFEST" \
    --out-root "$EVAL_OUT_ROOT" \
    --num-workers "$NUM_WORKERS" \
    $OW_FLAG

echo "ALL DONE"
