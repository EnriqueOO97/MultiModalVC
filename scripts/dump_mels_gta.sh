#!/bin/bash
# ============================================================
# Dump teacher-forced predicted mels for BigVGAN GTA (SLURM, single GPU).
# Runs scripts/dump_mels_gta.py over the P2H train + valid splits, writing each
# predicted mel as <healthy_target_stem>.npy next to its target wav.
#
# Submit:   sbatch scripts/dump_mels_gta.sh
# Override: CHECKPOINT=/path/to/checkpoint_best.pt NUM_WORKERS=12 sbatch scripts/dump_mels_gta.sh
# ONE GPU (the frozen acoustic model); CPUs drive the dataloader workers.
# ============================================================
#SBATCH --job-name=melvc_dumpmel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/melvc_dumpmel_%j.log
#SBATCH --error=/tmp/melvc_dumpmel_%j.log
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

# Log to SHARED storage so it is tail-able live (the #SBATCH /tmp path above is
# node-local private on this cluster, unreadable from the login node).
mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_dumpmel_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# NOTE: dedicated var names (not generic DATA) so a variable exported in the
# submitting login shell can't leak in via SLURM and override these.
P2H_DATA=${P2H_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT}
CHECKPOINT=${CHECKPOINT:-$ROOT/exp/multiModalVC-synth/melvc_p2h_fullopen_aug_reg/checkpoints/checkpoint_best.pt}
NUM_WORKERS=${NUM_WORKERS:-12}

# Opt-in flags (both default OFF -> launcher behavior unchanged unless requested):
#   TEACHER_FORCE=true -> dump at GT length (duration predictor disabled, dump-only)
#   OVERWRITE=true     -> re-dump existing .npy (needed when stale mels from another
#                         checkpoint are present; otherwise they are SKIPPED).
TF_FLAG=""; [ "${TEACHER_FORCE:-}" = "true" ] && TF_FLAG="--teacher-force-duration"
OW_FLAG=""; [ "${OVERWRITE:-}"     = "true" ] && OW_FLAG="--overwrite"
echo "[flags] TEACHER_FORCE=${TEACHER_FORCE:-false} OVERWRITE=${OVERWRITE:-false}"

for SPLIT in trainPATH-HE validPATH-HE; do
    echo "================ dumping mels for $SPLIT ================"
    python scripts/dump_mels_gta.py \
        --manifest "$P2H_DATA/$SPLIT.tsv" \
        --checkpoint "$CHECKPOINT" \
        --num-workers "$NUM_WORKERS" \
        $TF_FLAG $OW_FLAG
done

echo "ALL SPLITS DONE"
