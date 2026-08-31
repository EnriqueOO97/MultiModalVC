#!/bin/bash
# ============================================================
# MelVC (Musica/SLURM) — single self-contained launch script.
# Mel-output model (BigVGAN-format mels) + pure L1 mel loss.
# No vocoder training, no discriminator, no adversarial.
#
# Submit:   sbatch scripts/melvc-MUSICA.sh
# Override anything via env, e.g.:
#   RUN_NAME=melvc_v1 FINETUNE_DATA=/path LR=3e-5 sbatch scripts/melvc-MUSICA.sh
# ============================================================
#ASC --vanilla
#SBATCH --job-name=mms_musica
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88               # Request all CPUs for the node
#SBATCH --time=72:00:00                 # Max limit on Musica
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log
set -e

# ---------- PATHS ----------
ROOT=/data/fs201163/eo49197/MultiModalVC
FINETUNE_DATA=${FINETUNE_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/pre-trainVoxcelebYoutube}
WHISPER_PRETRAINED_PATH=${WHISPER_PRETRAINED_PATH:-$ROOT/pretrained_models/whisper-medium/checkpoint-2900}
W2V_PATH=${W2V_PATH:-$ROOT/pretrained_models/avhubert/large_vox_iter5.pt}

# Noise injection (train-only, on the INPUT audio). Default OFF (prob 0.0).
# Set NOISE_PROB>0 to enable; babble file ships in $ROOT/noise.
NOISE_WAV=${NOISE_WAV:-$ROOT/noise/babble_noise.wav}
NOISE_PROB=${NOISE_PROB:-0.0}
# Vocoder is inherited from the parent class but UNUSED in MelVC; pass a real
# checkpoint so the parent __init__ doesn't choke building it.
HIFIGAN_CKPT=${HIFIGAN_CKPT:-$ROOT/pretrained_models/hifigan/model-best.pt}

RUN_NAME=${RUN_NAME:-melvc_default}
OUT_PATH=$ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH/logs"

# Route per-node stdio to the run dir (single-node, no srun — see finetune script).
exec >> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_out_${SLURM_JOBID}.log" \
     2>> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_err_${SLURM_JOBID}.log"

eval "$(conda shell.bash hook)"
conda activate torchEnv

# ---------- distributed / GPU ----------
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROCS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NPROCS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
fi
[ "${NPROCS_PER_NODE:-0}" -ge 1 ] || NPROCS_PER_NODE=1
NGPUS=$NPROCS_PER_NODE
MASTER_ADDR=127.0.0.1
MASTER_PORT=$(expr 10000 + $(echo -n "${SLURM_JOBID:-0}" | tail -c 4))
DIST_ARGS="distributed_training.distributed_port=${MASTER_PORT} distributed_training.distributed_init_method=tcp://${MASTER_ADDR}:${MASTER_PORT} distributed_training.distributed_rank=0"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# ---------- environment ----------
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ -n "$CONDA_PREFIX" ]; then
    PYV=$(python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    for d in "$CONDA_PREFIX/lib/python${PYV}/site-packages/torch/lib" \
             "$CONDA_PREFIX/lib/python${PYV}/site-packages/nvidia/cusparse/lib"; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
    done
fi

# ---------- per-module trainable knobs ----------
AFEAT_TRAINABLE=${AFEAT_TRAINABLE:-true}
FUSION_TRAINABLE=${FUSION_TRAINABLE:-true}
QFORMER_TRAINABLE=${QFORMER_TRAINABLE:-true}
PROJ_TRAINABLE=${PROJ_TRAINABLE:-true}
CONFORMER_TRAINABLE=${CONFORMER_TRAINABLE:-true}
WHISPER_TOP_N=${WHISPER_TOP_N:-2}
WHISPER_LN_TRAINABLE=${WHISPER_LN_TRAINABLE:-true}
MEL_HEAD_TRAINABLE=${MEL_HEAD_TRAINABLE:-true}   # head-only salvage: keep true, freeze the rest

echo "====== MelVC ======"
echo "RUN_NAME=$RUN_NAME  OUT=$OUT_PATH  GPUS=$NGPUS  DATA=$FINETUNE_DATA"
echo "==================="

# ============================================================
# RESUME mode (set RESUME=true). Mirrors finetuneVC-MUSICA.sh:
#   * reads the EXACT original overrides from $OUT_PATH/.hydra/overrides.yaml
#     (so batch_size, p_modality, modality_mode, etc. are reproduced exactly —
#      no risk of a manual env-var mismatch corrupting the resume),
#   * auto-selects the highest-numbered checkpoint (no checkpoint_last.pt is
#     saved because no_last_checkpoints=true),
#   * restores model + optimizer + epoch and continues.
# ============================================================
if [ "${RESUME}" = "true" ]; then
    OVERRIDES_FILE=${OUT_PATH}/.hydra/overrides.yaml
    if [ ! -f "$OVERRIDES_FILE" ]; then
        echo "ERROR: cannot resume — $OVERRIDES_FILE not found."; exit 1
    fi
    SAVED_OVERRIDES=$(grep '^- ' "$OVERRIDES_FILE" \
        | sed 's/^- //' \
        | grep -v '^distributed_training\.' \
        | grep -v '^hydra\.run\.dir' \
        | grep -v '^checkpoint\.restore_file')
    LATEST_CKPT=$(ls -1 ${OUT_PATH}/checkpoints/checkpoint[0-9]*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: RESUME requested but no numbered checkpoint in ${OUT_PATH}/checkpoints"; exit 1
    fi
    echo "RESUME MODE: restoring from $LATEST_CKPT"
    echo "RESUME MODE: reusing saved overrides from $OVERRIDES_FILE"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-hydra-train \
        --config-dir ${ROOT}/src/conf/ \
        --config-name mms-speech-nollm-melvc.yaml \
        hydra.run.dir=${OUT_PATH} \
        $SAVED_OVERRIDES \
        checkpoint.restore_file=${LATEST_CKPT} \
        model.w2v_path=${W2V_PATH} \
        distributed_training.distributed_world_size=${NGPUS} \
        $DIST_ARGS \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
    exit $?
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-hydra-train \
    --config-dir ${ROOT}/src/conf/ \
    --config-name mms-speech-nollm-melvc.yaml \
    task._name=MMS_LLaMA_pathological_finetune \
    task.data=${FINETUNE_DATA} \
    +task.afeat_1d_conv_trainable=${AFEAT_TRAINABLE} \
    +task.fusion_trainable=${FUSION_TRAINABLE} \
    +task.qformer_trainable=${QFORMER_TRAINABLE} \
    +task.proj_trainable=${PROJ_TRAINABLE} \
    +task.conformer_trainable=${CONFORMER_TRAINABLE} \
    +task.vocoder_trainable=false \
    +task.whisper_top_n_trainable=${WHISPER_TOP_N} \
    +task.whisper_layernorm_trainable=${WHISPER_LN_TRAINABLE} \
    +task.mel_head_trainable=${MEL_HEAD_TRAINABLE} \
    +task.number_of_synths=${NUMBER_OF_SYNTHS:-0} \
    +task.whisper_pretrained_path=${WHISPER_PRETRAINED_PATH} \
    task.noise_wav=${NOISE_WAV} \
    task.noise_prob=${NOISE_PROB} \
    dataset.batch_size=${BATCH_SIZE:-24} \
    dataset.max_tokens=4000 \
    dataset.required_batch_size_multiple=1 \
    dataset.valid_subset=valid \
    dataset.validate_interval=${VALIDATE_INTERVAL:-1} \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${ROOT}/src \
    common.fp16=false \
    common.bf16=true \
    common.seed=1 \
    common.log_interval=1 \
    common.empty_cache_freq=5 \
    common.tensorboard_logdir=${OUT_PATH}/tensorboard \
    checkpoint.save_dir=${OUT_PATH}/checkpoints \
    checkpoint.restore_file=${RESTORE_FILE:-checkpoint_last.pt} \
    ${FINETUNE_FROM_MODEL:+checkpoint.finetune_from_model=$FINETUNE_FROM_MODEL} \
    checkpoint.no_last_checkpoints=true \
    checkpoint.best_checkpoint_metric=loss \
    checkpoint.maximize_best_checkpoint_metric=false \
    model.modality_fuse=concat \
    +model.transconv_layers=${TRANSCONV_LAYERS:-2} \
    model.queries_per_sec=${QUERIES_PER_SEC:-4} \
    model.mel_bands=${MEL_BANDS:-80} \
    model.p_modality_av=${P_AV:-1.0} \
    model.p_modality_video_only=${P_VIDEO_ONLY:-0.0} \
    model.p_modality_audio_only=${P_AUDIO_ONLY:-0.0} \
    +model.modality_mode=${MODALITY_MODE:-av} \
    model.w2v_path=${W2V_PATH} \
    model.stage1_checkpoint="" \
    model.vocoder_checkpoint="" \
    criterion.mel_loss_weight=${MEL_LOSS_WEIGHT:-1.0} \
    optimization.lr=[${LR:-3e-5}] \
    optimization.max_update=${MAX_UPDATE:-300000} \
    optimization.max_epoch=${MAX_EPOCH:-20000} \
    optimization.update_freq=[${UPDATE_FREQ:-1}] \
    lr_scheduler.warmup_updates=${WARMUP_UPDATES:-2000} \
    distributed_training.distributed_world_size=${NGPUS} \
    $DIST_ARGS \
    distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true
