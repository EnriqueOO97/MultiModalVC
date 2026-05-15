#!/bin/bash
# ============================================================
# Pathological -> Healthy fine-tune training script (launched via srun)
# Mirrors musica_train.sh but:
#   - new task (MMS_LLaMA_pathological_finetune) reading 9-column manifest
#   - loads Phase 2 weights via checkpoint.finetune_from_model (fresh optim/EMA)
#   - architecture flags pinned to the Phase 2 reference run
# ============================================================

NPROCS_PER_NODE=4
if [ -n "$SLURM_JOB_NUM_NODES" ]; then
    N_NODES=$SLURM_JOB_NUM_NODES
else
    N_NODES=${N_NODES:-1}
fi
NGPUS=$((N_NODES * NPROCS_PER_NODE))

if [ -n "$SLURM_NODEID" ]; then
    NODE_RANK=$((SLURM_NODEID * NPROCS_PER_NODE))
else
    NODE_RANK=0
fi

RUN_NAME=${RUN_NAME:-pathological_finetune_default}
PROJECT_ROOT=/data/fs201163/eo49197/MultiModalVC
OUT_PATH=$PROJECT_ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$PROJECT_ROOT/fairseq:$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    TORCH_LIB="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/torch/lib"
    if [ -d "$TORCH_LIB" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
    fi
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/nvidia/cusparse/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$LD_LIBRARY_PATH"
    fi
fi

# --- Checkpoint to start from (must be supplied) ---
PHASE2_CKPT=${PHASE2_CKPT:?must set PHASE2_CKPT to the source checkpoint .pt path}

# --- Auto-inherit architecture flags from the source run's saved overrides ---
# The architecture flags MUST match the run that produced PHASE2_CKPT, otherwise
# the loaded weights will not map onto the freshly-built model.  We read the
# saved overrides.yaml that sits next to the checkpoint and inherit only the
# architecture-affecting keys: model.* (all of them) plus the criterion fields
# that govern which modules instantiate (mel_num_mels / mel_hop_size /
# use_multires_mel).  Everything else (training hyper-params, dataset, etc.)
# is set explicitly below.
SOURCE_RUN_DIR=$(dirname $(dirname "$PHASE2_CKPT"))
SOURCE_OVERRIDES_FILE=$SOURCE_RUN_DIR/.hydra/overrides.yaml
if [ ! -f "$SOURCE_OVERRIDES_FILE" ]; then
    echo "ERROR: cannot find source overrides at $SOURCE_OVERRIDES_FILE"
    echo "       (derived from PHASE2_CKPT=$PHASE2_CKPT)"
    exit 1
fi
echo "[finetune] inheriting architecture from $SOURCE_OVERRIDES_FILE"
ARCH_OVERRIDES=$(grep '^- ' "$SOURCE_OVERRIDES_FILE" \
    | sed 's/^- //' \
    | grep -E '^\+?model\.|^criterion\.(mel_num_mels|mel_hop_size|use_multires_mel)=' \
    | grep -vE '^model\.(stage1_checkpoint|vocoder_checkpoint)=' \
    | tr '\n' ' ')
echo "[finetune] inherited overrides: $ARCH_OVERRIDES"

# --- Manifest directory holding trainPATH-HE.tsv / validPATH-HE.tsv / testPATH-HE.tsv ---
FINETUNE_DATA=${FINETUNE_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/Sync_recordings_cropped}

# --- Launch ---
if [ "${RESUME}" = "true" ]; then
    OVERRIDES_FILE=${OUT_PATH}/.hydra/overrides.yaml
    if [ ! -f "$OVERRIDES_FILE" ]; then
        echo "ERROR: Cannot resume — $OVERRIDES_FILE not found."
        exit 1
    fi
    echo "RESUME MODE: Reading overrides from $OVERRIDES_FILE"
    SAVED_OVERRIDES=$(grep '^- ' "$OVERRIDES_FILE" \
        | sed 's/^- //' \
        | grep -v '^distributed_training\.' \
        | grep -v '^hydra\.run\.dir')

    fairseq-hydra-train \
        --config-dir ${PROJECT_ROOT}/src/conf/ \
        --config-name mms-speech-nollm-e2e-synthvc.yaml \
        hydra.run.dir=${OUT_PATH} \
        $SAVED_OVERRIDES \
        distributed_training.distributed_world_size=${NGPUS} \
        distributed_training.distributed_port=${MASTER_PORT} \
        distributed_training.distributed_init_method="tcp://${MASTER_ADDR}:${MASTER_PORT}" \
        distributed_training.distributed_rank=${NODE_RANK} \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
else
    fairseq-hydra-train \
        --config-dir ${PROJECT_ROOT}/src/conf/ \
        --config-name mms-speech-nollm-e2e-synthvc.yaml \
        task._name=MMS_LLaMA_pathological_finetune \
        task.data=${FINETUNE_DATA} \
        dataset.batch_size=${BATCH_SIZE:-9} \
        dataset.max_tokens=23000 \
        dataset.required_batch_size_multiple=1 \
        dataset.valid_subset=valid \
        dataset.validate_interval=${VALIDATE_INTERVAL:-1} \
        hydra.run.dir=${OUT_PATH} \
        common.user_dir=${PROJECT_ROOT}/src \
        common.fp16=false \
        common.bf16=true \
        common.seed=1 \
        common.log_interval=10 \
        common.empty_cache_freq=5 \
        common.tensorboard_logdir=${OUT_PATH}/tensorboard \
        checkpoint.save_dir=${OUT_PATH}/checkpoints \
        checkpoint.finetune_from_model=${PHASE2_CKPT} \
        $ARCH_OVERRIDES \
        model.stage1_checkpoint="" \
        model.vocoder_checkpoint="" \
        criterion.mel_loss_weight=${MEL_LOSS_WEIGHT:-15.0} \
        criterion.conv_loss_weight=${CONV_LOSS_WEIGHT:-10.0} \
        criterion.disc_start_updates=${DISC_START_UPDATES:-0} \
        criterion.disc_grad_clip=${DISC_GRAD_CLIP:-20.0} \
        criterion.adv_warmup_updates=${ADV_WARMUP_UPDATES:-200} \
        +criterion.use_discriminator=${USE_DISC:-true} \
        +criterion.disc_pretrain=${DISC_PRETRAIN:-false} \
        optimization.update_freq=[${UPDATE_FREQ:-3}] \
        optimization.clip_norm=${CLIP_NORM:-20.0} \
        optimization.lr=[${LR:-1e-5}] \
        optimizer._name=adam \
        +optimizer.weight_decay=0.01 \
        optimization.max_update=${MAX_UPDATE:-10000} \
        optimization.max_epoch=${MAX_EPOCH:-50} \
        lr_scheduler._name=cosine \
        lr_scheduler.warmup_updates=${WARMUP_UPDATES:-200} \
        distributed_training.distributed_world_size=${NGPUS} \
        distributed_training.distributed_port=${MASTER_PORT} \
        distributed_training.distributed_init_method="tcp://${MASTER_ADDR}:${MASTER_PORT}" \
        distributed_training.distributed_rank=${NODE_RANK} \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
fi
