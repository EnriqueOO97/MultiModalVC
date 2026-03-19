#!/bin/bash
# ============================================================
# Training script for Musica (launched via srun)
# ============================================================

# ---------- Auto-detect node/GPU config from SLURM ----------
NPROCS_PER_NODE=4  # H100 nodes have 4 GPUs
if [ -n "$SLURM_JOB_NUM_NODES" ]; then
    N_NODES=$SLURM_JOB_NUM_NODES
else
    N_NODES=${N_NODES:-1}
fi
NGPUS=$((N_NODES * NPROCS_PER_NODE))

# Compute global base rank for this node (matching original script)
if [ -n "$SLURM_NODEID" ]; then
    NODE_RANK=$((SLURM_NODEID * NPROCS_PER_NODE))
else
    NODE_RANK=0
fi
# -------------------------------------------------------------

RUN_NAME=${RUN_NAME:-default_run}
PROJECT_ROOT=/data/fs201163/eo49197/MultiModalVC
OUT_PATH=$PROJECT_ROOT/exp/multiModalVC-synth/$RUN_NAME

# Create output directory
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$PROJECT_ROOT/fairseq:$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Single-node job: force NVLink (NVSwitch) instead of InfiniBand

# Auto-detect and append CUDA libraries (fixing version discovery)
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Add Torch libraries (essential for torchvision and distributed)
    TORCH_LIB="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/torch/lib"
    if [ -d "$TORCH_LIB" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
    fi

    # Add Nvidia/Cusparse libraries if present
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/nvidia/cusparse/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Checkpoints
STAGE1_CKPT=$PROJECT_ROOT/pretrained_models/checkpoint_best_STAGE1.pt
VOCODER_CKPT=$PROJECT_ROOT/pretrained_models/hifigan/model-best.pt
W2V_PATH=$PROJECT_ROOT/pretrained_models/avhubert/large_vox_iter5.pt

# Launch training logic
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
        task.data=$PROJECT_ROOT/manifest/germanManifest \
        task.label_dir=$PROJECT_ROOT/manifest/germanManifest \
        task.noise_prob=0.50 \
        task.noise_wav=$PROJECT_ROOT/noise/babble_noise.wav \
        dataset.batch_size=${BATCH_SIZE:-5} \
        dataset.max_tokens=23000 \
        dataset.required_batch_size_multiple=1 \
        hydra.run.dir=${OUT_PATH} \
        common.user_dir=${PROJECT_ROOT}/src \
        common.fp16=false \
        common.bf16=true \
        common.seed=1 \
        common.log_interval=10 \
        common.empty_cache_freq=5 \
        common.tensorboard_logdir=${OUT_PATH}/tensorboard \
        checkpoint.save_dir=${OUT_PATH}/checkpoints \
        model.w2v_path=${W2V_PATH} \
        model.queries_per_sec=3 \
        model.modality_fuse=cross-att \
        model.use_qformer=true \
        model.use_sr_predictor=true \
        model.p_modality_av=0.75 \
        model.p_modality_video_only=0.25 \
        model.p_modality_audio_only=0.0 \
        model.stage1_checkpoint=${STAGE1_CKPT} \
        model.vocoder_checkpoint=${VOCODER_CKPT} \
        model.freeze_stage1=false \
        +criterion.use_discriminator=${USE_DISC:-false} \
        criterion.mel_loss_weight=${MEL_LOSS_WEIGHT:-1.0} \
        criterion.conv_loss_weight=${CONV_LOSS_WEIGHT:-10.0} \
        criterion.disc_start_updates=${DISC_START_UPDATES:-120000} \
        criterion.mel_num_mels=${MEL_NUM_MELS:-128} \
        criterion.mel_hop_size=${MEL_HOP_SIZE:-160} \
        model.mel_hop_size=${MEL_HOP_SIZE:-160} \
        optimization.update_freq=[2] \
        optimization.lr=[2e-4] \
        optimizer._name=adam \
        +optimizer.weight_decay=0.01 \
        optimization.max_update=600000 \
        optimization.max_epoch=200 \
        lr_scheduler._name=cosine \
        lr_scheduler.warmup_updates=2000 \
        distributed_training.distributed_world_size=${NGPUS} \
        distributed_training.distributed_port=${MASTER_PORT} \
        distributed_training.distributed_init_method="tcp://${MASTER_ADDR}:${MASTER_PORT}" \
        distributed_training.distributed_rank=${NODE_RANK} \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
fi
