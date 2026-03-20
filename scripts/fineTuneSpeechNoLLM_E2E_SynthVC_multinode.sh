#!/bin/bash
#
# Multi-Node SynthVC-Inspired E2E Training
#
# This is the SynthVC variant of fineTuneSpeechNoLLM_E2E_multinode.sh.
# Key differences:
#   - Uses mms-speech-nollm-e2e-synthvc.yaml config
#   - Uses augmented manifests (*Augmented.tsv)
#   - Adds conv_loss_weight parameter
#   - Output goes to exp/mms-speech-NoLLM-E2E-SynthVC/
#
# Usage: launched via submit script (sbatch), NOT directly.
#

ROOT=$(pwd)
SRC_PTH=$ROOT/src

# ---------- Auto-detect node/GPU config from SLURM ----------
NPROCS_PER_NODE=2  # All GPU partitions on this cluster have 2 GPUs per node
if [ -n "$SLURM_JOB_NUM_NODES" ]; then
    N_NODES=$SLURM_JOB_NUM_NODES
else
    N_NODES=${N_NODES:-1}
fi
NGPUS=$((N_NODES * NPROCS_PER_NODE))

# Compute global base rank for this node
if [ -n "$SLURM_NODEID" ]; then
    NODE_RANK=$((SLURM_NODEID * NPROCS_PER_NODE))
else
    NODE_RANK=0
fi
# -------------------------------------------------------------

QPS=3
RUN_NAME=${RUN_NAME:-default_run}
OUT_PATH=/gpfs/data/fs72969/enriqueoo97/exp/mms-speech-NoLLM-E2E-SynthVC/$RUN_NAME

# Pretrained checkpoints
STAGE1_CKPT=$ROOT/pretrained_models/checkpoint_last.pt
VOCODER_CKPT=$ROOT/pretrained_models/hifigan/model-best.pt

# Create output directory
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"

# Auto-detect and append CUDA libraries
if [ -n "$CONDA_PREFIX" ]; then
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib"
    TORCH_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Explicitly set parallel tokenizers to false to avoid deadlock warnings
export TOKENIZERS_PARALLELISM=false

# NCCL configuration for multi-node communication
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

echo "====== SynthVC Multi-Node Training Config ======"
echo "N_NODES=$N_NODES, NPROCS_PER_NODE=$NPROCS_PER_NODE, NGPUS=$NGPUS"
echo "SLURM_NODEID=$SLURM_NODEID, NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"
echo "Output Dir: $OUT_PATH"
echo "Stage 1 Checkpoint: $STAGE1_CKPT"
echo "Vocoder Checkpoint: $VOCODER_CKPT"
echo "================================================="

if [ "${RESUME}" = "true" ]; then
    # ---- RESUME MODE: rebuild command from saved overrides ----
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

    CUDA_VISIBLE_DEVICES=0,1 fairseq-hydra-train \
        --config-dir ${SRC_PTH}/conf/ \
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
    # ---- NORMAL MODE: use hardcoded hyperparameters ----
    CUDA_VISIBLE_DEVICES=0,1 fairseq-hydra-train \
        --config-dir ${SRC_PTH}/conf/ \
        --config-name mms-speech-nollm-e2e-synthvc.yaml \
        task.data=$ROOT/manifest/germanManifest \
        task.label_dir=$ROOT/manifest/germanManifest \
        task.noise_prob=0.50 \
        task.noise_wav=$ROOT/noise/babble_noise.wav \
        dataset.batch_size=${BATCH_SIZE:-5} \
        dataset.max_tokens=23000 \
        dataset.required_batch_size_multiple=1 \
        hydra.run.dir=${OUT_PATH} \
        common.user_dir=${SRC_PTH} \
        common.fp16=false \
        common.bf16=true \
        common.seed=1 \
        common.log_interval=10 \
        common.empty_cache_freq=5 \
        common.tensorboard_logdir=${OUT_PATH}/tensorboard \
        checkpoint.save_dir=${OUT_PATH}/checkpoints \
        model.w2v_path=$ROOT/pretrained_models/avhubert/large_vox_iter5.pt \
        model.queries_per_sec=$QPS \
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
        criterion.disc_grad_clip=${DISC_GRAD_CLIP:-5.0} \
        criterion.adv_warmup_updates=${ADV_WARMUP_UPDATES:-5000} \
        criterion.use_multires_mel=${USE_MULTIRES_MEL:-false} \
        model.mel_hop_size=${MEL_HOP_SIZE:-160} \
        optimization.update_freq=[4] \
        optimization.lr=[2e-4] \
        optimization.clip_norm=${CLIP_NORM:-1.0} \
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
