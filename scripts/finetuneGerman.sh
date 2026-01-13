#! /bin/bash

ROOT=$(pwd)
SRC_PTH=$ROOT/src
LLM_PATH="meta-llama/Llama-3.2-3B"
NGPUS=8
QPS=3 
OUT_PATH=$ROOT/exp/mms-llama/1759h_german_finetune8GPU-e1
PRETRAINED_CHECKPOINT="$ROOT/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"

# Create output directory
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"

# Auto-detect and append CUDA libraries (exactly as in evalGerman.sh)
if [ -n "$CONDA_PREFIX" ]; then
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib"
    TORCH_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Explicitly set parallel tokenizers to false to avoid deadlock warnings
export TOKENIZERS_PARALLELISM=false

# --- NCCL / DISTRIBUTED CONFIGURATION FOR BLACKWELL/MIG ---
# Disable P2P and InfiniBand to prevent hangs/crashes on MIG instances
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Disable NVLink Sharpening (NVLS) and CUDA Memory (CUMEM) features to prevent MIG aliasing confusion
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
# Force NCCL to verify device uniqueness via UUIDs if possible (indirectly via disabling other paths)
export NCCL_GRAPH_DISABLE=1

# Enable detailed logging for diagnosing distributed issues
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

# Use expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running Fine-tuning on German Dataset..."
echo "Resume Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output Dir: $OUT_PATH"

# Run Fine-tuning
# Using fairseq-hydra-train as in the original train.sh
# Added checkpoint.finetune_from_model to load the weights but reset optimizer
# Reduced learning rate (2e-5) for fine-tuning stability
# 8 GPUs available, so update_freq=1 is sufficient (effective batch size scaled by 8)

fairseq-hydra-train \
    --config-dir ${SRC_PTH}/conf/ \
    --config-name mms-llama.yaml \
    task.data=$ROOT/manifest/germanManifest \
    task.label_dir=$ROOT/manifest/germanManifest \
    task.llm_path=${LLM_PATH} \
    task.noise_prob=0.75 \
    task.noise_wav=$ROOT/noise/babble_noise.wav \
    dataset.batch_size=12 \
    dataset.max_tokens=8000 \
    dataset.ignore_unused_valid_subsets=true \
    dataset.num_workers=3 \
    dataset.valid_subset=valid \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    common.seed=1 \
    common.log_interval=10 \
    common.empty_cache_freq=1000 \
    common.tensorboard_logdir=${OUT_PATH}/tensorboard \
    checkpoint.finetune_from_model=${PRETRAINED_CHECKPOINT} \
    checkpoint.save_dir=${OUT_PATH}/checkpoints \
    model.w2v_path=$ROOT/pretrained_models/avhubert/large_vox_iter5.pt \
    model.llm_path=${LLM_PATH} \
    model.llama_embed_dim=3072 \
    model.queries_per_sec=$QPS \
    model.target_modules=q_proj.k_proj.v_proj.o_proj \
    model.modality_fuse=concat \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.use_qformer=true \
    model.use_sr_predictor=true \
    optimization.update_freq=[1] \
    optimization.lr=[1e-4] \
    optimization.max_update=15000 \
    lr_scheduler._name=cosine \
    lr_scheduler.warmup_updates=500 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS} \
    distributed_training.distributed_backend=gloo \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true
