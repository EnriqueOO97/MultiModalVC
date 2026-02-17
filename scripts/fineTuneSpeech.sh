#! /bin/bash

ROOT=$(pwd)
SRC_PTH=$ROOT/src
LLM_PATH="meta-llama/Llama-3.2-3B"
NGPUS=1
QPS=2 
OUT_PATH=$ROOT/exp/mms-llama-speech/speech_finetune_1GPU_mels100hz_128bands_freq8-lesstokens
PRETRAINED_CHECKPOINT="$ROOT/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"

# Create output directory
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"

# Auto-detect and append CUDA libraries (exactly as in fineTuneAmpere-Blackwell.sh)
if [ -n "$CONDA_PREFIX" ]; then
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib"
    TORCH_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Explicitly set parallel tokenizers to false to avoid deadlock warnings
export TOKENIZERS_PARALLELISM=false

echo "Running Speech Fine-tuning..."
echo "Resume Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output Dir: $OUT_PATH"

# Run Fine-tuning
# Key Changes from Original:
# 1. config-name=mms-llama-speech (Uses mel_spectrogram_l1_loss)
# 2. model._name=MMS_LLaMA_Speech
# 3. checkpoint.finetune_from_model handles loading the ASR-finetuned weights into the base components
# 4. Strict loading might fail due to new layers (proj1, etc.), so we might need reset_optimizer=True implicitly 

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
    --config-dir ${SRC_PTH}/conf/ \
    --config-name mms-llama-speech.yaml \
    task.data=$ROOT/manifest/germanManifest \
    task.label_dir=$ROOT/manifest/germanManifest \
    task.llm_path=${LLM_PATH} \
    task.noise_prob=0.0 \
    task.noise_wav=$ROOT/noise/babble_noise.wav \
    dataset.batch_size=20 \
    dataset.max_tokens=10000 \
    dataset.required_batch_size_multiple=1 \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    common.seed=1 \
    common.log_interval=10 \
    common.empty_cache_freq=5 \
    common.tensorboard_logdir=${OUT_PATH}/tensorboard \
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
    optimization.update_freq=[8] \
    optimization.lr=[1e-4] \
    optimizer._name=adam \
    +optimizer.weight_decay=0.01 \
    optimization.max_update=60000 \
    optimization.max_epoch=200 \
    lr_scheduler._name=cosine \
    lr_scheduler.warmup_updates=500 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS} \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true
