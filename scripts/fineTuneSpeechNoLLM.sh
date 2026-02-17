#! /bin/bash

ROOT=$(pwd)
SRC_PTH=$ROOT/src
NGPUS=1
QPS=3 
OUT_PATH=$ROOT/exp/mms-speech-NoLLM/speech_mels100hz_128bands_bz20_48G_freq4_wrmp2000_mask_modalitydrop_norm_fiftyfifty

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

echo "Running Speech Fine-tuning (No LLM)..."
echo "Output Dir: $OUT_PATH"

# Run Fine-tuning
# No checkpoint loading!
# Batch size reduced for 48GB GPU

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
    --config-dir ${SRC_PTH}/conf/ \
    --config-name mms-speech-nollm.yaml \
    task.data=$ROOT/manifest/germanManifest \
    task.label_dir=$ROOT/manifest/germanManifest \
    task.noise_prob=0.0 \
    task.noise_wav=$ROOT/noise/babble_noise.wav \
    dataset.batch_size=20 \
    dataset.max_tokens=23000 \
    dataset.required_batch_size_multiple=1 \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    common.seed=1 \
    common.log_interval=10 \
    common.empty_cache_freq=5 \
    common.tensorboard_logdir=${OUT_PATH}/tensorboard \
    checkpoint.save_dir=${OUT_PATH}/checkpoints \
    model.w2v_path=$ROOT/pretrained_models/avhubert/large_vox_iter5.pt \
    model.queries_per_sec=$QPS \
    model.modality_fuse=concat \
    model.use_qformer=true \
    model.use_sr_predictor=true \
    model.p_modality_av=0.0 \
    model.p_modality_video_only=0.5 \
    model.p_modality_audio_only=0.5 \
    optimization.update_freq=[4] \
    optimization.lr=[1e-4] \
    optimizer._name=adam \
    +optimizer.weight_decay=0.01 \
    optimization.max_update=600000 \
    optimization.max_epoch=200 \
    lr_scheduler._name=cosine \
    lr_scheduler.warmup_updates=2000 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS} \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true
