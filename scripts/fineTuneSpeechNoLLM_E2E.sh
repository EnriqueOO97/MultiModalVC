#!/bin/bash
#
# End-to-End Speech Synthesis Training: Stage 1 + HiFi-GAN
#
# This script trains the merged model where conformer output (512-dim)
# feeds directly into HiFi-GAN's upsampling chain.
#

ROOT=$(pwd)
SRC_PTH=$ROOT/src
NGPUS=1
QPS=3
OUT_PATH=$ROOT/exp/mms-speech-NoLLM-E2E/e2e_96G_cross-att_MD_av50_vo50_noise75

# Pretrained checkpoints
STAGE1_CKPT=$ROOT/exp/mms-speech-NoLLM/speech_bz10_4MiG_freq4_wrmp2000_mask_modalitydrop_noise_fiftyfifty/checkpoints/checkpoint_last.pt
VOCODER_CKPT=/ceph/home/TUG/olivares-tug/DiVISe/exp-vocoder/exp7-finetuning-on-german-EMA-bz-64/model-best.pt

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

echo "Running E2E Speech Training (Stage 1 + HiFi-GAN)..."
echo "Output Dir: $OUT_PATH"
echo "Stage 1 Checkpoint: $STAGE1_CKPT"
echo "Vocoder Checkpoint: $VOCODER_CKPT"

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
    --config-dir ${SRC_PTH}/conf/ \
    --config-name mms-speech-nollm-e2e.yaml \
    task.data=$ROOT/manifest/germanManifest \
    task.label_dir=$ROOT/manifest/germanManifest \
    task.noise_prob=0.75 \
    task.noise_wav=$ROOT/noise/babble_noise.wav \
    dataset.batch_size=15 \
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
    checkpoint.restore_file=${STAGE1_CKPT} \
    checkpoint.reset_optimizer=true \
    checkpoint.reset_lr_scheduler=true \
    checkpoint.reset_dataloader=true \
    checkpoint.reset_meters=true \
    model.w2v_path=$ROOT/pretrained_models/avhubert/large_vox_iter5.pt \
    model.queries_per_sec=$QPS \
    model.modality_fuse=cross-att \
    model.use_qformer=true \
    model.use_sr_predictor=true \
    model.p_modality_av=0.5 \
    model.p_modality_video_only=0.5 \
    model.p_modality_audio_only=0.0 \
    model.vocoder_checkpoint=${VOCODER_CKPT} \
    model.freeze_stage1=true \
    optimization.update_freq=[10] \
    optimization.lr=[2e-4] \
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
