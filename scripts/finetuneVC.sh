#!/bin/bash
# ============================================================
# Local single-GPU (Blackwell) fine-tune script
# Mirrors musica_train_finetune.sh but for 1 GPU, no SLURM.
# Launch directly: bash scripts/finetuneVC.sh
# ============================================================

set -e

# ---------- Configuration ----------
ROOT=/ceph/home/TUG/olivares-tug/MMS-LLaMA
NGPUS=${NGPUS:-1}
NPROCS_PER_NODE=${NPROCS_PER_NODE:-1}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN_NAME=${RUN_NAME:-pathological_finetune_VocoderFrozen_NoDisc}
OUT_PATH=$ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH"

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Auto-detect and append CUDA libraries
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

# --- Checkpoint to start from ---
PHASE2_CKPT=${PHASE2_CKPT:-/ceph/home/TUG/olivares-tug/MMS-LLaMA/outputsInference/checkpoint331-mel30.pt}

# --- Auto-inherit architecture flags from the source run's saved overrides ---
SOURCE_OVERRIDES_FILE=/ceph/home/TUG/olivares-tug/MMS-LLaMA/pretrained_models/overrides.yaml
if [ ! -f "$SOURCE_OVERRIDES_FILE" ]; then
    echo "WARNING: cannot find source overrides at $SOURCE_OVERRIDES_FILE"
    echo "         Continuing without architecture overrides — ensure you pass them manually if needed."
    ARCH_OVERRIDES=""
else
    echo "[finetune] inheriting architecture from $SOURCE_OVERRIDES_FILE"
    ARCH_OVERRIDES=$(grep '^- ' "$SOURCE_OVERRIDES_FILE" \
        | sed 's/^- //' \
        | grep -E '^\+?model\.|^criterion\.(mel_num_mels|mel_hop_size|use_multires_mel)=' \
        | grep -vE '^model\.(stage1_checkpoint|vocoder_checkpoint)=' \
        | tr '\n' ' ')
    echo "[finetune] inherited overrides: $ARCH_OVERRIDES"
fi

# --- Manifest directory holding trainPATH-HE.tsv / validPATH-HE.tsv / testPATH-HE.tsv ---
FINETUNE_DATA=${FINETUNE_DATA:-/ceph/shared/ALL/datasets/VoiceConversion-fwf/Sync_recordings_cropped}

# --- Per-module freeze knobs (each one toggles independently) ---
AFEAT_TRAINABLE=${AFEAT_TRAINABLE:-true}
FUSION_TRAINABLE=${FUSION_TRAINABLE:-true}
QFORMER_TRAINABLE=${QFORMER_TRAINABLE:-true}
PROJ_TRAINABLE=${PROJ_TRAINABLE:-true}
CONFORMER_TRAINABLE=${CONFORMER_TRAINABLE:-true}
VOCODER_TRAINABLE=${VOCODER_TRAINABLE:-true}
WHISPER_TOP_N=${WHISPER_TOP_N:-2}
WHISPER_LN_TRAINABLE=${WHISPER_LN_TRAINABLE:-true}

# --- Target selection: real healthy always used; this many synths (1..N) added ---
NUMBER_OF_SYNTHS=${NUMBER_OF_SYNTHS:-0}

# --- Externally-finetuned Whisper encoder swap (HF dir with model.safetensors, or .safetensors file) ---
# Empty string => stock openai/whisper-medium.
WHISPER_PRETRAINED_PATH=${WHISPER_PRETRAINED_PATH:-$ROOT/pretrained_models/whisper-medium/checkpoint-2900}

# ---------- Launch ----------
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
        | grep -v '^hydra\.run\.dir' \
        | grep -v '^checkpoint\.restore_file' \
        | grep -v 'reset_disc_schedule')

    # No checkpoint_last.pt is saved (no_last_checkpoints=true), so fairseq's
    # auto-resume can't find a restore point. Pick the highest-numbered checkpoint.
    LATEST_CKPT=$(ls -1 ${OUT_PATH}/checkpoints/checkpoint[0-9]*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: RESUME requested but no numbered checkpoint found in ${OUT_PATH}/checkpoints"
        exit 1
    fi
    echo "RESUME MODE: restoring from $LATEST_CKPT"

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} fairseq-hydra-train \
        --config-dir ${ROOT}/src/conf/ \
        --config-name mms-speech-nollm-e2e-synthvc.yaml \
        hydra.run.dir=${OUT_PATH} \
        $SAVED_OVERRIDES \
        +criterion.reset_disc_schedule=false \
        checkpoint.restore_file=${LATEST_CKPT} \
        model.w2v_path=${ROOT}/pretrained_models/avhubert/large_vox_iter5.pt \
        distributed_training.distributed_world_size=${NGPUS} \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
else
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} fairseq-hydra-train \
        --config-dir ${ROOT}/src/conf/ \
        --config-name mms-speech-nollm-e2e-synthvc.yaml \
        task._name=MMS_LLaMA_pathological_finetune \
        task.data=${FINETUNE_DATA} \
        +task.afeat_1d_conv_trainable=${AFEAT_TRAINABLE} \
        +task.fusion_trainable=${FUSION_TRAINABLE} \
        +task.qformer_trainable=${QFORMER_TRAINABLE} \
        +task.proj_trainable=${PROJ_TRAINABLE} \
        +task.conformer_trainable=${CONFORMER_TRAINABLE} \
        +task.vocoder_trainable=${VOCODER_TRAINABLE} \
        +task.whisper_top_n_trainable=${WHISPER_TOP_N} \
        +task.whisper_layernorm_trainable=${WHISPER_LN_TRAINABLE} \
        +task.number_of_synths=${NUMBER_OF_SYNTHS} \
        +task.whisper_pretrained_path=${WHISPER_PRETRAINED_PATH} \
        dataset.batch_size=${BATCH_SIZE:-10} \
        dataset.max_tokens=4000 \
        dataset.required_batch_size_multiple=1 \
        dataset.valid_subset=valid \
        dataset.validate_interval=${VALIDATE_INTERVAL:-10} \
        hydra.run.dir=${OUT_PATH} \
        common.user_dir=${ROOT}/src \
        common.fp16=false \
        common.bf16=true \
        common.seed=1 \
        common.log_interval=1 \
        common.empty_cache_freq=5 \
        common.tensorboard_logdir=${OUT_PATH}/tensorboard \
        checkpoint.save_dir=${OUT_PATH}/checkpoints \
        checkpoint.finetune_from_model=${PHASE2_CKPT} \
        checkpoint.no_last_checkpoints=true \
        checkpoint.best_checkpoint_metric=mcd_healthy \
        checkpoint.maximize_best_checkpoint_metric=false \
        $ARCH_OVERRIDES \
        model.p_modality_av=1.0 \
        model.p_modality_video_only=0.0 \
        model.p_modality_audio_only=0.0 \
        model.w2v_path=${ROOT}/pretrained_models/avhubert/large_vox_iter5.pt \
        model.stage1_checkpoint="" \
        model.vocoder_checkpoint="" \
        criterion.mel_loss_weight=${MEL_LOSS_WEIGHT:-30.0} \
        criterion.conv_loss_weight=${CONV_LOSS_WEIGHT:-10.0} \
        criterion.disc_lr=${DISC_LR:-3e-5} \
        criterion.disc_start_updates=${DISC_START_UPDATES:-25000} \
        criterion.disc_grad_clip=${DISC_GRAD_CLIP:-20.0} \
        criterion.adv_warmup_updates=${ADV_WARMUP_UPDATES:-5000} \
        +criterion.use_discriminator=${USE_DISC:-false} \
        +criterion.disc_pretrain=${DISC_PRETRAIN:-true} \
        +criterion.freeze_disc=${FREEZE_DISC:-false} \
        +criterion.reset_disc_schedule=${RESET_DISC_SCHEDULE:-true} \
        +criterion.use_mrstft_loss=${USE_MRSTFT:-false} \
        +criterion.mrstft_loss_weight=${MRSTFT_WEIGHT:-2.0} \
        +criterion.use_dnsmos_loss=${USE_DNSMOS:-false} \
        +criterion.dnsmos_loss_weight=${DNSMOS_WEIGHT:-1.0} \
        +criterion.dnsmos_checkpoint=${DNSMOS_CKPT:-/ceph/home/TUG/olivares-tug/DNSMOSPro/runs/BVCC/model_best.pt} \
        optimization.update_freq=[${UPDATE_FREQ:-4}] \
        optimization.clip_norm=${CLIP_NORM:-20.0} \
        optimization.lr=[${LR:-3e-5}] \
        optimizer._name=adam \
        +optimizer.weight_decay=0.01 \
        optimization.max_update=${MAX_UPDATE:-90000} \
        optimization.max_epoch=${MAX_EPOCH:-2000} \
        lr_scheduler._name=fixed \
        lr_scheduler.warmup_updates=${WARMUP_UPDATES:-500} \
        distributed_training.distributed_world_size=${NGPUS} \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
fi
