#!/bin/bash
# ============================================================
# Musica (SLURM) fine-tune script — pathological -> healthy stage
# SLURM-compatible port of finetuneVC.sh.
#   * Launched via:  srun bash scripts/finetuneVC-MUSICA.sh
#     (see scripts/musica_submit_finetune.sh, which discovers MASTER_ADDR /
#      MASTER_PORT and activates the conda env before srun).
#   * Same training recipe/flags as finetuneVC.sh; only the cluster-specific
#     PATHS and the distributed-launch wiring differ.
#   * Still runs locally with NO SLURM (single GPU) for quick debugging.
#
# >>> The only things you should ever need to edit are the PATHS block below. <<<
# ============================================================

set -e

# ============================================================
# PATHS  (Musica cluster)  — edit these if files move
# ============================================================
ROOT=/data/fs201163/eo49197/MultiModalVC

# Checkpoint to fine-tune from (old cluster: outputsInference/checkpoint331-mel30.pt)
PHASE2_CKPT=${PHASE2_CKPT:-$ROOT/exp/multiModalVC-synth/2level_transconv_CQT_mel30/checkpoints/checkpoint331.pt}

# Manifest dir holding trainPATH-HE.tsv / validPATH-HE.tsv / testPATH-HE.tsv
FINETUNE_DATA=${FINETUNE_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-healthyTY}

# DNSMOS-Pro checkpoint (only used when USE_DNSMOS=true)
DNSMOS_CKPT=${DNSMOS_CKPT:-$ROOT/pretrained_models/dnsmos/model_best.pt}

# Pretrained HiFi-GAN generator used to warm-start the vocoder (the generator is
# NOT inherited from any phase-2 checkpoint in this fresh concat run).
HIFIGAN_CKPT=${HIFIGAN_CKPT:-$ROOT/pretrained_models/hifigan/model-best.pt}

# CQT-discriminator warm-start. We discard the cross-att checkpoint's GENERATOR
# (contaminated by the fusion bug), but its CQT discriminator only ever saw
# waveforms/spectra — it is bug-independent and shape-independent of modality_fuse,
# so we rescue just its weights. Set to "" to start the disc from scratch instead.
DISC_INIT_CKPT=${DISC_INIT_CKPT:-$ROOT/exp/multiModalVC-synth/2level_transconv_CQT_mel30/checkpoints/checkpoint331.pt}

# Externally-finetuned Whisper encoder (HF dir w/ model.safetensors, or .safetensors).
# Empty string => stock openai/whisper-medium.
WHISPER_PRETRAINED_PATH=${WHISPER_PRETRAINED_PATH:-$ROOT/pretrained_models/whisper-medium/checkpoint-2900}

# AV-HuBERT pretrained weights
W2V_PATH=${W2V_PATH:-$ROOT/pretrained_models/avhubert/large_vox_iter5.pt}
# ============================================================

RUN_NAME=${RUN_NAME:-pathological_finetune_default}
OUT_PATH=$ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH"

# ---------- Distributed / GPU config ----------
# Under SLURM (submitted via musica_submit_finetune.sh) inherit the node/GPU
# layout plus the master addr/port discovered by the submit script. With no
# SLURM it falls back to a single-GPU local run.
if [ -n "$SLURM_JOB_ID" ]; then
    # Single-node launch (invoked directly from the batch body, no srun — see
    # musica_submit_finetune.sh for why). Spawn one rank per GPU actually visible
    # to this process; counting CUDA_VISIBLE_DEVICES is the most reliable signal
    # of what we can use under the cluster's managed partial-GPU layout.
    if [ -n "$NPROCS_PER_NODE" ]; then
        :                                          # explicit override wins
    elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        NPROCS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
        NPROCS_PER_NODE=$SLURM_GPUS_ON_NODE
    else
        NPROCS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
    fi
    [ "${NPROCS_PER_NODE:-0}" -ge 1 ] || NPROCS_PER_NODE=1
    NGPUS=$NPROCS_PER_NODE                          # one node
    NODE_RANK=0
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n "${SLURM_JOBID:-0}" | tail -c 4))}
    # Local rendezvous; fairseq spawns NPROCS_PER_NODE ranks (rank = NODE_RANK + local).
    DIST_ARGS="distributed_training.distributed_port=${MASTER_PORT} distributed_training.distributed_init_method=tcp://${MASTER_ADDR}:${MASTER_PORT} distributed_training.distributed_rank=${NODE_RANK}"
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1                        # single-node: force NVLink/NVSwitch
    CUDA_DEVICES=${CUDA_VISIBLE_DEVICES}
else
    NGPUS=${NGPUS:-1}
    NPROCS_PER_NODE=${NPROCS_PER_NODE:-1}
    NODE_RANK=0
    DIST_ARGS=""
    CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi

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

# --- Auto-inherit architecture flags from the source run's saved overrides ---
# The architecture flags MUST match the run that produced PHASE2_CKPT. We read the
# overrides.yaml that sits next to the checkpoint (run_dir/.hydra/overrides.yaml)
# and inherit only architecture-affecting keys.
SOURCE_RUN_DIR=$(dirname "$(dirname "$PHASE2_CKPT")")
SOURCE_OVERRIDES_FILE=$SOURCE_RUN_DIR/.hydra/overrides.yaml
if [ ! -f "$SOURCE_OVERRIDES_FILE" ]; then
    echo "WARNING: cannot find source overrides at $SOURCE_OVERRIDES_FILE"
    echo "         (derived from PHASE2_CKPT=$PHASE2_CKPT)"
    echo "         Continuing without architecture overrides — pass them manually if needed."
    ARCH_OVERRIDES=""
else
    echo "[finetune] inheriting architecture from $SOURCE_OVERRIDES_FILE"
    ARCH_OVERRIDES=$(grep '^- ' "$SOURCE_OVERRIDES_FILE" \
        | sed 's/^- //' \
        | grep -E '^\+?model\.|^criterion\.(mel_num_mels|mel_hop_size|use_multires_mel)=' \
        | grep -vE '^model\.(stage1_checkpoint|vocoder_checkpoint|modality_fuse)=' \
        | tr '\n' ' ')
    echo "[finetune] inherited overrides: $ARCH_OVERRIDES"
fi

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

echo "====== finetuneVC-MUSICA ======"
echo "RUN_NAME          : $RUN_NAME"
echo "OUT_PATH          : $OUT_PATH"
echo "PHASE2_CKPT       : $PHASE2_CKPT"
echo "FINETUNE_DATA     : $FINETUNE_DATA"
echo "NGPUS / per-node  : $NGPUS / $NPROCS_PER_NODE   (NODE_RANK=$NODE_RANK)"
echo "================================"

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
        model.w2v_path=${W2V_PATH} \
        distributed_training.distributed_world_size=${NGPUS} \
        $DIST_ARGS \
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
        +task.disc_init_checkpoint=${DISC_INIT_CKPT} \
        dataset.batch_size=${BATCH_SIZE:-12} \
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
        checkpoint.no_last_checkpoints=true \
        checkpoint.best_checkpoint_metric=mcd_healthy \
        checkpoint.maximize_best_checkpoint_metric=false \
        $ARCH_OVERRIDES \
        model.modality_fuse=concat \
        +model.transconv_layers=${TRANSCONV_LAYERS:-2} \
        model.p_modality_av=1.0 \
        model.p_modality_video_only=0.0 \
        model.p_modality_audio_only=0.0 \
        model.w2v_path=${W2V_PATH} \
        model.stage1_checkpoint="" \
        model.vocoder_checkpoint=${HIFIGAN_CKPT} \
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
        +criterion.dnsmos_checkpoint=${DNSMOS_CKPT} \
        optimization.update_freq=[${UPDATE_FREQ:-2}] \
        optimization.clip_norm=${CLIP_NORM:-20.0} \
        optimization.lr=[${LR:-3e-5}] \
        optimizer._name=adam \
        +optimizer.weight_decay=0.01 \
        optimization.max_update=${MAX_UPDATE:-900000} \
        optimization.max_epoch=${MAX_EPOCH:-20000} \
        lr_scheduler._name=fixed \
        lr_scheduler.warmup_updates=${WARMUP_UPDATES:-500} \
        distributed_training.distributed_world_size=${NGPUS} \
        $DIST_ARGS \
        distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
        distributed_training.ddp_backend=legacy_ddp \
        distributed_training.find_unused_parameters=true
fi
