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
#SBATCH --time=72:00:00                 # sweep needs many epochs; 48h to reach ~ep300+ with save_interval=50
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=enrique.orozco1997@gmail.com
set -e

# ---------- PATHS (EDIT for the new cluster) ----------
ROOT=/data/fs201163/eo49197/MultiModalVC                                                 # <-- EDIT
FINETUNE_DATA=${FINETUNE_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/pre-trainVoxcelebYoutube}            # <-- EDIT
WHISPER_PRETRAINED_PATH=${WHISPER_PRETRAINED_PATH:-$ROOT/pretrained_models/whisper-medium/checkpoint-2900}
W2V_PATH=${W2V_PATH:-$ROOT/pretrained_models/avhubert/large_vox_iter5.pt}

# Noise injection (train-only, on the INPUT audio). Default OFF (prob 0.0).
NOISE_WAV=${NOISE_WAV:-$ROOT/noise/babble_noise.wav}
NOISE_PROB=${NOISE_PROB:-0.0}
# Vocoder is inherited from the parent class but UNUSED in MelVC; pass a real
# checkpoint so the parent __init__ doesn't choke building it.
HIFIGAN_CKPT=${HIFIGAN_CKPT:-$ROOT/pretrained_models/hifigan/model-best.pt}

RUN_NAME=${RUN_NAME:-melvc_default}
OUT_PATH=$ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH/logs"

# Route per-node stdio to the run dir (single-node, no srun).
exec >> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_out_${SLURM_JOBID}.log" \
     2>> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_err_${SLURM_JOBID}.log"

# VSC-5 ships CUDA as a module; the conda torch wheels usually bundle their own
# runtime so this is rarely needed. Uncomment only if torch can't see the GPUs.
# module purge
# module load cuda/12.2.0

eval "$(conda shell.bash hook)"
conda activate torchEnv

# ---------- Weights & Biases (optional; OFF unless WANDB_PROJECT is set) ----------
# fairseq has native support: common.wandb_project wraps the progress bar and ships
# the SAME metrics as tensorboard. Auth comes from ~/.netrc (shared home => visible
# on the compute node). WANDB_MODE=offline writes to disk and you `wandb sync` later
# from the login node -- use that if the compute nodes have no outbound internet.
WANDB_PROJECT=${WANDB_PROJECT:-}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}          # run name in the UI
export WANDB_MODE=${WANDB_MODE:-online}             # online | offline
export WANDB_DIR=${WANDB_DIR:-$OUT_PATH}            # keeps wandb/ next to the run

# ---------- optional: stage data on node-local NVMe (huge I/O win on Node B) ----------
# VSC-5 A100 nodes have a local 2TB NVMe (typically $TMPDIR). Video I/O over shared
# storage has been your bottleneck, so staging helps a lot. OFF by default because it
# only works if your manifests reference paths UNDER $FINETUNE_DATA (so the rsync'd
# copy's paths line up). Enable with STAGE_DATA=true.
if [ "${STAGE_DATA}" = "true" ]; then
    LOCAL_DIR=${TMPDIR:-/tmp}/melvc_data
    echo "STAGE_DATA: copying $FINETUNE_DATA -> $LOCAL_DIR ..."
    mkdir -p "$LOCAL_DIR"
    rsync -a --info=progress2 "$FINETUNE_DATA/" "$LOCAL_DIR/"
    FINETUNE_DATA="$LOCAL_DIR"
    echo "STAGE_DATA: FINETUNE_DATA now $FINETUNE_DATA"
fi

# ---------- distributed / GPU ----------
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROCS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NPROCS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
fi
[ "${NPROCS_PER_NODE:-0}" -ge 1 ] || NPROCS_PER_NODE=1
NGPUS=$NPROCS_PER_NODE          # auto = 2 here; distributed_world_size follows
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

# ---------- Task-1 duration predictor + Task-2 content-warp knobs (default OFF) ----------
USE_CONTENT_XATTN=${USE_CONTENT_XATTN:-false}
CONTENT_INTERLEAVE_START=${CONTENT_INTERLEAVE_START:-odd}
USE_DUR_PRED=${USE_DUR_PRED:-false}
DUR_PRED_LAYERS=${DUR_PRED_LAYERS:-2}
DUR_R_MIN=${DUR_R_MIN:-0.5}
SOFTDTW_WEIGHT=${SOFTDTW_WEIGHT:-0.0}
SOFTDTW_GAMMA=${SOFTDTW_GAMMA:-0.1}
DUR_LOSS_WEIGHT=${DUR_LOSS_WEIGHT:-0.0}

echo "====== MelVC (VSC-5 Node B / A100) ======"
echo "RUN_NAME=$RUN_NAME  OUT=$OUT_PATH  GPUS=$NGPUS  DATA=$FINETUNE_DATA"
echo "BATCH=${BATCH_SIZE:-24}  UPDATE_FREQ=${UPDATE_FREQ:-2}  (effective = $NGPUS x ${BATCH_SIZE:-24} x ${UPDATE_FREQ:-2})"
echo "========================================="

# ============================================================
# RESUME mode (set RESUME=true): continue an interrupted run identically —
# restores model+optimizer+epoch and reuses the EXACT original overrides.
# (For the head-only salvage use FINETUNE_FROM_MODEL on the normal path instead;
#  that starts a fresh run from old weights with a clean optimizer.)
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
        | grep -v '^checkpoint\.restore_file' \
        | grep -v '^checkpoint\.finetune_from_model')
    LATEST_CKPT=$(ls -1 ${OUT_PATH}/checkpoints/checkpoint[0-9]*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: RESUME requested but no numbered checkpoint in ${OUT_PATH}/checkpoints"; exit 1
    fi
    echo "RESUME MODE: restoring from $LATEST_CKPT"

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
    +task.avhubert_top_n_trainable=${AVHUBERT_TOP_N:-0} \
    +task.avhubert_layernorm_trainable=${AVHUBERT_LN_TRAINABLE:-false} \
    +task.ogm_enabled=${OGM_ENABLED:-false} \
    +task.ogm_alpha=${OGM_ALPHA:-0.3} \
    +task.mel_head_trainable=${MEL_HEAD_TRAINABLE} \
    +task.number_of_synths=${NUMBER_OF_SYNTHS:-0} \
    +task.whisper_pretrained_path=${WHISPER_PRETRAINED_PATH} \
    task.noise_wav=${NOISE_WAV} \
    task.noise_prob=${NOISE_PROB} \
    dataset.batch_size=${BATCH_SIZE:-24} \
    dataset.num_workers=${NUM_WORKERS:-8} \
    dataset.max_tokens=4000 \
    dataset.required_batch_size_multiple=1 \
    dataset.valid_subset=valid \
    dataset.validate_interval=${VALIDATE_INTERVAL:-1} \
    dataset.validate_interval_updates=${VALIDATE_INTERVAL_UPDATES:-0} \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${ROOT}/src \
    common.fp16=false \
    common.bf16=true \
    common.seed=1 \
    common.log_interval=1 \
    common.empty_cache_freq=5 \
    common.tensorboard_logdir=${OUT_PATH}/tensorboard \
    checkpoint.save_dir=${OUT_PATH}/checkpoints \
    checkpoint.save_interval=${SAVE_INTERVAL:-1} \
    checkpoint.keep_last_epochs=${KEEP_LAST_EPOCHS:-1} \
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
    model.use_content_xattn=${USE_CONTENT_XATTN} \
    model.content_interleave_start=${CONTENT_INTERLEAVE_START} \
    model.use_duration_predictor=${USE_DUR_PRED} \
    model.dur_pred_layers=${DUR_PRED_LAYERS} \
    model.dur_r_min=${DUR_R_MIN} \
    criterion.softdtw_weight=${SOFTDTW_WEIGHT} \
    criterion.softdtw_gamma=${SOFTDTW_GAMMA} \
    criterion.dur_loss_weight=${DUR_LOSS_WEIGHT} \
    criterion.mel_loss_weight=${MEL_LOSS_WEIGHT:-1.0} \
    +criterion.ssim_weight=${SSIM_WEIGHT:-0.0} \
    +criterion.mse_weight=${MSE_WEIGHT:-0.0} \
    +criterion.gv_weight=${GV_WEIGHT:-0.0} \
    +model.use_discriminator=${USE_DISCRIMINATOR:-false} \
    +criterion.use_discriminator=${FORCE_DISC_ACTIVE:-false} \
    +criterion.disc_pretrain=${DISC_PRETRAIN:-true} \
    +criterion.disc_start_updates=${DISC_START_UPDATES:-0} \
    +criterion.adv_warmup_updates=${ADV_WARMUP_UPDATES:-0} \
    +criterion.adv_weight=${ADV_WEIGHT:-0.2} \
    +criterion.fm_weight=${FM_WEIGHT:-2.0} \
    +criterion.disc_lr=${DISC_LR:-2e-4} \
    +criterion.disc_beta1=${DISC_BETA1:-0.8} \
    +criterion.disc_beta2=${DISC_BETA2:-0.99} \
    +criterion.disc_grad_clip=${DISC_GRAD_CLIP:-20.0} \
    +criterion.freeze_disc=${FREEZE_DISC:-false} \
    ${WANDB_PROJECT:+common.wandb_project=$WANDB_PROJECT} \
    optimizer.weight_decay=${WEIGHT_DECAY:-0.01} \
    optimization.lr=[${LR:-3e-5}] \
    optimization.max_update=${MAX_UPDATE:-3000000} \
    optimization.max_epoch=${MAX_EPOCH:-20000} \
    optimization.update_freq=[${UPDATE_FREQ:-2}] \
    lr_scheduler.warmup_updates=${WARMUP_UPDATES:-2000} \
    distributed_training.distributed_world_size=${NGPUS} \
    $DIST_ARGS \
    distributed_training.nprocs_per_node=${NPROCS_PER_NODE} \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true
