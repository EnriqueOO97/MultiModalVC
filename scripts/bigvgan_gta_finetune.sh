#!/bin/bash
# ============================================================
# BigVGAN GTA fine-tuning on the P2H predicted mels (SLURM, single H100).
#
# Uses BigVGAN's OWN train.py --fine_tuning (precomputed mels from disk). The
# acoustic model is frozen (mels already dumped); BigVGAN's generator + MPD + MRD
# + optimizer are trainable. Resumes from the released checkpoint via train.py's
# native HF-filename fallback (bigvgan_generator.pt / bigvgan_discriminator_optimizer.pt).
#
# Recipe = the released config.json (lr/betas/lr_decay/segment/arch untouched).
# Batch size (verified from BigVGAN README L212-214): the released v2 checkpoints
# used GLOBAL batch_size=32 over 8 A100 GPUs (=4/GPU); the ./configs ship 4 as the
# "fits a single A100" default. train.py treats config batch_size as GLOBAL and
# divides by num_gpus, so on ONE GPU the device sees exactly the config value.
# BATCH=32 reproduces the recipe's global batch (no BatchNorm -> 1x32 is optimizer-
# equivalent to 8x4) but is 8x the per-device load the README says fits one A100,
# so it MAY OOM on a single H100 -> fall back to 16 or 8 if so. Else unchanged.
#
# Submit:   sbatch scripts/bigvgan_gta_finetune.sh
# Override: RUN_NAME=myrun BATCH=32 sbatch scripts/bigvgan_gta_finetune.sh
# ============================================================
#SBATCH --job-name=bigvgan_gta
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/bigvgan_gta_%j.log
#SBATCH --error=/tmp/bigvgan_gta_%j.log
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
BIGVGAN=/data/fs201163/eo49197/BigVGAN
PRETRAINED=$ROOT/pretrained_models/BigVGAN
GTA_BASE=$ROOT/exp/bigvganGTA                     # all GTA runs live here

# ---- per-run knobs ----
# Run name: 1st positional arg wins, else $RUN_NAME env, else the default.
#   sbatch scripts/bigvgan_gta_finetune.sh my_run_name
RUN_NAME=${1:-${RUN_NAME:-p2h_fullopen_aug_reg_ck120}}  # source P2H model these mels came from
BATCH=${BATCH:-12}                                # GLOBAL batch (train.py divides by num_gpus). With
                                                  # --gres=gpu:2 -> 6/GPU (~67GB/card, the size verified to fit;
                                                  # ~10-11GB/sample for the v2 alias-free training graph).
P2H_DATA=${P2H_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT}
# Checkpoint + validation cadence: by default ONE EPOCH (computed below from the
# train filelist size and BATCH). train.py only supports step-based intervals
# (steps % interval == 0), so per-epoch == interval = ceil(N_train / BATCH).
# Override CKPT_INTERVAL/VAL_INTERVAL to force a fixed step cadence instead.

RUN_DIR=$GTA_BASE/$RUN_NAME
FILELISTS=$RUN_DIR/filelists
mkdir -p "$RUN_DIR" "$FILELISTS" "$RUN_DIR/logs"

# Log INSIDE this run's folder (so each GTA run owns its own logs) and on SHARED
# storage (the #SBATCH /tmp path is node-local/unreadable here).
exec >> "$RUN_DIR/logs/bigvgan_gta_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv
export PYTHONPATH="$BIGVGAN:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
# Reduce allocator fragmentation (batch 32 OOM'd on a 93GB H100 -> use BATCH<=16).
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Safety margin for the DDP validation deadlock: validation runs on rank 0 only,
# so rank 1 waits on the next collective for the whole validation. The seen-set is
# capped below (N_VAL) to keep that well under the default 480s NCCL watchdog, but
# raise the watchdog too as cheap insurance against a slow node / IO stall.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}

# ---------- Weights & Biases (optional; OFF unless WANDB_PROJECT is set) ----------
# Same convention as melvc-NODEB-warp.sh. train.py mirrors the SAME scalars it
# already sends to tensorboard -- logging only, no effect on training or ckpts.
# Auth comes from ~/.netrc (shared home => visible on the compute node).
# WANDB_MODE=offline writes to disk; `wandb sync` it from the login node later
# if the compute nodes have no outbound internet (test: scripts/test_wandb_node.sh).
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}          # run name in the UI
export WANDB_MODE=${WANDB_MODE:-online}             # online | offline
export WANDB_DIR=${WANDB_DIR:-$RUN_DIR}             # keeps wandb/ next to the run
# Stable id per run name -> resubmissions (24h wall clock) continue ONE wandb run
# instead of starting a new one each time.
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}

# 1) Filelists (absolute "-22k" stems; verifies every resampled wav exists).
python "$ROOT/scripts/make_bigvgan_filelists.py" --data "$P2H_DATA" --out-dir "$FILELISTS"

# 1a-cap) Cap the SEEN validation set. With 2 GPUs, validation runs on rank 0
# only while rank 1 waits on the next collective; if that wait exceeds the NCCL
# watchdog (480s) rank 1 aborts and kills the job. Full 11k val set took ~46min
# on rank 0 -> deadlock. A few hundred-to-1k clips give an identical vocoder
# metric curve in ~4min. Deterministic seed => the SAME sample every resubmission
# (so the metric curve is consistent across resumes). Overridable via N_VAL.
N_VAL=${N_VAL:-1000}
python - "$FILELISTS/val.txt" "$N_VAL" <<'PY'
import random, sys
path, n = sys.argv[1], int(sys.argv[2])
lines = [l for l in open(path).read().splitlines() if l.strip()]
total = len(lines)
if total > n:
    random.seed(1234)                 # fixed -> stable sample across resubmissions
    lines = sorted(random.sample(lines, n))
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"[val-cap] capped seen validation to {n} clips (of {total})")
else:
    print(f"[val-cap] {len(lines)} val clips <= N_VAL={n}; no cap applied")
PY

# 1a) train.py REQUIRES a non-empty unseen-validation list, but a 2nd full pass
#     over our val set is pure waste (identical to the seen pass). Use a single
#     throwaway clip so the unseen validation is trivial.
head -n 1 "$FILELISTS/val.txt" > "$FILELISTS/unseen_tiny.txt"

# 1b) Per-epoch cadence: steps_per_epoch = ceil(N_train / BATCH).
N_TRAIN=$(wc -l < "$FILELISTS/train.txt")
STEPS_PER_EPOCH=$(( (N_TRAIN + BATCH - 1) / BATCH ))
CKPT_INTERVAL=${CKPT_INTERVAL:-$STEPS_PER_EPOCH}
VAL_INTERVAL=${VAL_INTERVAL:-$STEPS_PER_EPOCH}
echo "[cadence] N_train=$N_TRAIN BATCH=$BATCH -> steps_per_epoch=$STEPS_PER_EPOCH "\
     "(ckpt_interval=$CKPT_INTERVAL val_interval=$VAL_INTERVAL)"

# 2) Per-run config = released config.json with batch_size overridden (canonical
#    config untouched; each run records its own setting).
# Loss weights: default to the released config's values (empty env -> unchanged).
# MEL_WEIGHT=0 + FM_WEIGHT=0 -> adversarial-only (perceptual GTA on misaligned pairs).
python - "$PRETRAINED/config.json" "$RUN_DIR/config.json" "$BATCH" "${MEL_WEIGHT:-}" "${FM_WEIGHT:-}" "${USE_PERCEPTUAL:-}" "${LAMBDA_PERC:-}" "${PERC_BACKEND:-}" <<'PY'
import json, sys
src, dst, batch, mel_w, fm_w, use_perc, lam_perc, perc_backend = sys.argv[1:9]
with open(src) as f: cfg = json.load(f)
cfg["batch_size"] = int(batch)
if mel_w != "": cfg["lambda_melloss"] = float(mel_w)
if fm_w  != "": cfg["lambda_fm"]      = float(fm_w)
# Perceptual (neural-MOS) loss (train.py reads these; default OFF -> stock behavior).
# perc_backend: "dnsmos" (default) | "utmos" -- always eval with the OTHER one.
if use_perc     != "": cfg["use_perceptual_loss"] = (use_perc.lower() == "true")
if lam_perc     != "": cfg["lambda_perc"]         = float(lam_perc)
if perc_backend != "": cfg["perc_backend"]        = perc_backend.lower()
with open(dst, "w") as f: json.dump(cfg, f, indent=2)
print(f"[config] {dst}: batch_size={batch} lambda_melloss={cfg.get('lambda_melloss')} "
      f"lambda_fm={cfg.get('lambda_fm', '(default 1.0)')} "
      f"use_perceptual_loss={cfg.get('use_perceptual_loss', False)} "
      f"perc_backend={cfg.get('perc_backend', 'dnsmos')} "
      f"lambda_perc={cfg.get('lambda_perc', 0.0)} lr={cfg.get('learning_rate')} "
      f"segment={cfg.get('segment_size')} fmax={cfg.get('fmax')}")
PY

# 3) Seed the run dir with the released checkpoint so train.py's HF-filename
#    fallback resumes from it on the FIRST launch. cp -n: never clobber the
#    g_/do_ checkpoints written by later/continued runs.
cp -n "$PRETRAINED/bigvgan_generator.pt"                 "$RUN_DIR/" || true
cp -n "$PRETRAINED/bigvgan_discriminator_optimizer.pt"  "$RUN_DIR/" || true

# 4) Train. Absolute filelist entries -> input_wavs_dir/input_mels_dir are unused
#    (mels are read co-located via the patched loader); pass "/" as placeholders.
#    Unseen-validation list reuses our val set so train.py doesn't look for the
#    LibriTTS defaults.
cd "$BIGVGAN"
python train.py \
    --config "$RUN_DIR/config.json" \
    --checkpoint_path "$RUN_DIR" \
    --fine_tuning True \
    --input_wavs_dir / \
    --input_mels_dir / \
    --input_training_file "$FILELISTS/train.txt" \
    --input_validation_file "$FILELISTS/val.txt" \
    --list_input_unseen_wavs_dir / \
    --list_input_unseen_validation_file "$FILELISTS/unseen_tiny.txt" \
    --checkpoint_interval "$CKPT_INTERVAL" \
    --validation_interval "$VAL_INTERVAL" \
    --summary_interval 10 \
    --stdout_interval 10 \
    --eval_subsample "${EVAL_SUBSAMPLE:-100}"   # log media for ~1/100 val clips -> small event files
                                                # (metrics still use the FULL val set; this only gates audio/image logging)

echo "BIGVGAN GTA DONE"
