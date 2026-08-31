#!/bin/bash
# ============================================================
# Resample GTA targets to 22050 Hz for BigVGAN (SLURM, CPU-only, SINGLE CORE).
#
# Single core is not a compromise — it is the fast path. The per-file resample is
# tiny, so torch's default thread count (=cores/2) thrashes on sync overhead:
# 842ms/file at 88 threads vs 5.4ms/file at 1 thread (~156x). The script pins
# torch to 1 thread; the remaining cost is writing ~115 GB, which is I/O-bound
# and gains nothing from more cores.
#
# Submit:   sbatch scripts/resample_targets_22k.sh
# Override: RESAMPLE_DATA=/path/to/manifest_dir sbatch scripts/resample_targets_22k.sh
# ============================================================
#SBATCH --job-name=melvc_resample22k
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -p zen4_0768
#SBATCH --qos zen4_0768
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/melvc_resample22k_%j.log
#SBATCH --error=/tmp/melvc_resample22k_%j.log
set -e

ROOT=/data/fs201163/eo49197/MultiModalVC
cd "$ROOT"

# Log to SHARED storage so it is tail-able live (the #SBATCH /tmp path above is
# node-local private on this cluster, unreadable from the login node).
mkdir -p "$ROOT/exp/eval-logs"
exec >> "$ROOT/exp/eval-logs/melvc_resample22k_${SLURM_JOBID}.log" 2>&1

eval "$(conda shell.bash hook)"
conda activate torchEnv

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"
# Belt-and-braces: matches torch.set_num_threads(1) in the script, and also pins
# the BLAS/OMP pools that numpy/soundfile could otherwise spin up.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Dedicated var name (not generic DATA) so a variable exported in the submitting
# login shell can't leak in via SLURM and override this.
RESAMPLE_DATA=${RESAMPLE_DATA:-/data/fs201163/eo49197/VoiceConversion-fwf/pre-trainVoxcelebYoutube}

echo "================ resampling targets in $RESAMPLE_DATA ================"
python scripts/resample_targets_22k.py --data "$RESAMPLE_DATA"

echo "RESAMPLE DONE"
