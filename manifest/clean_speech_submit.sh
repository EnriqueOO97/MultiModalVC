#!/bin/bash
# ============================================================
# Musica Cluster Slurm Submission Script — Clean Speech Inference
# Runs 4 workers in parallel, one per H100 GPU.
# ============================================================
#ASC --vanilla
#SBATCH --job-name=clean_speech
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --time=10:00:00
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log

# -------------------------------------------------------------
# 1. Setup paths and log directory
# -------------------------------------------------------------
PROJECT_ROOT=/data/fs201163/eo49197/MultiModalVC
LOG_DIR=$PROJECT_ROOT/logs/clean_speech
mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/job_${SLURM_JOBID}_out.log" 2>> "$LOG_DIR/job_${SLURM_JOBID}_err.log"

# -------------------------------------------------------------
# 2. Environment (mirrors musica_submit.sh exactly)
# -------------------------------------------------------------
module reload
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate torchEnv

# Auto-detect and append CUDA libraries (fixing version discovery)
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

echo "====== Clean Speech Inference ======"
echo "Job ID:  $SLURM_JOBID"
echo "Node:    $(hostname)"
echo "Log dir: $LOG_DIR"
echo "======================================"

# -------------------------------------------------------------
# 3. Launch 4 workers (one per GPU) as background processes.
#    Each worker logs to its own file.
# -------------------------------------------------------------
for RANK in 0 1 2 3; do
    python $PROJECT_ROOT/scripts/clean_speech_dataset.py \
        --rank $RANK \
        --world-size 4 \
        >> "$LOG_DIR/worker_${RANK}_out.log" 2>> "$LOG_DIR/worker_${RANK}_err.log" &
done

wait
echo "All 4 workers finished."
