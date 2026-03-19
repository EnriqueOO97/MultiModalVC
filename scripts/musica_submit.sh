#!/bin/bash
# ============================================================
# Musica Cluster Slurm Submission Script (4 H100 GPUs)
# ============================================================
#ASC --vanilla
#SBATCH --job-name=mms_musica
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88               # Request all CPUs for the node
#SBATCH --time=72:00:00                 # Max limit on Musica
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log

set -e

# -------------------------------------------------------------
# 1. Compute OUT_PATH early
# -------------------------------------------------------------
RUN_NAME=${RUN_NAME:-default_run}
PROJECT_ROOT=/data/fs201163/eo49197/MultiModalVC
OUT_PATH=$PROJECT_ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH/logs"

# Redirect this node's stdout and stderr into the per-run logs directory.
# (Matching original script naming convention)
exec >> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_out_${SLURM_JOBID}.log" 2>> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_err_${SLURM_JOBID}.log"

# Robust Conda Initialization for Scripts
# (Note: conda init only works for future terminals, not the current script)
module reload
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate torchEnv

# Discover Master IP for distributed training
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "====== Musica Submission ======"
echo "Head Node IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Output Directory: $OUT_PATH"
echo "=============================="

# Launch training via srun (will run musica_train.sh 4 times per node)
srun bash $PROJECT_ROOT/scripts/musica_train.sh
