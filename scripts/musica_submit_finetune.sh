#!/bin/bash
# ============================================================
# SLURM submission for the pathological -> healthy fine-tune stage
# Mirrors musica_submit.sh but invokes musica_train_finetune.sh
# ============================================================
#ASC --vanilla
#SBATCH --job-name=mms_finetune
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log

set -e

RUN_NAME=${RUN_NAME:-pathological_finetune_default}
PROJECT_ROOT=/data/fs201163/eo49197/MultiModalVC
OUT_PATH=$PROJECT_ROOT/exp/multiModalVC-synth/$RUN_NAME
mkdir -p "$OUT_PATH/logs"

exec >> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_out_${SLURM_JOBID}.log" 2>> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_err_${SLURM_JOBID}.log"

eval "$(conda shell.bash hook)"
conda activate torchEnv

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "====== Pathological Finetune Submission ======"
echo "Head Node IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Output Directory: $OUT_PATH"
echo "Phase 2 ckpt: ${PHASE2_CKPT:-<default>}"
echo "Manifest dir: ${FINETUNE_DATA:-<default>}"
echo "=============================================="

srun bash $PROJECT_ROOT/scripts/musica_train_finetune.sh
