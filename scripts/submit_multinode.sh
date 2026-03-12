#!/bin/bash
# ============================================================
# SLURM Multi-Node Submit Script
# ============================================================
# To change the number of nodes, just edit the --nodes line.
# The training script auto-detects the node count from SLURM.
#
# Usage:
#   sbatch scripts/submit_multinode.sh          # 2 nodes (default)
#   sbatch scripts/submit_multinode.sh           # edit --nodes for more
# ============================================================

#SBATCH --job-name=mms_e2e_multinode
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --nodes=2                       # <-- CHANGE THIS for more nodes
#SBATCH --gres=gpu:2                    # 2 GPUs per node (A100)
#SBATCH --ntasks-per-node=1             # 1 launcher per node (fairseq spawns GPU processes)
#SBATCH --time=72:00:00                 # Safety: max 24h (adjust as needed)
# Note: logs are redirected at runtime (see below) so they land inside
# the per-run OUT_PATH on the $DATA filesystem.
#SBATCH --output=/tmp/slurm_%j_out.log
#SBATCH --error=/tmp/slurm_%j_err.log

# Exit immediately if any command fails
set -e

# If anything goes wrong, make sure to clean up
trap 'echo "ERROR: Script failed. Releasing SLURM allocation."; exit 1' ERR

# -------------------------------------------------------------
# 1. Compute OUT_PATH early (mirrors logic in the training script)
#    so we can redirect logs into the right directory.
# -------------------------------------------------------------
_RUN_NAME=${RUN_NAME:-default_run}
OUT_PATH=/gpfs/data/fs72969/enriqueoo97/exp/mms-speech-NoLLM-E2E-SynthVC/$_RUN_NAME
mkdir -p "$OUT_PATH/logs"

# Redirect this node's stdout and stderr into the per-run logs directory.
# Each node writes its own file using SLURM_NODEID.
exec >> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_out_${SLURM_JOBID}.log" 2>> "$OUT_PATH/logs/node${SLURM_NODEID:-0}_err_${SLURM_JOBID}.log"

# -------------------------------------------------------------
# 2. Activate the Python environment
# -------------------------------------------------------------
source /gpfs/data/fs72969/enriqueoo97/mms-llama-l/bin/activate

# -------------------------------------------------------------
# 3. Discover the Master node IP
# -------------------------------------------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_JOB_NUM_NODES"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$head_node_ip

# -------------------------------------------------------------
# 4. NCCL Configuration
# -------------------------------------------------------------
export NCCL_IB_DISABLE=0

# -------------------------------------------------------------
# 5. Launch training on all nodes
# -------------------------------------------------------------
srun bash /home/fs72969/enriqueoo97/MultiModalVC/scripts/fineTuneSpeechNoLLM_E2E_SynthVC_multinode.sh
