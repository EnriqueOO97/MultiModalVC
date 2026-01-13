#! /bin/bash
# ...existing code...

ROOT=$(pwd)
SRC_PTH=$ROOT/src

# --- USER CONFIGURATION ---
# Paths from demo.sh
CHECKPOINT_PATH="$ROOT/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"
SR_PREDICTOR_PATH="$ROOT/pretrained_models/sr_predictor/checkpoint.pt"
LLM_PATH="meta-llama/Llama-3.2-3B"

# Manifest configuration
MANIFEST_DIR="$ROOT/manifest/germanManifest"
MANIFEST_NAME="test"

# Output directory
OUT_PATH="$ROOT/results/german_eval"
mkdir -p $OUT_PATH

# --- ENVIRONMENT SETUP ---
export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"

# Auto-detect and append CUDA libraries (from demo.sh)
if [ -n "$CONDA_PREFIX" ]; then
    CUSPARSE_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib"
    TORCH_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib"
    if [ -d "$CUSPARSE_LIB" ]; then
        export LD_LIBRARY_PATH="$CUSPARSE_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"
    fi
fi

echo "Running Evaluation on German Dataset..."
echo "Manifest Dir: $MANIFEST_DIR"
echo "Model: $CHECKPOINT_PATH"

# Run Evaluation
CUDA_VISIBLE_DEVICES=0 python -B $SRC_PTH/eval.py --config-dir ${SRC_PTH}/conf --config-name s2s_decode \
    dataset.gen_subset=$MANIFEST_NAME \
    common.user_dir=${SRC_PTH} \
    generation.beam=1 \
    generation.temperature=0.3 \
    override.llm_path=${LLM_PATH} \
    dataset.max_tokens=10000 \
    dataset.num_workers=8 \
    override.modalities=['video','audio'] \
    common_eval.path=${CHECKPOINT_PATH} \
    common_eval.results_path=${OUT_PATH} \
    override.label_dir=$MANIFEST_DIR \
    override.data=$MANIFEST_DIR \
    override.noise_wav=$ROOT/noise/babble_noise.wav \
    override.noise_prob=0