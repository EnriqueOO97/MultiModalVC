#! /bin/bash

# Get the root directory of the repo
ROOT=$(pwd)
SRC_PTH=$ROOT/src

# --- USER CONFIGURATION ---
# Adjust these paths to match your setup
CHECKPOINT_PATH="$ROOT/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"
SR_PREDICTOR_PATH="$ROOT/pretrained_models/sr_predictor/ckpt-srp.pt"
LLM_PATH="meta-llama/Llama-3.2-3B"

# Input files
# NOTE: These paths are specific to your machine. 
# For a PR, you might want to leave these empty or point to a sample file included in the repo.
AUDIO_PATH="/ceph/shared/ALL/datasets/voxceleb2-V2/VoxCeleb2-German/dev/processedVideos/vox2_german/vox2_german_video_seg16s/id01869/8DSA1QWxYjA/00008_00.wav"
VIDEO_PATH="/ceph/shared/ALL/datasets/voxceleb2-V2/VoxCeleb2-German/dev/processedVideos/vox2_german/vox2_german_video_seg16s/id01869/8DSA1QWxYjA/00008_00.mp4"
# --------------------------

export PYTHONPATH="$ROOT/fairseq:$ROOT:$PYTHONPATH"

# 2. Auto-detect and append CUDA libraries to LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ]; then
    echo "Detected Conda Env: $CONDA_PREFIX"
    
    # Get the site-packages path
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "Detected Site Packages: $SITE_PACKAGES"

    # --- UPDATED LOGIC FOR YOUR SCREENSHOT ---
    # We search recursively inside 'nvidia' because your files are in 'nvidia/cu13/lib'
    # This finds the full path to libcusparse.so.12 (or similar) and gets its folder
    CUSPARSE_PATH=$(find "$SITE_PACKAGES/nvidia" -name "libcusparse.so*" | head -n 1)
    
    if [ -n "$CUSPARSE_PATH" ]; then
        CUSPARSE_DIR=$(dirname "$CUSPARSE_PATH")
        export LD_LIBRARY_PATH="$CUSPARSE_DIR:$LD_LIBRARY_PATH"
        echo "Found CUDA libraries at: $CUSPARSE_DIR"
        echo "Added to LD_LIBRARY_PATH"
    else
        echo "WARNING: Could not find libcusparse.so* anywhere inside $SITE_PACKAGES/nvidia"
    fi

    # Check for Torch Libs (usually standard)
    TORCH_LIB="$SITE_PACKAGES/torch/lib"
    if [ -d "$TORCH_LIB" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
        echo "Added torch/lib to LD_LIBRARY_PATH"
    fi
fi

# -------------------------

echo "Running Inference..."
echo "Audio: $AUDIO_PATH"
echo "Video: $VIDEO_PATH"

python $SRC_PTH/demo.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --llm-path "$LLM_PATH" \
    --sr-predictor-path "$SR_PREDICTOR_PATH" \
    --audio-path "$AUDIO_PATH" \
    --video-path "$VIDEO_PATH" \
    --instruction "Recognize this speech in English. Input : " \
    --beam-size 5