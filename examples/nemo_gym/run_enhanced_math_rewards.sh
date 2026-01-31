cat << 'EOF' > run_nano.sh
#!/bin/bash
set -e

# --- CONFIGURATION ---
export HF_TOKEN="your_hf_token_here" # <--- ADD YOUR TOKEN HERE
export WANDB_API_KEY="your_wandb_key_here" # <--- ADD YOUR WANDB KEY HERE
export MODEL_CHECKPOINT="owenisas/nemotron-3-nano-reasoning"
export BASE_DIR="/workspace"
export DATA_DIR="$BASE_DIR/data"
export CODE_DIR="$BASE_DIR/RL"
export LOCAL_MODEL_DIR="$BASE_DIR/model_weights"

# --- FIX 2: VERSION MISMATCH & VENV REBUILD ---
# Forces Ray to rebuild environments with our updated Gym code
export NRL_FORCE_REBUILD_VENVS=true
export NRL_IGNORE_VERSION_MISMATCH=1

# --- FIX 1: REDIRECT RAY & CACHE (Fixes Disk Full Crash) ---
# We force Ray to write temp files to your large disk, not the small boot disk.
export RAY_TMPDIR="$BASE_DIR/ray_tmp"
export TMPDIR="$BASE_DIR/tmp"
export HF_HOME="$BASE_DIR/hf_cache"
export NEMO_CACHE_DIR="$BASE_DIR/nemo_cache"

mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$HF_HOME" "$NEMO_CACHE_DIR" "$LOCAL_MODEL_DIR"

echo ">>> [1/5] Installing Dependencies..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

apt-get update && apt-get install -y git

echo ">>> [2/5] Setting up Codebase..."
if [ ! -d "$CODE_DIR" ]; then
    # Clone your enhanced fork instead of the original repo
    git clone -b enhanced-math-rewards https://github.com/owenisas/NeMo-RL-Enhanced-Math-Rewards.git "$CODE_DIR"
fi
# Use -C to avoid changing the terminal's working directory globally
git -C "$CODE_DIR" submodule update --init --recursive

echo ">>> [2.5/5] Pre-downloading Model Weights..."
# Use huggingface-cli to ensure weights are fully downloaded locally
# This fixes the Megatron-Bridge "No .safetensors found" error
uv run huggingface-cli download "$MODEL_CHECKPOINT" --local-dir "$LOCAL_MODEL_DIR" --token "$HF_TOKEN"

echo ">>> [3/5] Downloading Data..."
mkdir -p "$DATA_DIR"
# Install datasets for the download script using uvx for global access
uvx pip install datasets

# Use absolute paths for everything to ensure it works from any directory
python3 "$CODE_DIR/examples/nemo_gym/download_bigmath_dataset.py" --output-dir "$DATA_DIR/bigmath"

# Use the processed Big-Math files
TRAIN_FILE="$DATA_DIR/bigmath/train.jsonl"
VAL_FILE="$DATA_DIR/bigmath/validation.jsonl"

# Check for placeholders
if [[ "$HF_TOKEN" == "your_hf_token_here" || "$WANDB_API_KEY" == "your_wandb_key_here" ]]; then
    echo -e "${RED}Error: You must set your HF_TOKEN and WANDB_API_KEY in the script configuration section.${NC}"
    echo -e "${YELLOW}1. Get HF Token: https://huggingface.co/settings/tokens${NC}"
    echo -e "${YELLOW}2. Get W&B Key: https://wandb.ai/authorize${NC}"
    echo ""
    echo -e "${YELLOW}To run without W&B, set logger.wandb_enabled=False at the end of this script.${NC}"
    exit 1
fi

# Log in to W&B to ensure session is active
uv run wandb login "$WANDB_API_KEY"

echo ">>> [4/5] Launching Training (H200 Optimized)..."

# Run everything using absolute paths
# POINT POLICY.MODEL_NAME TO THE LOCAL DIRECTORY
uv run "$CODE_DIR/examples/nemo_gym/run_grpo_nemo_gym.py" \
    --config "$CODE_DIR/examples/nemo_gym/grpo_enhanced_math_rewards_2xH200.yaml" \
    data.train_jsonl_fpath="$TRAIN_FILE" \
    data.validation_jsonl_fpath="$VAL_FILE" \
    policy.model_name="$LOCAL_MODEL_DIR" \
    logger.wandb_enabled=True
EOF

# Run it
chmod +x run_nano.sh
./run_nano.sh
