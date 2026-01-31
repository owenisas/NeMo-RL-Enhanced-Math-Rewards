cat << 'EOF' > run_nano.sh
#!/bin/bash
set -e

# --- CONFIGURATION ---
export WANDB_API_KEY="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"
export MODEL_CHECKPOINT="owenisas/nemotron-3-nano-reasoning"
export BASE_DIR="/workspace"
export DATA_DIR="$BASE_DIR/data"
export CODE_DIR="$BASE_DIR/RL"

# --- FIX 1: REDIRECT RAY & CACHE (Fixes Disk Full Crash) ---
# We force Ray to write temp files to your large disk, not the small boot disk.
export RAY_TMPDIR="$BASE_DIR/ray_tmp"
export TMPDIR="$BASE_DIR/tmp"
export HF_HOME="$BASE_DIR/hf_cache"
export NEMO_CACHE_DIR="$BASE_DIR/nemo_cache"

mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$HF_HOME" "$NEMO_CACHE_DIR"

echo ">>> [0/5] Cleaning up corrupted sessions..."
# Clean up previous crashed sessions to free ports and memory
# Use timeout to prevent hanging on large /tmp directories
timeout 10s pkill -9 -f "ray" || true
timeout 10s pkill -9 -f "python" || true
timeout 15s rm -rf /tmp/ray/* || echo "Ray cleanup timed out, continuing anyway..."

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

echo ">>> [3/5] Downloading Data..."
mkdir -p "$DATA_DIR"
# Install datasets for the download script using uvx for global access
uvx pip install datasets

# Use absolute paths for everything to ensure it works from any directory
python3 "$CODE_DIR/examples/nemo_gym/download_bigmath_dataset.py" --output-dir "$DATA_DIR/bigmath"

# Use the processed Big-Math files
TRAIN_FILE="$DATA_DIR/bigmath/train.jsonl"
VAL_FILE="$DATA_DIR/bigmath/validation.jsonl"

echo ">>> [4/5] Launching Training (H200 Optimized)..."

# Run everything using absolute paths
uv run "$CODE_DIR/examples/nemo_gym/run_grpo_nemo_gym.py" \
    --config "$CODE_DIR/examples/nemo_gym/grpo_enhanced_math_rewards_2xH200.yaml" \
    data.train_jsonl_fpath="$TRAIN_FILE" \
    data.validation_jsonl_fpath="$VAL_FILE" \
    policy.model_name="$MODEL_CHECKPOINT" \
    logger.wandb_enabled=True
EOF

# Run it
chmod +x run_nano.sh
./run_nano.sh