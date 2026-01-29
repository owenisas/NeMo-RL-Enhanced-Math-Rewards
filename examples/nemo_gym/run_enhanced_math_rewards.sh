#!/bin/bash
# Launch script for Enhanced Math Rewards GRPO Training
# This script runs GRPO training with multi-component scoring and difficulty scaling

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}Enhanced Math Rewards GRPO Training${NC}"
echo -e "${GREEN}Multi-component scoring + Difficulty scaling${NC}"
echo -e "${GREEN}=====================================================${NC}"

# Check if config file exists
CONFIG_NAME="${1:-grpo_enhanced_math_rewards}"
CONFIG_FILE="$(pwd)/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}Usage: $0 [config_name]${NC}"
    echo -e "${YELLOW}Example: $0 grpo_enhanced_math_rewards${NC}"
    exit 1
fi

echo -e "${GREEN}Using config: ${CONFIG_NAME}${NC}"

# Parse optional overrides from command line
OVERRIDES=""
shift || true  # Remove first arg if exists
while [[ $# -gt 0 ]]; do
    OVERRIDES="$OVERRIDES $1"
    shift
done

# Display key settings
echo ""
echo -e "${YELLOW}Enhanced Features:${NC}"
echo "  ✓ Multi-component reward scoring"
echo "  ✓ Difficulty-based reward scaling"
echo "  ✓ LLM judge with JSON evaluation"
echo "  ✓ Trace logging to logs/training_traces/"
echo ""

# Check if data paths are set
echo -e "${YELLOW}Checking configuration...${NC}"
if grep -q "/path/to/" "$CONFIG_FILE"; then
    echo -e "${RED}Warning: Data paths not configured!${NC}"
    echo -e "${YELLOW}Please update the following in ${CONFIG_NAME}.yaml:${NC}"
    echo "  - data.train_jsonl_fpath"
    echo "  - data.validation_jsonl_fpath"
    echo ""
    echo -e "${YELLOW}Or use example data for testing:${NC}"
    echo "  ./run_enhanced_math_rewards.sh $CONFIG_NAME \\"
    echo "    ++data.train_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \\"
    echo "    ++data.validation_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs/training_traces
echo -e "${GREEN}✓ Created trace logging directory${NC}"

# Run training
echo ""
echo -e "${GREEN}Starting GRPO training...${NC}"
echo -e "${YELLOW}Command:${NC}"
echo "python run_grpo_nemo_gym.py --config-path=. --config-name=${CONFIG_NAME} ${OVERRIDES}"
echo ""

python run_grpo_nemo_gym.py \
    --config-path=. \
    --config-name="${CONFIG_NAME}" \
    ${OVERRIDES}

echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}Training complete!${NC}"
echo -e "${GREEN}Check logs/training_traces/ for detailed reward analysis${NC}"
echo -e "${GREEN}=====================================================${NC}"
