#!/bin/bash
# GitHub Setup Script for Enhanced Math Rewards Fork
# This script will fork the NVIDIA NeMo RL repo and push your changes

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GitHub Setup - Enhanced Math Rewards${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Authenticate with GitHub
echo -e "${YELLOW}Step 1: Authenticate with GitHub CLI${NC}"
echo "You'll be prompted to log in to GitHub..."
echo ""
gh auth login -h github.com -p https -w

echo ""
echo -e "${GREEN}✓ Authentication successful!${NC}"
echo ""

# Step 2: Fork the repository
echo -e "${YELLOW}Step 2: Forking NVIDIA-NeMo/RL repository${NC}"
echo "Creating a private fork under your account..."
echo ""

cd /Users/user/Documents/hf-AI/RL

# Fork the repo with a new name and make it private
gh repo fork NVIDIA-NeMo/RL \
    --fork-name "NeMo-RL-Enhanced-Math-Rewards" \
    --remote-name=myfork \
    --clone=false

echo ""
echo -e "${GREEN}✓ Fork created: $(gh api user | jq -r .login)/NeMo-RL-Enhanced-Math-Rewards${NC}"
echo ""

# Step 3: Make the repo private
echo -e "${YELLOW}Step 3: Making repository private${NC}"
USERNAME=$(gh api user | jq -r .login)
gh repo edit "${USERNAME}/NeMo-RL-Enhanced-Math-Rewards" --visibility private

echo -e "${GREEN}✓ Repository is now private${NC}"
echo ""

# Step 4: Add the fork as a remote
echo -e "${YELLOW}Step 4: Adding fork as git remote${NC}"
git remote add myfork "https://github.com/${USERNAME}/NeMo-RL-Enhanced-Math-Rewards.git" 2>/dev/null || echo "Remote already exists"

echo -e "${GREEN}✓ Remote added${NC}"
echo ""

# Step 5: Create branch and commit changes
echo -e "${YELLOW}Step 5: Creating branch and committing changes${NC}"
git checkout -b enhanced-math-rewards || git checkout enhanced-math-rewards

# First, commit changes in the Gym submodule
echo "Committing changes in Gym submodule..."
cd 3rdparty/Gym-workspace/Gym
git add resources_servers/math_with_judge/app.py \
        resources_servers/math_with_judge/VERIFICATION_SUMMARY.md \
        resources_servers/math_with_judge/test_app.py \
        resources_servers/math_with_judge/verify_import.py

git commit -m "Enhance math_with_judge with multi-component rewards and difficulty scaling

- Add format_reward, axiom_reward, coherence_reward, relevance_reward, efficiency_reward
- Implement difficulty scaling based on llama8b_solve_rate
- Enhanced LLM judge with JSON multi-score output
- Add trace logging for training analysis
- Backward compatible with existing JSONL data format

All tests passing. Full verification in VERIFICATION_SUMMARY.md"

echo -e "${GREEN}✓ Gym submodule changes committed${NC}"
cd /Users/user/Documents/hf-AI/RL

# Now commit in the main repo
echo ""
echo "Committing changes in main repo..."
git add GITHUB_SETUP.sh \
        SETUP_INSTRUCTIONS.md \
        examples/nemo_gym/grpo_enhanced_math_rewards.yaml \
        examples/nemo_gym/run_enhanced_math_rewards.sh \
        examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md \
        docs/guides/nemotron-3-nano.md \
        3rdparty/Gym-workspace/Gym

echo ""
echo "Files to be committed:"
git status --short
echo ""

read -p "Proceed with commit? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted by user${NC}"
    exit 1
fi

git commit -m "Add enhanced math rewards configuration and documentation

Features:
- Multi-component reward scoring system
- Difficulty-based reward scaling
- Enhanced LLM judge with structured evaluation
- Comprehensive documentation and examples
- Easy-to-use launch script

Files added:
- examples/nemo_gym/grpo_enhanced_math_rewards.yaml (example config)
- examples/nemo_gym/run_enhanced_math_rewards.sh (launch script)
- examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md (user guide)
- GITHUB_SETUP.sh (automated setup)
- SETUP_INSTRUCTIONS.md (setup guide)

Updated Gym submodule with enhanced math_with_judge server.
Full backward compatibility maintained."

echo ""
echo -e "${GREEN}✓ Main repo changes committed${NC}"
echo ""

# Step 6: Push to your fork
echo -e "${YELLOW}Step 6: Pushing to your private fork${NC}"
git push -u myfork enhanced-math-rewards

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Your enhanced version is now on GitHub:${NC}"
echo "https://github.com/${USERNAME}/NeMo-RL-Enhanced-Math-Rewards"
echo ""
echo -e "${BLUE}Branch:${NC} enhanced-math-rewards"
echo -e "${BLUE}Visibility:${NC} Private"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. View your repo: gh repo view ${USERNAME}/NeMo-RL-Enhanced-Math-Rewards --web"
echo "2. Test the enhanced version: cd examples/nemo_gym && ./run_enhanced_math_rewards.sh"
echo "3. Read the docs: cat examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md"
echo ""
