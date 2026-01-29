# Setup Instructions - Enhanced Math Rewards Fork

## 📦 What Has Been Created

I've prepared everything for you to fork the NVIDIA NeMo RL repository with your enhanced math rewards:

### New Files Created

1. **`GITHUB_SETUP.sh`** - Automated setup script
   - Handles GitHub authentication
   - Forks the repo to your account
   - Makes it private
   - Commits and pushes your changes

2. **`examples/nemo_gym/grpo_enhanced_math_rewards.yaml`** - Example configuration
   - Enables all enhanced features
   - Multi-component reward weights configured
   - Difficulty scaling enabled
   - Trace logging enabled

3. **`examples/nemo_gym/run_enhanced_math_rewards.sh`** - Launch script
   - Easy command-line interface
   - Validates configuration
   - Creates log directories
   - Provides helpful output

4. **`examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md`** - Documentation
   - Feature overview
   - Quick start guide
   - Configuration examples
   - Troubleshooting tips

5. **`3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/VERIFICATION_SUMMARY.md`**
   - Verification test results
   - Backward compatibility proof
   - Implementation details

### Modified Files

1. **`3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/app.py`**
   - Enhanced with multi-component rewards
   - Difficulty scaling logic
   - JSON-based LLM judge
   - Trace logging

## 🚀 Quick Start

### Step 1: Run the Setup Script

```bash
cd /Users/user/Documents/hf-AI/RL
./GITHUB_SETUP.sh
```

This script will:
1. ✓ Authenticate you with GitHub (interactive)
2. ✓ Fork NVIDIA-NeMo/RL to your account as "NeMo-RL-Enhanced-Math-Rewards"
3. ✓ Make the repository private
4. ✓ Create a branch called "enhanced-math-rewards"
5. ✓ Commit all your changes
6. ✓ Push to your private fork

### Step 2: Verify the Setup

After the script completes, verify your fork:

```bash
# View your repo in browser
gh repo view $(gh api user | jq -r .login)/NeMo-RL-Enhanced-Math-Rewards --web

# Or check it from command line
gh repo view $(gh api user | jq -r .login)/NeMo-RL-Enhanced-Math-Rewards
```

### Step 3: Test the Enhanced Version

```bash
cd examples/nemo_gym

# Test with example data
./run_enhanced_math_rewards.sh grpo_enhanced_math_rewards \
  ++data.train_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \
  ++data.validation_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl
```

## 📋 Manual Setup (Alternative)

If you prefer to do it manually or the script fails:

```bash
cd /Users/user/Documents/hf-AI/RL

# 1. Authenticate
gh auth login -h github.com -p https -w

# 2. Fork the repo
gh repo fork NVIDIA-NeMo/RL \
  --fork-name "NeMo-RL-Enhanced-Math-Rewards" \
  --remote-name=myfork \
  --clone=false

# 3. Make it private
USERNAME=$(gh api user | jq -r .login)
gh repo edit "${USERNAME}/NeMo-RL-Enhanced-Math-Rewards" --visibility private

# 4. Add remote
git remote add myfork "https://github.com/${USERNAME}/NeMo-RL-Enhanced-Math-Rewards.git"

# 5. Create branch
git checkout -b enhanced-math-rewards

# 6. Stage changes
git add \
  3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/app.py \
  3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/VERIFICATION_SUMMARY.md \
  3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/test_app.py \
  3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/verify_import.py \
  examples/nemo_gym/grpo_enhanced_math_rewards.yaml \
  examples/nemo_gym/run_enhanced_math_rewards.sh \
  examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md \
  GITHUB_SETUP.sh \
  SETUP_INSTRUCTIONS.md

# 7. Commit
git commit -m "Add enhanced math rewards with multi-component scoring

Features:
- Multi-component reward scoring
- Difficulty-based reward scaling
- Enhanced LLM judge with JSON evaluation
- Trace logging for analysis
- Backward compatible with existing datasets"

# 8. Push
git push -u myfork enhanced-math-rewards
```

## 📊 What Gets Pushed to GitHub

Your private fork will include:

### Core Enhancement
- **Enhanced reward logic** (`app.py`) with 6 reward components
- **Backward compatible** - works with existing datasets

### Configuration & Scripts
- **Example config** with all features enabled
- **Launch script** for easy training runs
- **Comprehensive README** with usage examples

### Documentation
- **Verification summary** proving backward compatibility
- **Test scripts** for validation
- **Setup instructions** (this file)

### Original Codebase
- Complete NVIDIA NeMo RL framework
- All existing examples and configs
- Full documentation

## 🎯 Features Summary

### Multi-Component Rewards
| Component | Weight | What It Rewards |
|-----------|--------|----------------|
| Outcome | 2.0 | Correct final answer |
| Format | 0.1 | `<think>` tags & `\boxed{}` |
| Axiom | 0.1 | Reasoning tags (`[Def]`, `[Axiom]`, etc.) |
| Coherence | 0.2 | Logical reasoning flow |
| Relevance | 0.1 | Addressing the problem |
| Efficiency | 0.05 | Concise vs verbose |

### Difficulty Scaling
- Easy problem (solve_rate=1.0): 1.0x multiplier
- Medium problem (solve_rate=0.5): 2.0x multiplier  
- Hard problem (solve_rate=0.25): 4.0x multiplier

### Trace Logging
Every generation logged with full reward breakdown to `logs/training_traces/*.jsonl`

## 🔧 Customization

### Adjust Reward Weights

Edit `examples/nemo_gym/grpo_enhanced_math_rewards.yaml`:

```yaml
env:
  nemo_gym:
    math_with_judge:
      resources_servers:
        math_with_judge:
          format_weight: 0.1      # Adjust as needed
          outcome_weight: 2.0
          axiom_weight: 0.1
          coherence_weight: 0.2
          relevance_weight: 0.1
          efficiency_weight: 0.05
```

### Disable Features

```yaml
should_use_judge: false          # Disable LLM judge
use_difficulty_scaling: false    # Disable difficulty scaling
save_traces: false               # Disable logging
```

## 📚 Documentation

After pushing, read the detailed documentation:

```bash
# View the comprehensive README
cat examples/nemo_gym/ENHANCED_MATH_REWARDS_README.md

# View verification results
cat 3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/VERIFICATION_SUMMARY.md
```

## 🐛 Troubleshooting

### Issue: GitHub CLI authentication fails
```bash
# Re-authenticate
gh auth logout
gh auth login -h github.com -p https -w
```

### Issue: Fork already exists
```bash
# Use existing fork
USERNAME=$(gh api user | jq -r .login)
git remote add myfork "https://github.com/${USERNAME}/NeMo-RL-Enhanced-Math-Rewards.git"
```

### Issue: Need to update the fork
```bash
# Push new changes
git add -A
git commit -m "Update: description of changes"
git push myfork enhanced-math-rewards
```

## 🎓 Next Steps

1. **Run the setup script**: `./GITHUB_SETUP.sh`
2. **Test with example data**: Follow the quick start guide
3. **Prepare your dataset**: Add difficulty metrics if desired
4. **Configure training**: Adjust weights and parameters
5. **Start training**: Run with your full dataset

## 📝 Repository Info

- **Visibility**: Private (not visible to others)
- **Branch**: `enhanced-math-rewards`
- **Base**: NVIDIA-NeMo/RL (forked)
- **License**: Apache 2.0 (same as original)

## 💡 Tips

- Keep the fork private if using proprietary data
- The original NeMo RL updates won't auto-sync (it's a fork)
- You can merge upstream changes if needed
- All changes are backward compatible with the original

## ✅ Checklist

Before starting training:
- [ ] Run `GITHUB_SETUP.sh` successfully
- [ ] Verify fork is private on GitHub
- [ ] Test with example data
- [ ] Update data paths in config
- [ ] Adjust reward weights if needed
- [ ] Run a short test training run

---

**Questions?** Check the README files or review the verification summary for detailed implementation info.
