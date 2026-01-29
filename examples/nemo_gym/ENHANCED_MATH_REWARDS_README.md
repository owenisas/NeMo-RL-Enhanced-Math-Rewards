# Enhanced Math Rewards for NeMo GRPO Training

This enhanced version of the `math_with_judge` resource server adds sophisticated multi-component reward scoring and difficulty-based reward scaling for improved RL training on mathematical reasoning tasks.

## 🎯 Key Features

### 1. Multi-Component Reward Scoring
Instead of simple correct/incorrect rewards, the system evaluates multiple aspects:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Outcome** | 2.0 | Correctness of the final answer (scaled by difficulty) |
| **Format** | 0.1 | Use of `<think>` tags and `\boxed{}` formatting |
| **Axiom** | 0.1 | Usage of reasoning tags: `[Def]`, `[Axiom]`, `[Theorem]`, etc. |
| **Coherence** | 0.2 | Logical flow of reasoning (from LLM judge) |
| **Relevance** | 0.1 | Reasoning addresses the problem (from LLM judge) |
| **Efficiency** | 0.05 | Conciseness vs. unnecessary verbosity (from LLM judge) |

### 2. Difficulty-Based Reward Scaling
Hard problems yield higher rewards when solved correctly:
- Easy problem (solve_rate=1.0): 1.0x multiplier
- Medium problem (solve_rate=0.5): 2.0x multiplier
- Hard problem (solve_rate=0.25): 4.0x multiplier (capped)

### 3. Enhanced LLM Judge
Uses structured JSON evaluation for fine-grained scoring across all dimensions.

### 4. Trace Logging
Detailed logs of every generation and reward breakdown saved to JSONL for analysis.

## 🚀 Quick Start

### Basic Usage (Backward Compatible)
Your existing JSONL datasets work without modification:
```json
{"question": "What is 2+2?", "expected_answer": "4"}
```

Run with the enhanced config:
```bash
cd /Users/user/Documents/hf-AI/RL/examples/nemo_gym
chmod +x run_enhanced_math_rewards.sh
./run_enhanced_math_rewards.sh
```

### Advanced Usage (With Difficulty Scaling)
Add difficulty metadata to your dataset:
```json
{
  "question": "Solve the equation...",
  "expected_answer": "42",
  "llama8b_solve_rate": 0.3,
  "domain": "algebra",
  "source": "AMC12"
}
```

## 📝 Configuration

### Enable/Disable Features

Edit `grpo_enhanced_math_rewards.yaml`:

```yaml
env:
  nemo_gym:
    math_with_judge:
      resources_servers:
        math_with_judge:
          # Enable LLM judge for multi-component scoring
          should_use_judge: true
          
          # Adjust reward weights
          format_weight: 0.1
          outcome_weight: 2.0
          axiom_weight: 0.1
          coherence_weight: 0.2
          relevance_weight: 0.1
          efficiency_weight: 0.05
          
          # Difficulty scaling
          use_difficulty_scaling: true
          difficulty_scaling_cap: 4.0
          
          # Trace logging
          save_traces: true
          trace_dir: "logs/training_traces"
```

### Quick Testing with Example Data

```bash
./run_enhanced_math_rewards.sh grpo_enhanced_math_rewards \
  ++data.train_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \
  ++data.validation_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl
```

## 📊 Understanding Rewards

### Example Reward Calculation

For a correct answer on a hard problem (solve_rate=0.25):

```
Format reward:        0.05  (has <think> and \boxed{})
Outcome reward:       2.0   (correct answer, weight=2.0)
Difficulty multiplier: 4.0  (1.0 / 0.25)
Scaled outcome:       8.0   (2.0 × 4.0, capped at 4.0 = 8.0)
Axiom reward:         0.1   (used 3+ reasoning tags)
Coherence reward:     0.15  (0.75 score × 0.2 weight)
Relevance reward:     0.08  (0.8 score × 0.1 weight)
Efficiency reward:    0.02  ((0.9-0.5) × 2 × 0.05 weight)
─────────────────────────────
Total reward:         8.4
```

### Allowed Reasoning Tags

The following tags in brackets are rewarded:
- `[Def]` or `[Definition]` - Defining terms
- `[Axiom]` or `[Axioms]` - Fundamental assumptions
- `[Theorem]` - Theorems being applied
- `[Property]` - Mathematical properties
- `[Calculation]` - Computational steps
- `[Fact]` or `[Facts]` - Known mathematical facts

Example:
```
<think>
[Def] A prime number is only divisible by 1 and itself.
[Axiom] The fundamental theorem of arithmetic states...
[Calculation] 2 × 3 = 6
</think>

The answer is \boxed{6}
```

## 📈 Analyzing Training

Check trace logs for detailed reward breakdowns:
```bash
cd logs/training_traces
# View latest trace file
cat trace_*.jsonl | jq .
```

Each entry contains:
```json
{
  "question": "...",
  "response": "...",
  "ground_truth": "...",
  "reward": 8.4,
  "reward_components": {
    "format_reward": 0.05,
    "outcome_reward": 2.0,
    "scaled_outcome_reward": 8.0,
    "axiom_reward": 0.1,
    "coherence_reward": 0.15,
    "relevance_reward": 0.08,
    "efficiency_reward": 0.02,
    "difficulty_multiplier": 4.0
  },
  "difficulty": {
    "llama8b_solve_rate": 0.25,
    "difficulty_multiplier": 4.0,
    "domain": "algebra",
    "source": "AMC12"
  }
}
```

## 🔧 Troubleshooting

### Issue: Rewards seem too high/low
Adjust the weight parameters in the config file.

### Issue: Judge evaluations failing
- Check `should_use_judge: true` is set
- Ensure judge model is accessible
- Try lowering `judge_responses_create_params.max_new_tokens`

### Issue: Want to disable specific features
```yaml
should_use_judge: false          # Disable LLM judge
use_difficulty_scaling: false    # Disable difficulty scaling
save_traces: false               # Disable logging
```

## 📚 Dataset Preparation

### Option 1: Use Existing Datasets
Compatible with standard math RL datasets:
- OpenMathReasoning
- AIME problems
- GSM8K
- MATH dataset

### Option 2: Add Difficulty Metrics
Enhance your dataset with solve rates:
```python
import json

# Calculate solve rates from a baseline model
for item in dataset:
    item['llama8b_solve_rate'] = calculate_solve_rate(item['question'])
    item['domain'] = classify_domain(item['question'])
    item['source'] = 'custom'
```

## 🎓 Best Practices

1. **Start with judge disabled** for faster iterations
2. **Enable judge** for final training runs
3. **Monitor trace logs** to tune reward weights
4. **Use difficulty scaling** with curated datasets
5. **Adjust weights** based on your model's behavior

## 📄 Files Modified

- `resources_servers/math_with_judge/app.py` - Enhanced reward logic
- `examples/nemo_gym/grpo_enhanced_math_rewards.yaml` - Example config
- `examples/nemo_gym/run_enhanced_math_rewards.sh` - Launch script

## 🤝 Contributing

This is a fork of NVIDIA NeMo RL with enhanced math reward capabilities. See `VERIFICATION_SUMMARY.md` for implementation details and backward compatibility verification.

## 📝 License

Same as the original NVIDIA NeMo RL repository (Apache 2.0).
