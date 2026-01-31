# Dataset Guide for Enhanced Math Rewards

This guide explains how to prepare datasets for training with the enhanced math rewards system.

## Quick Start

### Option 1: Test with Built-in Examples (Fastest)
No download needed! Use the included example data:

```bash
./run_enhanced_math_rewards.sh grpo_enhanced_math_rewards_2xH200 \
  ++data.train_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \
  ++data.validation_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl
```

### Option 2: Download Big-Math-RL-Verified (Recommended)
This dataset has difficulty metrics built-in for enhanced rewards:

```bash
# Install dependencies
pip install datasets

# Download and convert to JSONL
cd /Users/user/Documents/hf-AI/RL/examples/nemo_gym
python download_bigmath_dataset.py --output-dir data/bigmath

# This creates:
# - data/bigmath/train.jsonl
# - data/bigmath/validation.jsonl

# Then run training:
./run_enhanced_math_rewards.sh grpo_enhanced_math_rewards_2xH200 \
  ++data.train_jsonl_fpath=data/bigmath/train.jsonl \
  ++data.validation_jsonl_fpath=data/bigmath/validation.jsonl
```

### Option 3: Use Your Own Dataset
See "Dataset Format" section below.

## Dataset Format

### Minimum Required Format (Backward Compatible)
```jsonl
{"question": "What is 2+2?", "expected_answer": "4"}
{"question": "Solve x^2 = 16", "expected_answer": "4 or -4"}
```

**Fields:**
- `question` (required): The problem to solve
- `expected_answer` (required): The correct answer

### Enhanced Format (With Difficulty Scaling)
```jsonl
{
  "question": "Solve the equation...",
  "expected_answer": "42",
  "llama8b_solve_rate": 0.3,
  "domain": "algebra",
  "source": "AMC12"
}
```

**Additional Fields:**
- `llama8b_solve_rate` (optional, float 0-1): 
  - How often a baseline model (e.g., Llama 3 8B) solves this correctly
  - 0.0 = very hard (gets 4x reward multiplier)
  - 0.5 = medium (gets 2x reward multiplier)
  - 1.0 = easy (gets 1x reward multiplier)
- `domain` (optional, string): Category like "algebra", "geometry", "number_theory"
- `source` (optional, string): Where it came from like "AMC12", "AIME", "synthetic"

## Available Datasets

### 1. Big-Math-RL-Verified (Best for Enhanced Rewards)
- **Source**: HuggingFace `SynthLabsAI/Big-Math-RL-Verified`
- **Size**: Large collection of verified math problems
- **Features**: Includes difficulty metrics
- **Download**: Use `download_bigmath_dataset.py`

### 2. OpenMathReasoning (NVIDIA Official)
- **Location**: Already included in repo
- **Path**: `resources_servers/math_with_judge/data/train.jsonl`
- **Features**: Curated by NVIDIA for Nemotron training
- **Note**: May not have difficulty metrics (will use default 0.5)

### 3. AIME Problems
- **Location**: Included for validation
- **Path**: `resources_servers/math_with_judge/data/aime24_validation.jsonl`
- **Features**: Competition-level problems
- **Use**: Good for validation/testing

### 4. Custom Dataset
Create your own JSONL file with the format above.

## Calculating Difficulty Metrics

If you have a dataset without `llama8b_solve_rate`, you can calculate it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load a baseline model (e.g., Llama 3 8B)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Test each problem
with open("input.jsonl") as f_in, open("output.jsonl", "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        
        # Generate answer
        response = generate_answer(model, tokenizer, item["question"])
        
        # Check if correct
        is_correct = check_answer(response, item["expected_answer"])
        
        # Add to dataset (you'd average over multiple runs for accuracy)
        item["llama8b_solve_rate"] = 1.0 if is_correct else 0.0
        
        f_out.write(json.dumps(item) + "\n")
```

Or use a simpler heuristic:
- **Easy problems**: `llama8b_solve_rate: 0.8-1.0`
- **Medium problems**: `llama8b_solve_rate: 0.3-0.7`
- **Hard problems**: `llama8b_solve_rate: 0.0-0.3`

## Dataset Size Recommendations

### For 2x H200 GPUs

| Purpose | Train Samples | Val Samples | Training Time |
|---------|---------------|-------------|---------------|
| **Quick Test** | 100-1,000 | 50-100 | Hours |
| **Research** | 10,000-50,000 | 500-1,000 | Days |
| **Production** | 100,000+ | 2,000-5,000 | Weeks-Months |

### For 256 GPUs

| Purpose | Train Samples | Val Samples | Training Time |
|---------|---------------|-------------|---------------|
| **Quick Test** | 1,000-10,000 | 100-500 | Hours |
| **Research** | 50,000-200,000 | 1,000-5,000 | Days |
| **Production** | 500,000+ | 5,000-10,000 | Days-Weeks |

## Download Script Options

```bash
# Download full dataset
python download_bigmath_dataset.py

# Download limited samples for testing
python download_bigmath_dataset.py --max-train 1000 --max-val 100

# Custom output directory
python download_bigmath_dataset.py --output-dir /path/to/data
```

## Validating Your Dataset

After creating/downloading, validate the format:

```bash
# Check first few lines
head -n 3 data/bigmath/train.jsonl | jq .

# Count samples
wc -l data/bigmath/train.jsonl

# Verify all fields are present
cat data/bigmath/train.jsonl | jq -c 'select(.question == null or .expected_answer == null)' | wc -l
# Should output 0 (no missing fields)
```

## Troubleshooting

### Error: "File not found"
Make sure paths are absolute or relative to where you run the command:

```bash
# Absolute path (recommended)
++data.train_jsonl_fpath=/full/path/to/train.jsonl

# Relative path (from RL/examples/nemo_gym/)
++data.train_jsonl_fpath=data/bigmath/train.jsonl
```

### Error: "Invalid JSON"
Each line must be valid JSON. Common issues:
- Missing quotes around strings
- Trailing commas
- Newlines inside strings (use `\n`)

Test each line:
```bash
cat yourfile.jsonl | while read line; do echo "$line" | jq . > /dev/null || echo "Bad line: $line"; done
```

### Dataset too large for memory
Use the `--max-train` and `--max-val` flags to limit size during testing.

## Best Practices

1. **Start small**: Test with 100-1,000 samples first
2. **Validate format**: Check a few samples manually
3. **Check diversity**: Ensure variety in difficulty and domains
4. **Split properly**: Keep validation set representative
5. **Document sources**: Track where problems came from

## Data Augmentation (Optional)

To increase dataset size:
- Rephrase questions (keep same answer)
- Generate similar problems
- Add step-by-step solutions
- Include multiple solution paths

## Example Datasets by Domain

### Algebra
```jsonl
{"question": "Solve 2x + 5 = 15", "expected_answer": "5", "domain": "algebra", "llama8b_solve_rate": 0.9}
```

### Geometry  
```jsonl
{"question": "Find the area of a circle with radius 5", "expected_answer": "25π", "domain": "geometry", "llama8b_solve_rate": 0.7}
```

### Number Theory
```jsonl
{"question": "Find all prime factors of 84", "expected_answer": "2, 2, 3, 7", "domain": "number_theory", "llama8b_solve_rate": 0.4}
```

---

**Questions?** Check `ENHANCED_MATH_REWARDS_README.md` for more details on using the enhanced reward system.
