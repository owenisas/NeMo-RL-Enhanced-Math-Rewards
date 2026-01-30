# Configuration Comparison: 32-Node vs 2x H200

This document explains the optimizations made for the 2x H200 GPU configuration.

## Hardware Comparison

| Setting | Original (32 nodes) | 2x H200 | Reason |
|---------|---------------------|---------|--------|
| **Total GPUs** | 256 GPUs (32×8) | 2 GPUs | Hardware constraint |
| **GPU Memory** | ~640 GB total | ~282 GB (2×141GB) | H200 has 141GB HBM3 |
| **Compute** | Massive parallelism | Limited parallelism | Must optimize for efficiency |

## Key Configuration Changes

### 1. Cluster Settings
```yaml
# Original
cluster:
  gpus_per_node: 8
  num_nodes: 32

# 2x H200
cluster:
  gpus_per_node: 2
  num_nodes: 1
```

### 2. Batch Sizes (Most Important)
```yaml
# Original (256 GPUs)
grpo:
  num_prompts_per_step: 128
  num_generations_per_prompt: 16
  val_batch_size: 256
policy:
  train_global_batch_size: 2048
  generation_batch_size: 64

# 2x H200 (2 GPUs)
grpo:
  num_prompts_per_step: 16      # ÷8 = ~128 GPUs worth
  num_generations_per_prompt: 8  # ÷2 to fit memory
  val_batch_size: 32             # ÷8 = ~16 GPUs worth
policy:
  train_global_batch_size: 128   # ÷16 = ~8 GPUs worth
  generation_batch_size: 16      # ÷4 = ~4 GPUs worth
```

**Effective batch size ratio**: ~1:128 (2 GPUs vs 256 GPUs)

### 3. Sequence Length
```yaml
# Original
max_total_sequence_length: 49152  # Very long sequences

# 2x H200
max_total_sequence_length: 16384  # ⅓ length to fit memory
```
**Why**: Longer sequences require exponentially more memory. 16K is still very good for math problems.

### 4. Parallelism Strategy

#### Original (256 GPUs - Heavy Parallelism)
```yaml
tensor_model_parallel_size: 2      # Split model across 2 GPUs
expert_model_parallel_size: 8      # Split MoE experts across 8 GPUs
pipeline_model_parallel_size: 2    # 2-stage pipeline
context_parallel_size: 4           # Split context across 4 GPUs
sequence_parallel: true
```

#### 2x H200 (Minimal Parallelism)
```yaml
tensor_model_parallel_size: 1      # No tensor parallelism
expert_model_parallel_size: 1      # All experts on same GPU
pipeline_model_parallel_size: 2    # Use both GPUs in pipeline
context_parallel_size: 1           # No context parallelism
sequence_parallel: false           # Disabled (requires TP>1)
```

**Strategy**: Use **pipeline parallelism** to split the model across 2 GPUs vertically.

### 5. Memory Optimizations

```yaml
# 2x H200 Optimizations
megatron_cfg:
  empty_unused_memory_level: 2       # Increased from 1 (more aggressive cleanup)
  activation_checkpointing: true     # Keep enabled (saves memory)
  moe_per_layer_logging: false       # Reduced logging overhead

generation:
  vllm_cfg:
    gpu_memory_utilization: 0.85     # Increased from 0.5 (H200 has more memory)
    tensor_parallel_size: 2          # Use both GPUs for generation
```

### 6. Checkpointing
```yaml
# Original
checkpointing:
  keep_top_k: 1000000  # Keep all checkpoints
  save_period: 10      # Save every 10 steps

# 2x H200
checkpointing:
  keep_top_k: 5        # Keep only top 5 (save disk space)
  save_period: 20      # Save every 20 steps (less I/O overhead)
```

### 7. Training Speed Adjustments
```yaml
# 2x H200
grpo:
  val_period: 10  # Increased from 5 (validate less frequently)

data:
  num_workers: 2  # More CPU threads for data loading
```

## Performance Expectations

### Training Speed
- **Original**: ~128 samples/step × 256 GPUs = 32,768 samples/step
- **2x H200**: ~16 samples/step × 2 GPUs = 128 samples/step
- **Ratio**: ~256x slower (expected for 128x fewer GPUs)

### Memory Usage Per GPU
- **Original**: ~60-70% utilization per GPU
- **2x H200**: ~85-90% utilization (more efficient use)

### Convergence
- Should converge to similar final performance
- Will take ~256x more wall-clock time
- Gradients will be noisier (smaller batch size)

## When to Use Each Config

### Use `grpo_enhanced_math_rewards.yaml` (full config) when:
- ✅ You have access to 32+ nodes (256+ GPUs)
- ✅ Training time is critical
- ✅ You need maximum throughput
- ✅ Budget allows for large-scale training

### Use `grpo_enhanced_math_rewards_2xH200.yaml` when:
- ✅ You have 1-2 H200 GPUs (or similar high-memory GPUs)
- ✅ You're doing research/experimentation
- ✅ You want to verify the setup works before scaling up
- ✅ Budget is limited

### Use `grpo_nanov3.yaml` (original) when:
- ✅ You want the original reward system (no enhancements)
- ✅ You have the full 32-node cluster

## Quick Start for 2x H200

```bash
cd /Users/user/Documents/hf-AI/RL/examples/nemo_gym

# Update model path and data paths in config
# Then run:
./run_enhanced_math_rewards.sh grpo_enhanced_math_rewards_2xH200 \
  ++policy.model_name=/path/to/your/model \
  ++data.train_jsonl_fpath=/path/to/train.jsonl \
  ++data.validation_jsonl_fpath=/path/to/val.jsonl
```

## Further Optimizations (If Needed)

If you still run out of memory, try these in order:

1. **Reduce sequence length further**:
   ```yaml
   max_total_sequence_length: 8192  # Half of 16384
   ```

2. **Reduce generations per prompt**:
   ```yaml
   num_generations_per_prompt: 4  # Half of 8
   ```

3. **Enable gradient checkpointing more aggressively** (slower but saves memory)

4. **Reduce logprob chunk size**:
   ```yaml
   logprob_chunk_size: 1024  # Half of 2048
   ```

5. **Consider CPU offloading** (much slower):
   ```yaml
   optimizer_cpu_offload: True
   cpu_offload: True
   ```

## Monitoring Your Run

```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Watch training logs
tail -f logs_2xH200/*.log

# Check reward traces
ls -lh logs/training_traces_2xH200/
```

## Expected Timeline

With 2x H200 GPUs:
- **Warmup**: ~10-20 minutes
- **Per step**: ~30-60 seconds
- **1000 steps**: ~8-16 hours
- **Full training (100K steps)**: ~1-2 months

Compare to 256 GPUs:
- **Per step**: ~5-10 seconds
- **1000 steps**: ~1-2 hours
- **Full training (100K steps)**: ~3-5 days

## Cost Comparison

Assuming H200 cloud pricing (~$5/GPU/hour):
- **2x H200 for 1 month**: ~$7,200
- **256 GPUs for 5 days**: ~$153,600

**Savings**: ~95% cost reduction, but 30x longer training time.

---

**Recommendation**: Use the 2x H200 config for initial experiments and debugging, then scale up to the full cluster for production training runs.
