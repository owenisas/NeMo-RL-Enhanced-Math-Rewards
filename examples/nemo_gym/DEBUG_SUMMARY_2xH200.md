# Debug Summary: Enhanced Math Rewards on 2× H200

## What Ran Successfully (End-to-End, Before OOM)

The following **completed without error** on 2× H200:

1. **Config & data**
   - Loaded `grpo_enhanced_math_rewards_2xH200.yaml` with overrides.
   - Data: `/workspace/data/bigmath/train.jsonl` (250,122 samples), `validation.jsonl` (1,000 samples).
   - JSONL rows include: `agent_ref`, `responses_create_params`, `question`, `expected_answer`, `llama8b_solve_rate`, `domain`, `source`.

2. **Cluster & workers**
   - Ray cluster: 1 node, 2 GPUs.
   - vLLM generation worker: initialized, model loaded, HTTP server up.
   - Megatron policy workers: 2 ranks (pipeline parallel), model + reference loaded from `/workspace/model_weights`.

3. **NeMo-Gym & math_with_judge**
   - All 13 servers up (including `math_with_judge`, `math_with_judge_simple_agent`).
   - `math_with_judge` no longer crashed after fixes (trace_file → _trace_file, domain list/string, judge disabled for temperature issue).

4. **Step 1 rollout (with judge disabled)**
   - Batch prep, generation for 128 samples, **rollout collection 100% (128/128)**.
   - **Rewards processed**, **advantages computed**.
   - Then run failed with **out-of-memory** in the next phase (see below).

So the **script and pipeline ran successfully** through: env init → data → generation → rollout → reward → advantages. Only the **training step** (logprobs + optimizer) hit OOM.

---

## Where It Went Out of Memory (2× H200)

OOM occurred in **two different places** depending on run:

### 1. During logprob computation (first runs)

- **Location:**  
  `MegatronPolicyWorker.train()` → loss / logprob path →  
  `from_parallel_logits_to_logprobs` → `DistributedLogprob` →  
  `_compute_distributed_log_softmax` → `vocab_parallel_logits.exp().sum(...)`  
- **Error:**  
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.32 GiB. GPU 0 has a total capacity of 139.80 GiB of which 5.94 GiB is free.`
- **Meaning:**  
  After rollout and advantage computation, the **forward + logprob backward** for the policy needed an extra ~7.32 GiB and there wasn’t enough contiguous GPU memory left.

### 2. During optimizer step (later run with smaller batch)

- **Location:**  
  `MegatronPolicyWorker.train()` → `optimizer.step()` →  
  Transformer Engine fused Adam → `initialize_state(..., "exp_avg_sq", ...)` →  
  `torch.empty_like(param, dtype=dtype)`  
- **Error:**  
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 139.80 GiB of which 19.62 MiB is free. ... 130.83 GiB is allocated by PyTorch`
- **Meaning:**  
  Logprobs/loss step had run, but when **optimizer state** (e.g. second moment) was being allocated on GPU, almost no memory was left (~20 MiB requested, ~19.62 MiB free). So OOM happened during **optimizer step**, not during logprobs.

In both cases the **limiting factor was GPU memory on the policy (Megatron) workers** during the **training phase** (logprobs + optimizer), not during data load, generation, or rollout.

---

## Fixes Applied During Debug (For Reference)

| Issue | Fix |
|-------|-----|
| `math_with_judge` crash: `"object has no field \"trace_file\""` | Use private attribute `_trace_file` in `app.py` (Pydantic model). |
| `KeyError: 'responses_create_params'` | Add `responses_create_params.input` to each JSONL row in data generation. |
| `KeyError: 'agent_ref'` | Add `agent_ref: { "name": "math_with_judge_simple_agent" }` to each row. |
| `422 Unprocessable Entity` on `/verify`: `domain` "Input should be a valid string" | Allow `domain: Union[str, List[str]]` in `LibraryJudgeMathRunRequest` and `_verify_answer`. |
| Judge path: `assert request.temperature == generation_config["temperature"]` | Disabled judge (`should_use_judge: false`) so rollout uses only library (math-verify) reward; avoids vLLM temperature mismatch. |
| OOM during training | Reduced batch/seq and tried optimizer CPU offload; 2× H200 still OOM. For 4× H200, scale up instead (see below). |

---

## What to Adjust When Upgrading to 4× H200

With **4× H200** you have ~4× the GPU memory for the policy. You can:

### 1. Cluster and parallelism

- **Cluster:**  
  - Either `gpus_per_node: 4` and `num_nodes: 1`, or keep `gpus_per_node: 2` and `num_nodes: 2` (if multi-node).
- **Policy (Megatron):**  
  - Increase pipeline/tensor parallel so the 30B model is spread over 4 GPUs, e.g.  
    - `pipeline_model_parallel_size: 4`, `tensor_model_parallel_size: 1`, or  
    - `pipeline_model_parallel_size: 2`, `tensor_model_parallel_size: 2`.
- **Generation (vLLM):**  
  - Set `vllm_cfg.tensor_parallel_size: 4` so inference uses all 4 GPUs.

### 2. Batch and sequence length (increase from 2× H200 settings)

Current 2× H200 “last attempted” values (before OOM):

- `num_prompts_per_step: 4`
- `num_generations_per_prompt: 4`
- `train_global_batch_size: 32`
- `max_total_sequence_length: 8192`
- `max_new_tokens: 4096` (or tied to `max_total_sequence_length`)
- `optimizer_cpu_offload: true`, `optimizer_offload_fraction: 1.0`

For **4× H200** you can try (and tune from here):

- `num_prompts_per_step`: 16–32  
- `num_generations_per_prompt`: 8  
- `train_global_batch_size`: 128–256  
- `max_total_sequence_length`: 16384  
- `max_new_tokens`: 8192–16384  
- **Turn off CPU offload:** `optimizer_cpu_offload: false`, `optimizer_offload_fraction: 0` (optional; only if you still hit OOM with offload on, try with it off and smaller batch first).

### 3. Judge (optional)

- If you want the LLM judge again: set `should_use_judge: true`.
- The vLLM worker currently enforces a single temperature; either keep judge temperature equal to generation temperature or change the worker/API so judge requests can use a different temperature without triggering the assertion.

### 4. Config file to edit (4× H200)

- **File:** `examples/nemo_gym/grpo_enhanced_math_rewards_2xH200.yaml` (or a copy, e.g. `grpo_enhanced_math_rewards_4xH200.yaml`).
- **Sections:**  
  - `cluster` (gpus/nodes),  
  - `policy.megatron_cfg` (pipeline/tensor parallel, optimizer offload),  
  - `policy.generation.vllm_cfg.tensor_parallel_size`,  
  - `grpo` (num_prompts_per_step, num_generations_per_prompt, etc.),  
  - `policy` (train_global_batch_size, max_total_sequence_length, max_new_tokens).

---

## Short Recap

- **Script ran successfully** through: data load → cluster → vLLM + Megatron init → NeMo-Gym (all servers, math_with_judge) → Step 1 rollout (128/128) → rewards → advantages.  
- **OOM happened** in the **training step**: either in **logprob** computation or in the **optimizer step** (allocating optimizer state on GPU).  
- For **4× H200**: scale cluster and parallelism to 4 GPUs, then **increase** batch sizes and sequence lengths, and consider **disabling** optimizer CPU offload; optionally re-enable the judge once temperature is handled.
