# Qwen3.5-9B Agent LoRA

This folder is independent from the original agent implementation. It converts the
strict positive SWE-smith trajectories into step-level tool-calling SFT data and trains
an adapter for `Qwen/Qwen3.5-9B` with ms-swift.

## Data policy

Only trajectories satisfying both conditions are exported:

```text
agent.status == "succeeded"
evaluation.resolved == true
```

Each exact model request/response pair becomes one SFT sample. The split is performed
by `instance_id`, so steps from one trajectory can never leak across train and eval.
`data/index.jsonl` maps every sample back to its source trajectory and model call.
Training uses `last_round` loss so only the final teacher decision in each step-level
sample is supervised; historical assistant messages are context rather than duplicate
targets.

Regenerate and validate the data from the project root:

```bash
trajectory_experiments/.venv/bin/python qwen35_9b_lora/prepare_data.py
trajectory_experiments/.venv/bin/python qwen35_9b_lora/validate_data.py
```

## Environment

Use a Linux CUDA machine with Python 3.12. Install the PyTorch build matching that
machine first, then install the training stack:

```bash
python3.12 -m venv qwen35_9b_lora/.venv
qwen35_9b_lora/.venv/bin/python -m pip install -U pip
qwen35_9b_lora/.venv/bin/python -m pip install -r qwen35_9b_lora/requirements.txt
```

For faster Qwen3.5 training, the official ms-swift recipe additionally recommends
`flash-attn` and `flash-linear-attention>=0.4.2`; install them only after the CUDA PyTorch
environment is working. Set `ATTN_IMPL=flash_attention_2` when those kernels are ready.

## Train

Recommended multi-GPU BF16 LoRA starting point:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
DEEPSPEED_CONFIG=zero3 \
MAX_LENGTH=16384 \
bash qwen35_9b_lora/train_lora.sh
```

The default context length retains approximately 99% of the teacher decision points;
increase `MAX_LENGTH=24576` if GPU memory allows. LoRA is applied only to the language
model's linear layers; the vision tower and aligner stay frozen because this dataset is
text-only.

Useful overrides include `EPOCHS`, `LEARNING_RATE`, `GRAD_ACCUM`, `LORA_RANK`,
`LORA_ALPHA`, `OUTPUT_DIR`, and `ATTN_IMPL`. To resume with optimizer state:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N \
bash qwen35_9b_lora/train_lora.sh
```

QLoRA is available as a lower-memory fallback after installing
`requirements-qlora.txt`, but BF16 LoRA is preferred when hardware permits.

## Test and merge

```bash
bash qwen35_9b_lora/infer_adapter.sh /path/to/checkpoint-N
bash qwen35_9b_lora/merge_lora.sh /path/to/checkpoint-N
```

Always compare the adapter against the untouched base model on held-out SWE-smith tasks;
training/eval loss alone does not measure coding-agent resolve rate.
