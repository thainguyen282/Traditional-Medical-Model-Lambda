# Exp0015 general evaluation on Slurm

Submit from the repository root. This evaluates an existing checkpoint; it does
not retrain it. The script requests account `phan`, QoS `standard`, partition
`gpu`, one full A100, 16 CPUs, 64 GB host RAM, and 24 hours. Override Slurm
options on the `sbatch` command if your allocation requires different resources.

## Full evaluation

```bash
sbatch scripts/slurm/eval_exp0015.sbatch --upload
```

Omit `--upload` to keep results only on the shared filesystem. Hugging Face
credentials and `HF_SFT_REPO` are read from the repository `.env`; credentials
are never passed in job arguments. Uploads use a separate evaluation branch
`eval-general-vi-exp0015-slurm-<job-id>` and do not modify the model branch.

A two-question-per-benchmark trial:

```bash
sbatch --export=ALL,EVAL_LIMIT=2 scripts/slurm/eval_exp0015.sbatch
```

The first job installs the existing pinned vLLM 0.31.0/CUDA 12.9 stack into
`.venv-vllm-slurm`, separate from training. Later jobs reuse it. Avoid simultaneous
first-time installations into that same environment. You can select an existing
compatible environment using `VLLM_ENV=/absolute/path/to/environment`.
The driver on the allocated node must support CUDA 12.9.

## Exact checkpoint and protocol

- SFT repository: `Maverfrick/Venera-Traditional-Model`
- Requested branch: `cpt-sft-unsloth-qwen3-4B-exp0015-tm-sft-1792332824`
- Pinned SFT commit: `06fc5e88287113fc29b1cdf236dd294260272daa`
- CPT source: `Maverfrick/Venera-CPT-TMModel@e0589a27e7b66bcbd9cfd8b709c0785d18bf0bf5`
- Underlying base: `unsloth/Qwen3-4B-Thinking-2507@fb56efbd0f60c5bcd531901fa5fe3ee3157b5135`

SFT continued the CPT LoRA adapter, rather than creating a second adapter.
Consequently, the final SFT adapter is applied **once** to the underlying Qwen
weights. Loading the CPT adapter first and applying SFT on top would double
count its CPT contribution. The adapter uses rank 64 and rsLoRA, which the pinned
[vLLM loader supports](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/lora/peft_helper.py).

The job reuses `eval_general_vi_vllm.py` for all four Vietnamese benchmarks:
MLogiQA (80), INCLUDE (550), MT-AIME24 (30), and PolyMath (500). Each benchmark
runs in a separate process to release GPU memory. It matches the prior thinking
run: BF16, thinking enabled, temperature 0.6, top-p 0.95, top-k 20, seed 3407,
4,096 input context allowance, 32,768 output tokens, up to eight active sequences,
70% GPU memory, and eager execution. Math uses the same pinned PolyMath scorer.
Length-limited outputs and thinking outputs without a closing thinking tag count
as invalid. This preserves the previous baseline scoring even though exp0015 was
fine-tuned on short `Final Answer: X` completions with thinking disabled.

## Progress and saved results

```bash
squeue -u "$USER"
tail -f slurm-exp0015-<job-id>.out
```

vLLM shows completed/total prompts for each benchmark. Results persist at:

```text
EVAL/outputs/general_vi_exp0015/slurm-<job-id>/
  evaluation.log
  comparison.csv
  comparison.md
  exp0015/<benchmark>/run_config.json
  exp0015/<benchmark>/responses.jsonl
  exp0015/<benchmark>/summary.json
  exp0015/<benchmark>/infer_result.txt
```

Comparison files are produced only when all four full evaluations succeed.
Failed benchmarks produce a nonzero job exit code, while completed outputs are
preserved. Limited trials skip comparison export. Logs also remain in the Slurm
output if environment setup or preflight fails. A rerun should use a fresh job
output directory. `OUTPUT_DIR`, `EVAL_MAX_NEW_TOKENS`, `EVAL_BATCH_SIZE`, and
`EVAL_GPU_MEMORY_UTILIZATION` can override defaults.

CPU-only preflight (requires the scorer dependencies in the selected environment):

```bash
.venv-vllm-slurm/bin/python scripts/run_exp0015_general_vi.py --preflight-only
```
