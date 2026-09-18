# Exp0014 throughput settings

Both SFT and CPT-SFT variants use the following starting settings for a 40 GB
GPU. These settings have not been benchmarked on the target GPU.

| Stage | Train batch/GPU | Accumulation | Effective batch on one GPU | Eval batch/GPU |
| --- | ---: | ---: | ---: | ---: |
| QA-20K | 16 (was 2) | 1 (was 2) | 16 (was 4) | 8 |
| YHCT MCQ | 16 (was 8) | 4 (was 8) | 64 (unchanged) | 8 |

Stage 1 now has approximately one quarter as many optimizer updates per epoch.
Its warmup is 25 updates (was 100), and evaluation/checkpointing is every 50
updates (was 200), preserving their approximate positions in examples seen.
Epoch count, learning rate, data splits, and sequence length are unchanged.
The larger effective batch changes stage 1's training dynamics; compare final
validation and benchmark results before treating it as equivalent to old runs.
The historical experiment report describes the earlier configuration.

The trainer now honors `per_device_eval_batch_size` and `save_total_limit`.
Previously those YAML fields were ignored; evaluation used the trainer's default
batch size of 8. Exp0014 retains two checkpoints through the configured limit.

## Run

On the training GPU, from the repository root, run the stages sequentially:

```bash
# CPT -> QA-20K -> YHCT
python main.py --config configs/experiments/unsloth-qwen3-4B/exp0014/stage1_qa20k/config_cpt.yaml && \
python main.py --config configs/experiments/unsloth-qwen3-4B/exp0014/config_cpt.yaml
```

For SFT-only, replace both `config_cpt.yaml` filenames with `config.yaml`.
The existing Lambda config lists still run these stages in the correct order;
remote launches need these changes in their checkout.
Configuration changes take effect on the next launch, not in a running process.

## Check throughput and memory

Monitor `nvidia-smi -l 1` on the training host, including an evaluation pass.
Compare examples/second after compilation/warmup and across representative batch
lengths. Full VRAM occupancy is not itself a performance target; leave headroom
for longer padded batches and evaluation. The current trainer loads onto GPU 0.

If batch 16 runs out of memory:

- Stage 1: use batch 8, accumulation 2 to retain the new effective batch of 16
  and the new schedule.
- Stage 2: use batch 8, accumulation 8 to retain effective batch 64.
- If evaluation runs out of memory, lower its batch independently to 4 or 2.

If stage 2 still has ample memory, try batch 32 with accumulation 2 and compare
throughput; it retains effective batch 64. To preserve the original stage 1
optimization schedule instead, use batch 4, accumulation 1, warmup 100, and
both eval/save intervals 200.
