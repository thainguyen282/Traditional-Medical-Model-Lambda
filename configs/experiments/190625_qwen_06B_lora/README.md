# Example 
# Qwen 0.6B LoRA Baseline

## Purpose
Establish baseline metrics for LoRA tuning before hyperparameter search.

## Key Configurations
```yaml
learning_rate: 2e-4
batch_size: 4
lora_rank: 32
```

## Observations
- VRAM usage: 12.3GB
- Training stability: Good (no NaN detected)
- Convergence after ~500 steps

## Next Steps
- [ ] Sweep learning rates (1e-5 to 5e-5)
- [ ] Test different LoRA ranks