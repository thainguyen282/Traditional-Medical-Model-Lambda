# Exp0015: CPT followed by YHCT MCQ SFT

`config_cpt.yaml` runs raw-text CPT. `config_sft.yaml` continues the saved CPT
LoRA adapter on `configs/dataset/venera_yhct_mcq.yaml`, using exp0013's SFT
recipe: all cleaned YHCT MCQs, 5% validation, three epochs, learning rate
5e-5, batch size 8 with accumulation 8, and NEFTune alpha 5.

The SFT stage inherits the adapter's saved rank, alpha, and target modules.
It requires an adapter and does not initialize a replacement LoRA adapter.
The existing data formatter trains `Final Answer: X` completions without
reasoning, using the same question/options instruction as the evaluation
notebooks. This target contains a prefix; it is not literally one token.

Run from the repository root in the existing training environment, with
`ROOT` set to the repository root and Hugging Face access to the private dataset.
The selected CPT checkpoint is already uploaded:

```bash
export ROOT="$PWD"
python main.py --config configs/experiments/unsloth-qwen3-4B/exp0015/config_sft.yaml
```

The input is `Maverfrick/Venera-CPT-TMModel`, pinned to commit
`e0589a27e7b66bcbd9cfd8b709c0785d18bf0bf5`. Its adapter metadata specifies
`unsloth/Qwen3-4B-Thinking-2507` as the base model, rank 64, and alpha 64.
Authenticate with an HF token that can read both this checkpoint and the dataset.
To select a different CPT run, update both `model.name` and `model.revision`;
do not substitute the original base model.

The final SFT adapter is saved separately to
`group_outputs/exp0015/yhct/CPT_SFT/model`; the input CPT adapter is preserved.
The existing output/publishing code identifies CPT-SFT by `cpt` in
`model.name`, so retain that naming convention for a Hub checkpoint.

For Lambda, use the existing train, publish, and evaluation pipeline:

```bash
python scripts/lambda_launch.py --task sft \
  --config configs/experiments/unsloth-qwen3-4B/exp0015/config_sft.yaml
```

All eight Unsloth MCQ evaluation notebooks retain their original settings:
`enable_thinking=False`, greedy decoding, `max_new_tokens=250`, the
`Final Answer: X` prompt, and unrestricted output tokens. Following the
instruction remains the model's responsibility; reasoning or malformed
responses remain visible. No one-token constraint is applied.
