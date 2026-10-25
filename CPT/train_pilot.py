"""Pilot branch of CPT.py; legacy training remains unchanged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'CPT'))
from pilot_data import allocate, digest, packed_rows


def train(config):
    # Unsloth must patch Transformers before its imports.
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from importlib.metadata import version
    import yaml

    model_cfg = config['model']
    pilot = config['pilot']
    prepared = ROOT / pilot['prepared_dir']
    preparation = json.loads((prepared / 'preparation.json').read_text())
    for bucket, splits in preparation['buckets'].items():
        for split, metadata in splits.items():
            if digest(prepared / f'{bucket}-{split}.jsonl') != metadata['sha256']:
                raise ValueError(f'Prepared data hash mismatch: {bucket}/{split}')
    if preparation['tokenizer_revision'] != model_cfg['revision']:
        raise ValueError('Prepared tokens and model tokenizer revisions differ')
    revision = model_cfg['revision']
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('An immutable model revision is required')
    output = ROOT / config['training']['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'adapter_config.json').exists():
        raise ValueError('Refusing to reuse an existing adapter output directory')
    budget = allocate(preparation['buckets']['domain']['train']['tokens'], pilot['mode'])
    from transformers import set_seed
    set_seed(config['training']['seed'])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg['name'], revision=revision,
        max_seq_length=model_cfg['max_seq_length'],
        load_in_4bit=config['unsloth']['load_in_4bit'], dtype=torch.bfloat16)
    # A multimodal checkpoint can return a processor; this experiment uses text only.
    tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    targets = config['lora']['target_modules']
    modules = dict(model.named_modules())
    found = {name: sum(key.endswith('.' + name) and 'visual' not in key for key in modules)
             for name in targets}
    if not all(found.values()):
        raise ValueError(f'Unsupported LoRA targets: {found}')
    lora = dict(config['lora'])
    # Anchor adapters to language modules only, including hybrid linear attention.
    escaped = '|'.join(re.escape(name) for name in targets)
    lora['target_modules'] = rf'^(?!.*(?:visual|vision|multi_modal_projector)).*\.({escaped})$'
    if not any(re.fullmatch(lora['target_modules'], key) for key in modules):
        raise ValueError('No compatible language projection modules were found')
    lora['random_state'] = config['training']['seed']
    lora['use_gradient_checkpointing'] = config['unsloth']['gradient_checkpointing']
    model = FastLanguageModel.get_peft_model(model, **lora)
    train_data = Dataset.from_generator(lambda: packed_rows(prepared, budget,
                                        model_cfg['max_seq_length'], config['training']['seed']))
    validation = []
    for bucket in budget:
        for line in (prepared / f'{bucket}-validation.jsonl').read_text().splitlines():
            ids = json.loads(line)['input_ids']
            for start in range(0, len(ids), model_cfg['max_seq_length']):
                part = ids[start:start + model_cfg['max_seq_length']]
                if len(part) > 1:
                    validation.append({'input_ids': part, 'attention_mask': [1] * len(part), 'labels': part.copy()})
    # Pad labels by position, never by EOS ID: EOS is a supervised document boundary.
    def collate(rows):
        width = max(len(row['input_ids']) for row in rows)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        result = {}
        for field, fill in [('input_ids', pad), ('attention_mask', 0), ('labels', -100)]:
            result[field] = torch.tensor([row[field] + [fill] * (width - len(row[field])) for row in rows])
        if 'bucket_counts' in rows[0]:
            result['_bucket_counts'] = torch.tensor([[row['bucket_counts'][bucket] for bucket in budget] for row in rows])
        return result

    class CountingTrainer(UnslothTrainer):
        processed = 0
        processed_buckets = {bucket: 0 for bucket in budget}
        def training_step(self, model, inputs, *args, **kwargs):
            counts = inputs.pop('_bucket_counts').sum(dim=0).tolist()
            for bucket, count in zip(budget, counts):
                self.processed_buckets[bucket] += int(count)
            self.processed += int(inputs['attention_mask'].sum().item())
            return super().training_step(model, inputs, *args, **kwargs)

    args = dict(config['training'])
    args['output_dir'] = str(output)
    if args.get('num_train_epochs') != 1:
        raise ValueError('Pilot requires one epoch over its allocated token stream')
    args.update(dataloader_drop_last=False, dataloader_num_workers=0, remove_unused_columns=False,
                report_to='none')
    resolved = dict(config)
    resolved['data_snapshot'] = preparation
    resolved['token_budget'] = budget
    resolved['lora_matches'] = found
    resolved['source_bundle'] = json.loads((ROOT / 'CPT/pilot_source.json').read_text())
    resolved['versions'] = {name: version(name) for name in ['torch', 'transformers', 'unsloth', 'trl', 'peft']}
    (output / 'resolved_config.json').write_text(json.dumps(resolved, indent=2))
    (output / 'config.yaml').write_text(yaml.safe_dump(config))
    trainer = CountingTrainer(model=model, tokenizer=tokenizer, train_dataset=train_data,
        eval_dataset=Dataset.from_list(validation), data_collator=collate,
        max_seq_length=model_cfg['max_seq_length'], packing=False,
        dataset_kwargs={'skip_prepare_dataset': True}, args=UnslothTrainingArguments(**args))
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    # PEFT revision is explicit so later loaders cannot silently use base@main.
    adapter = json.loads((output / 'adapter_config.json').read_text())
    adapter.update(base_model_name_or_path=model_cfg['name'], revision=revision)
    (output / 'adapter_config.json').write_text(json.dumps(adapter, indent=2))
    planned = sum(budget.values())
    accounting = {'planned_nonpadding_tokens': planned, 'actual_nonpadding_tokens': trainer.processed,
                  'complete_budget': trainer.processed == planned,
                  'buckets': {bucket: {'allocated_tokens': value,
                      'actual_processed_tokens': trainer.processed_buckets[bucket],
                      'actual_share': trainer.processed_buckets[bucket] / max(1, trainer.processed),
                      'unique_available_tokens': preparation['buckets'][bucket]['train']['tokens'],
                      'allocated_share': value / planned,
                      'effective_passes': value / preparation['buckets'][bucket]['train']['tokens']}
                      for bucket, value in budget.items()}}
    (output / 'token_counts.json').write_text(json.dumps(accounting, indent=2))
    if config['training'].get('max_steps', -1) <= 0 and not accounting['complete_budget']:
        raise RuntimeError('Trainer did not consume exactly the allocated token budget')
    print(f'[pilot] adapter saved: {output}', flush=True)
