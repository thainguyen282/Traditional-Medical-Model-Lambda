#!/usr/bin/env python3
"""Isolated Transformers/PEFT raw-text CPT for Qwen3.5; legacy CPT.py is unchanged."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re

from qwen35_common import ROOT, LORA_PATTERN, check_dataset, load_config, output_dir, packed_examples


def train(config, directory):
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from importlib.metadata import version
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Use one visible CUDA GPU for this experiment')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The Qwen3.5 experiment requires BF16 support')
    set_seed(config['training']['seed'])
    manifest = check_dataset(config, os.getenv('HF_TOKEN'))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'adapter_config.json').exists():
        raise FileExistsError('Refusing to overwrite an existing adapter output')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], revision=config['model']['revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    length = config['model']['max_seq_length']
    def prepared(split):
        raw = load_dataset(config['dataset']['name'], split=split, revision=config['dataset']['revision'],
                            token=os.getenv('HF_TOKEN'))
        # Deliberately do not strip, collapse whitespace, append EOS, or apply a chat template.
        def tokenize(batch):
            if any(not isinstance(text, str) or not text for text in batch['text']):
                raise ValueError('Dataset contains invalid raw text')
            return {'input_ids': tokenizer(batch['text'], add_special_tokens=False, return_attention_mask=False)['input_ids']}
        encoded = raw.map(tokenize, batched=True, batch_size=128, remove_columns=raw.column_names)
        return Dataset.from_generator(packed_examples, gen_kwargs={'tokenized': encoded, 'length': length})
    training_data = prepared('train')
    validation_data = prepared('validation')
    if config.get('smoke'):
        training_data = training_data.select(range(min(len(training_data), 32)))
    validation_data = validation_data.select(range(min(len(validation_data), config['validation_blocks'])))
    model = AutoModelForImageTextToText.from_pretrained(config['model']['name'], revision=config['model']['revision'],
                dtype=torch.bfloat16, attn_implementation='sdpa')
    matches = [name for name, module in model.named_modules() if re.fullmatch(LORA_PATTERN, name)]
    if not matches:
        raise ValueError('No expected Qwen3.5 language projection targets were found')
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=LORA_PATTERN,
                               **config['lora']))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ('lora_' not in name or 'visual' in name or 'vision' in name):
            raise ValueError(f'Unexpected trainable parameter: {name}')
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.enable_input_require_grads()
    resolved = {'config': config, 'preparation_manifest': manifest, 'lora_targets': matches,
                'versions': {name: version(name) for name in ['torch', 'transformers', 'peft', 'datasets', 'bitsandbytes']}}
    source = ROOT / 'CPT/qwen35_source.json'
    if source.exists():
        resolved['source_bundle'] = json.loads(source.read_text())
    (directory / 'resolved_config.json').write_text(json.dumps(resolved, indent=2))
    def collate(rows):
        width = max(len(row['input_ids']) for row in rows)
        return {
            'input_ids': torch.tensor([row['input_ids'] + [tokenizer.pad_token_id] * (width-len(row['input_ids'])) for row in rows]),
            'attention_mask': torch.tensor([[1]*len(row['input_ids'])+[0]*(width-len(row['input_ids'])) for row in rows]),
            'labels': torch.tensor([row['labels']+[-100]*(width-len(row['labels'])) for row in rows])}
    class CountingTrainer(Trainer):
        processed_tokens = 0
        def training_step(self, model, inputs, *args, **kwargs):
            self.processed_tokens += int(inputs['attention_mask'].sum().item())
            return super().training_step(model, inputs, *args, **kwargs)
    arguments = dict(config['training'])
    arguments.update(output_dir=str(directory), report_to='wandb', run_name=directory.name,
                     remove_unused_columns=False,
                     dataloader_drop_last=False, gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={'use_reentrant': False}, save_strategy='no')
    import wandb
    # Keep W&B's local files with the run artifacts; never send credentials in config.
    with wandb.init(project=os.getenv('WANDB_PROJECT') or 'qwen35-cpt',
                    entity=os.getenv('WANDB_ENTITY') or None,
                    name=directory.name, dir=str(directory), config=config) as run:
        tracking = {'id': run.id, 'url': run.url, 'project': run.project, 'name': run.name}
        (directory/'wandb_run.json').write_text(json.dumps(tracking, indent=2))
        print(f'[qwen35] W&B: {run.url}', flush=True)
        trainer = CountingTrainer(model=model, processing_class=tokenizer,
            train_dataset=training_data, eval_dataset=validation_data, data_collator=collate,
            args=TrainingArguments(**arguments))
        result = trainer.train()
        run.summary['processed_nonpadding_tokens'] = trainer.processed_tokens
    trainer.save_model(str(directory))
    tokenizer.save_pretrained(directory)
    adapter = json.loads((directory/'adapter_config.json').read_text())
    adapter.update(base_model_name_or_path=config['model']['name'], revision=config['model']['revision'])
    (directory/'adapter_config.json').write_text(json.dumps(adapter, indent=2))
    (directory/'training_summary.json').write_text(json.dumps({'smoke': config.get('smoke', False),
        'processed_nonpadding_tokens': trainer.processed_tokens, 'prepared_train_tokens': manifest['actual_train_tokens'],
        'packed_train_sequences': len(training_data), 'validation_blocks': len(validation_data),
        'training_metrics': result.metrics}, indent=2))
    print(f'[qwen35] final adapter: {directory}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--print-config', action='store_true')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.print_config:
        print(json.dumps(config,indent=2)); return
    train(config, output_dir(config))


if __name__ == '__main__':
    main()
