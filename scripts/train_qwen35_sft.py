#!/usr/bin/env python3
"""Native completion-only SFT continuing the pinned Qwen3.5 CPT LoRA."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'CPT'))
from qwen35_common import sha256


def load_config(path):
    import yaml
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    if cfg.get('pipeline', {}).get('implementation') != 'scripts/train_qwen35_sft.py':
        raise ValueError('Expected native Qwen3.5 SFT configuration')
    if cfg['model']['name'] != 'Qwen/Qwen3.5-4B':
        raise ValueError('Only Qwen3.5-4B is supported')
    for section in ('model', 'adapter', 'dataset'):
        if not re.fullmatch('[a-f0-9]{40}', cfg[section]['revision']):
            raise ValueError(f'{section} revision must be an immutable commit')
    for section, key in [('adapter', 'weights_sha256'), ('dataset', 'train_sha256')]:
        if not re.fullmatch('[a-f0-9]{64}', cfg[section][key]):
            raise ValueError(f'{section}.{key} must be a SHA-256 digest')
    if not re.fullmatch('[a-z0-9-]+', cfg['name']):
        raise ValueError('Experiment name must be a slug')
    if 'lora' in cfg:
        raise ValueError('Continue the CPT adapter; do not configure a new LoRA')
    if cfg['training'].get('max_steps', -1) > 0 and not cfg.get('smoke'):
        raise ValueError('Step-limited jobs must be marked smoke')
    if not 0 < cfg['dataset']['validation_fraction'] < 1:
        raise ValueError('Invalid validation fraction')
    return cfg


def artifact_dir(cfg):
    tag = os.getenv('JOB_TAG', 'local')
    if not re.fullmatch('[A-Za-z0-9._-]+', tag):
        raise ValueError('Invalid job tag')
    return ROOT / 'group_outputs/qwen35_sft' / f"{cfg['name']}-{tag}"


def preflight(cfg, token=None):
    """Verify immutable input identities and the dataset's published cleaning audit."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    api.model_info(cfg['model']['name'], revision=cfg['model']['revision'])
    if not api.model_info(cfg['publish']['repo']).private:
        raise ValueError('SFT destination must already exist and be private')
    adapter = json.loads(Path(hf_hub_download(cfg['adapter']['name'], 'adapter_config.json',
        revision=cfg['adapter']['revision'], token=token)).read_text())
    if (adapter['base_model_name_or_path'] != cfg['model']['name'] or
            adapter['revision'] != cfg['model']['revision'] or adapter['r'] != 64):
        raise ValueError('CPT adapter base/revision/rank mismatch')
    info = api.model_info(cfg['adapter']['name'], revision=cfg['adapter']['revision'], files_metadata=True)
    weights = next(f for f in info.siblings if f.rfilename == 'adapter_model.safetensors')
    if weights.lfs is None or weights.lfs.sha256 != cfg['adapter']['weights_sha256']:
        raise ValueError('CPT adapter weights checksum mismatch')
    source = cfg['dataset']
    def fetch(name):
        return Path(hf_hub_download(source['name'], name, repo_type='dataset',
                                   revision=source['revision'], token=token))
    audit = json.loads(fetch('audit.json').read_text())
    data = fetch('train.jsonl')
    if sha256(data) != source['train_sha256'] or audit['train_sha256'] != source['train_sha256']:
        raise ValueError('SFT data checksum mismatch')
    benchmark_hash = sha256(ROOT / 'EVAL/Data/eastern_medicine_VM14K.jsonl')
    if (audit['remaining_matching_rows'] != 0 or not any(
            f['sha256'] == benchmark_hash for f in audit['evaluation_files'])):
        raise ValueError('SFT cleaning audit does not match the frozen VM14K dataset')
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows) != audit['kept_rows']:
        raise ValueError('SFT row count does not match audit')
    for row in rows:
        labels = [chr(65 + i) for i in range(len(row['options']))]
        if (not isinstance(row['question'], str) or not row['question'].strip() or
                not labels or row['answer'] not in labels or
                row['answer_index'] != labels.index(row['answer']) or
                any(not isinstance(o, str) for o in row['options'])):
            raise ValueError('Invalid SFT question/options/answer')
    return rows, audit


def encode_example(row, tokenizer, max_length, system_prompt):
    from eval_medical_baselines import build_prompt
    sample = {'question': row['question'],
              'options': {chr(65+i): option for i, option in enumerate(row['options'])}}
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': build_prompt(sample, system_prompt)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if prompt.rstrip().endswith('<think>'):
        prompt += '</think>\n\n'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(f"Final Answer: {row['answer']}", add_special_tokens=False)
    completion_ids.append(tokenizer.eos_token_id)
    if tokenizer.eos_token_id is None or len(completion_ids) >= max_length:
        raise ValueError('Invalid EOS or completion length')
    # Preserve every answer/EOS token and the assistant prefix when a prompt is long.
    prompt_ids = prompt_ids[-(max_length-len(completion_ids)):]
    if not prompt_ids:
        raise ValueError('Empty SFT prompt')
    return {'input_ids': prompt_ids + completion_ids,
            'labels': [-100] * len(prompt_ids) + completion_ids}


def load_model(cfg, adapter, training, token=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    base = AutoModelForImageTextToText.from_pretrained(cfg['model']['name'],
        revision=cfg['model']['revision'], token=token, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    if training:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={'use_reentrant': False})
    kwargs = {'revision': cfg['adapter']['revision']} if training else {}
    model = PeftModel.from_pretrained(base, adapter, token=token, is_trainable=training, **kwargs)
    model.peft_config['default'].base_model_name_or_path = cfg['model']['name']
    model.peft_config['default'].revision = cfg['model']['revision']
    model.config.use_cache = not training
    model.config.get_text_config().use_cache = not training
    if training:
        model.enable_input_require_grads()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in name or 'visual' in name or 'vision' in name for name in trainable):
            raise ValueError('Only inherited language LoRA weights may be trained')
    else:
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'], token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' if not training else 'right'
    return model, tokenizer, None


def train(cfg):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from importlib.metadata import version
    from eval_medical_baselines import load_protocol
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Use one visible BF16-capable CUDA GPU')
    rows, audit = preflight(cfg, os.getenv('HF_TOKEN'))
    out = artifact_dir(cfg)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(cfg['training']['seed'])
    data = Dataset.from_list(rows).train_test_split(test_size=cfg['dataset']['validation_fraction'],
                                                  seed=cfg['dataset']['split_seed'])
    split_ids = {key: [r['id'] for r in split] for key, split in data.items()}
    if cfg.get('smoke'):
        data['train'] = data['train'].select(range(min(len(data['train']), 8)))
        data['test'] = data['test'].select(range(min(len(data['test']), 4)))
    model, tokenizer, _ = load_model(cfg, cfg['adapter']['name'], True, os.getenv('HF_TOKEN'))
    protocol = load_protocol(ROOT / 'EVAL/Unsloth_QA_goc_VM14K.ipynb')
    def encode(row):
        return encode_example(row, tokenizer, cfg['model']['max_seq_length'], protocol['system_prompt'])
    encoded = {key: split.map(encode, remove_columns=split.column_names) for key, split in data.items()}
    provenance = {'config': cfg, 'dataset_audit': audit,
        'split_ids_sha256': {key: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for key, ids in split_ids.items()},
        'rows': {key: len(split) for key, split in data.items()},
        'versions': {name: version(name) for name in ['torch', 'transformers', 'peft', 'datasets', 'bitsandbytes']},
        'completion_only_loss': True, 'thinking_enabled': False, 'adapter_continued': True}
    source = ROOT / 'CPT/qwen35_source.json'
    if source.exists():
        provenance['source_bundle'] = json.loads(source.read_text())
    (out / 'resolved_config.json').write_text(json.dumps(provenance, indent=2))
    def collate(batch):
        width = max(len(row['input_ids']) for row in batch)
        return {'input_ids': torch.tensor([r['input_ids'] + [tokenizer.pad_token_id]*(width-len(r['input_ids'])) for r in batch]),
                'attention_mask': torch.tensor([[1]*len(r['input_ids']) + [0]*(width-len(r['input_ids'])) for r in batch]),
                'labels': torch.tensor([r['labels'] + [-100]*(width-len(r['labels'])) for r in batch])}
    arguments = dict(cfg['training'])
    arguments.update(output_dir=str(out), remove_unused_columns=False, report_to='wandb',
        run_name=out.name, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    import wandb
    with wandb.init(project=os.getenv('WANDB_PROJECT') or 'qwen35-sft', entity=os.getenv('WANDB_ENTITY') or None,
                    name=out.name, dir=str(out), config=cfg) as run:
        (out / 'wandb_run.json').write_text(json.dumps({'id': run.id, 'url': run.url}, indent=2))
        trainer = Trainer(model=model, processing_class=tokenizer, args=TrainingArguments(**arguments),
            train_dataset=encoded['train'], eval_dataset=encoded['test'], data_collator=collate)
        result = trainer.train()
        trainer.save_model(str(out)); trainer.save_state(); tokenizer.save_pretrained(out)
        summary = {'smoke': cfg.get('smoke', False), 'training_metrics': result.metrics,
                   'best_model_checkpoint': trainer.state.best_model_checkpoint,
                   'best_metric': trainer.state.best_metric, 'final_weights_sha256': sha256(out/'adapter_model.safetensors')}
        (out / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[qwen35-sft] saved adapter: {out}', flush=True)


def publish(cfg):
    from huggingface_hub import HfApi
    out = artifact_dir(cfg)
    for name in ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json', 'training_summary.json']:
        if not (out/name).is_file():
            raise ValueError(f'Cannot publish incomplete SFT adapter: {name}')
    api = HfApi(token=os.getenv('HF_TOKEN')); repo = cfg['publish']['repo']
    branch = f"{cfg['name']}-{os.environ['JOB_TAG']}"
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError('SFT destination must be private')
    api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
    result = api.upload_folder(repo_id=repo, revision=branch, folder_path=str(out),
        ignore_patterns=['checkpoint-*/**', 'wandb/**', 'vm14k/**'])
    info = api.model_info(repo, revision=result.oid, files_metadata=True)
    remote = next(f for f in info.siblings if f.rfilename == 'adapter_model.safetensors')
    if remote.lfs is None or remote.lfs.sha256 != sha256(out/'adapter_model.safetensors'):
        raise ValueError('Uploaded SFT adapter checksum mismatch')
    (out/'publication.json').write_text(json.dumps({'repo': repo, 'branch': branch, 'revision': result.oid}, indent=2))
    print(f'[qwen35-sft] checkpoint verified: https://huggingface.co/{repo}/tree/{branch}', flush=True)


def evaluate(cfg):
    import eval_medical_baselines as runner
    out = artifact_dir(cfg)
    publication = json.loads((out/'publication.json').read_text())
    key = 'qwen35-sft'; runner.MODELS[key] = publication['repo']
    extra = {'base_model': cfg['model']['name'], 'base_revision': cfg['model']['revision'],
             'adapter_composition': 'continued CPT-SFT adapter applied once',
             'adapter_weights_sha256': sha256(out/'adapter_model.safetensors')}
    def loader(key, revision, token):
        return load_model(cfg, str(out), False, token)
    args = SimpleNamespace(limit=4, generation_mode='direct', batch_size=4, output_dir=out/'vm14k/smoke')
    runner.evaluate(key, 'vm14k', publication['revision'], args, model_loader=loader, extra_metadata=extra)
    smoke = json.loads((args.output_dir/key/'vm14k/summary.json').read_text())
    if smoke['samples'] != 4 or smoke['deviations'] or smoke['token_limit_rate']:
        raise ValueError('VM14K smoke check failed')
    if not cfg.get('smoke'):
        args.limit = 0; args.output_dir = out/'vm14k/full'
        runner.evaluate(key, 'vm14k', publication['revision'], args, model_loader=loader, extra_metadata=extra)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['preflight', 'train', 'publish', 'evaluate'], default='preflight')
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.mode == 'preflight':
        rows, _ = preflight(cfg, os.getenv('HF_TOKEN'))
        print(json.dumps({'config': cfg, 'validated_rows': len(rows)}, indent=2))
    else:
        {'train': train, 'publish': publish, 'evaluate': evaluate}[args.mode](cfg)


if __name__ == '__main__':
    main()
