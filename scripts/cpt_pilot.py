#!/usr/bin/env python3
"""Preflight, evaluate, publish and compare the Lambda CPT pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'CPT'))
from pilot_data import digest, VM14K_SHA


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if not config.get('pilot', {}).get('enabled'):
        raise ValueError('Expected pilot.enabled: true')
    for key in ['revision', 'tokenizer_revision']:
        if not re.fullmatch(r'[a-f0-9]{40}', config['model'].get(key, '')):
            raise ValueError(f'model.{key} must be an immutable commit')
    if config['model']['revision'] != config['model']['tokenizer_revision']:
        raise ValueError('Pilot uses the model revision for its tokenizer')
    if config['evaluation'].get('enable_thinking') is not True:
        raise ValueError('Pilot requires explicit thinking-enabled evaluation')
    if config['training']['num_train_epochs'] != 1 or config['training']['seed'] != 42:
        raise ValueError('Pilot requires one token-budget epoch and seed 42')
    if config['evaluation']['dataset'] != 'vm14k':
        raise ValueError('Pilot evaluates VM14K only')
    if config['pilot']['mode'] not in ('mixed', 'domain-only'):
        raise ValueError('Unknown pilot mode')
    for path_key in [('pilot', 'prepared_dir'), ('training', 'output_dir')]:
        path = Path(config[path_key[0]][path_key[1]])
        if path.is_absolute() or '..' in path.parts or not str(path).startswith('CPT/'):
            raise ValueError(f'{path_key} must be a repository-relative CPT/ path')
    if not config['training']['output_dir'].startswith('CPT/pilot_outputs/'):
        raise ValueError('Pilot output_dir must be under CPT/pilot_outputs/ for Lambda backup')
    return config


def preparation_inputs(config):
    """Validate the preparation/training contract without local tokenization."""
    from prepare_cpt_data import load_config as load_preparation
    reference = Path(config['pilot']['preparation_config'])
    if reference.is_absolute() or '..' in reference.parts:
        raise ValueError('Preparation config must be repository-relative')
    path = ROOT / reference
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('Preparation config must be in the repository')
    preparation = load_preparation(path)
    if (preparation['data']['output_dir'] != config['pilot']['prepared_dir'] or
            preparation['tokenizer']['name'] != config['model']['name'] or
            preparation['tokenizer']['revision'] != config['model']['tokenizer_revision'] or
            preparation['seed'] != config['training']['seed']):
        raise ValueError('Preparation output, tokenizer and seed must match training')
    files = {str(path.relative_to(ROOT)), 'scripts/prepare_cpt_data.py'}
    acquisition = preparation['data'].get('acquisition_config')
    if acquisition:
        from fetch_cpt_sources import load_config as load_acquisition
        acquisition_path = (ROOT / acquisition).resolve()
        if not acquisition_path.is_relative_to(ROOT):
            raise ValueError('Acquisition config must be in the repository')
        load_acquisition(acquisition_path)
        files.update({acquisition, 'scripts/fetch_cpt_sources.py'})
    else:
        from pilot_data import load_manifest
        manifest_path = ROOT / preparation['data']['source_manifest']
        manifest = load_manifest(manifest_path)
        files.add(str(manifest_path.relative_to(ROOT)))
        files.update(str((manifest_path.parent / item['path']).resolve().relative_to(ROOT))
                     for source in manifest['sources'] for item in source['files'])
    return preparation, files


def preflight(path, baseline=False, launch=False):
    config = load_config(path)
    from eval_medical_baselines import prepare_dataset
    rows, _, meta = prepare_dataset('vm14k', generation_mode='notebook')
    if len(rows) != 564 or meta['dataset_sha256'] != VM14K_SHA:
        raise ValueError('Frozen VM14K subset mismatch')
    if not baseline and launch and config['pilot'].get('preparation_config'):
        preparation_inputs(config)
        return config
    if not baseline:
        directory = ROOT / config['pilot']['prepared_dir']
        preparation_file = directory / 'preparation.json'
        if not preparation_file.is_file():
            raise FileNotFoundError(
                f'Prepared pilot corpus is missing: {preparation_file}. '
                'Training requires frozen exports for domain, FineWeb-Edu, Dolma, '
                'OpenWebMath, and licensed code; these are fetched by the Lambda preparation stage. '
                'After creating CPT/pilot_data/sources.json and its referenced exports, run: '
                'python scripts/prepare_cpt_data.py '
                '--config configs/data/cpt-pilot/prepare.yaml --fetch-sources, or launch '
                'the Lambda job to run fetching and preparation before training. '
                'See scripts/CPT_PILOT_REVIEW.md for the export schema. '
            )
        stats = json.loads(preparation_file.read_text())
        from pilot_data import SHARES
        if set(stats['buckets']) != set(SHARES) or stats['vm14k_sha256'] != VM14K_SHA:
            raise ValueError('Prepared snapshot must include all five sources and frozen VM14K checks')
        if stats['seed'] != config['training']['seed']:
            raise ValueError('Prepared snapshot seed mismatch')
        if stats['tokenizer_revision'] != config['model']['tokenizer_revision']:
            raise ValueError('Prepared tokenizer revision mismatch')
        for bucket, splits in stats['buckets'].items():
            for split, info in splits.items():
                if digest(directory / f'{bucket}-{split}.jsonl') != info['sha256']:
                    raise ValueError(f'Prepared snapshot changed: {bucket}/{split}')
        if (ROOT / config['training']['output_dir'] / 'adapter_config.json').exists():
            raise ValueError('Output adapter already exists; choose a fresh output directory')
    return config


def extract_final(raw, truncated):
    if truncated or '</think>' not in raw:
        return None
    final = raw.rsplit('</think>', 1)[1]
    explicit = re.findall(r'Final\s+Answer\s*:\s*\**\s*([A-D])\b', final, re.I)
    if explicit:
        return explicit[-1].upper()
    standalone = re.fullmatch(r'\s*\**([A-D])\**[.)]?\s*', final, re.I)
    return standalone[1].upper() if standalone else None


def evaluate(config, output, adapter=None, limit=0):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from eval_medical_baselines import prepare_dataset, build_prompt
    model_cfg = config['model']
    protocol_cfg = config['evaluation']
    rows, protocol, metadata = prepare_dataset('vm14k', generation_mode='notebook')
    if limit:
        rows = rows[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation_config.json').write_text(json.dumps(config, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['name'], revision=model_cfg['tokenizer_revision'])
    model = AutoModelForImageTextToText.from_pretrained(model_cfg['name'], revision=model_cfg['revision'],
        torch_dtype=torch.bfloat16, device_map='auto')
    if adapter:
        from peft import PeftModel
        adapter_cfg = json.loads((Path(adapter) / 'adapter_config.json').read_text())
        if (adapter_cfg['base_model_name_or_path'] != model_cfg['name'] or
                adapter_cfg.get('revision') != model_cfg['revision']):
            raise ValueError('Adapter base or revision differs from the pinned evaluation base')
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    settings = {k: protocol_cfg[k] for k in ['max_new_tokens', 'temperature', 'top_p', 'top_k']}
    settings.update(do_sample=True, pad_token_id=tokenizer.eos_token_id)
    predictions = []
    with (output / 'predictions.jsonl').open('w') as stream:
        for index, row in enumerate(rows):
            set_seed(protocol_cfg['seed'] + index)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': build_prompt(row, protocol['system_prompt'])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
            if inputs['input_ids'].shape[1] > protocol_cfg['max_input_tokens']:
                raise ValueError(f'VM14K prompt exceeds input budget: {row["id"]}')
            with torch.inference_mode():
                result = model.generate(**inputs, **settings)
            tokens = result[0, inputs['input_ids'].shape[1]:]
            raw = tokenizer.decode(tokens, skip_special_tokens=False)
            # A full-budget response ending at EOS is complete, not truncated.
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            truncated = len(tokens) >= settings['max_new_tokens'] and int(tokens[-1]) not in eos
            answer = extract_final(raw, truncated)
            item = {'id': row['id'], 'row_index': index, 'answer': row['answer'], 'prediction': answer,
                    'is_correct': answer == row['answer'], 'invalid': answer is None,
                    'truncated': truncated, 'generated_tokens': len(tokens), 'raw': raw}
            predictions.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'[vm14k] {index + 1}/{len(rows)}', flush=True)
    count = len(predictions)
    summary = {'dataset': 'VM14K traditional-medicine subset (564 questions)',
        'dataset_sha256': metadata['dataset_sha256'], 'protocol_sha256': metadata['protocol_sha256'],
        'model': model_cfg, 'generation': protocol_cfg, 'adapter': str(adapter) if adapter else None,
        'adapter_sha256': digest(Path(adapter) / 'adapter_model.safetensors') if adapter else None,
        'preparation_sha256': digest(ROOT / config['pilot']['prepared_dir'] / 'preparation.json') if adapter else None,
        'smoke_only': bool(limit), 'condition': config['pilot']['mode'] if adapter else 'original', 'total': count,
        'correct': sum(p['is_correct'] for p in predictions),
        'invalid': sum(p['invalid'] for p in predictions), 'truncated': sum(p['truncated'] for p in predictions)}
    summary.update(accuracy=summary['correct'] / count, invalid_rate=summary['invalid'] / count,
                   truncation_rate=summary['truncated'] / count)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def publish(config, job_tag, baseline=False, diagnostics=False):
    """Publish once after eval; a failed run gets diagnostics, never a success claim."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    repo = os.environ['HF_CPT_REPO']
    branch = f'cpt-pilot-{job_tag}'
    api.create_repo(repo_id=repo, exist_ok=True)
    api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
    output = ROOT / 'CPT/pilot_runs' / job_tag
    adapter = ROOT / config['training']['output_dir']
    if not baseline and not diagnostics:
        for name in ['adapter_config.json', 'adapter_model.safetensors', 'resolved_config.json', 'token_counts.json']:
            if not (adapter / name).is_file():
                raise ValueError(f'Missing required artifact: {name}')
        api.upload_folder(repo_id=repo, revision=branch, folder_path=str(adapter),
                          ignore_patterns=['checkpoint-*/*'])
    if not diagnostics and not (output / 'vm14k/summary.json').is_file():
        raise ValueError('VM14K summary missing')
    if diagnostics and adapter.exists():
        for name in ['resolved_config.json', 'token_counts.json', 'config.yaml']:
            if (adapter / name).is_file():
                api.upload_file(repo_id=repo, revision=branch, path_or_fileobj=str(adapter / name),
                                path_in_repo=f'diagnostics/{name}')
    if output.exists():
        api.upload_folder(repo_id=repo, revision=branch, folder_path=str(output), path_in_repo='pilot')
    for path, name in [(Path('/host/train.log'), 'train.log'),
                       (ROOT / 'CPT/pilot_source.json', 'pilot_source.json')]:
        if not path.exists():
            raise ValueError(f'Missing required diagnostic: {path}')
        api.upload_file(repo_id=repo, revision=branch, path_or_fileobj=str(path), path_in_repo=name)
    print(f'[pilot] artifacts: https://huggingface.co/{repo}/tree/{branch}', flush=True)


def compare(directories, output):
    summaries, predictions = [], []
    for directory in directories:
        directory = Path(directory)
        summary = json.loads((directory / 'summary.json').read_text())
        if summary['smoke_only'] or summary['total'] != 564:
            raise ValueError('Comparison requires three complete 564-question runs')
        summaries.append(summary)
        predictions.append([json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()])
    if [summary.get('condition') for summary in summaries] != ['original', 'domain-only', 'mixed']:
        raise ValueError('Runs must be original, domain-only, mixed in that order')
    if any(len(rows) != 564 for rows in predictions):
        raise ValueError('Each prediction file must contain exactly 564 rows')
    if not summaries[1].get('preparation_sha256') or summaries[1]['preparation_sha256'] != summaries[2].get('preparation_sha256'):
        raise ValueError('Domain-only and mixed runs must use the same cleaned data snapshot')
    base = summaries[0]
    for summary, rows in zip(summaries[1:], predictions[1:]):
        for key in ['dataset_sha256', 'protocol_sha256', 'model', 'generation']:
            if summary[key] != base[key]:
                raise ValueError(f'Incompatible comparison: {key}')
        if [(p['id'], p['answer']) for p in rows] != [(p['id'], p['answer']) for p in predictions[0]]:
            raise ValueError('Prediction IDs or reference answers differ')
    rng = random.Random(42)
    lines = ['# VM14K traditional-medicine subset (564 questions)', '',
             '| Condition | Correct/total | Accuracy | Invalid | Truncated | Δ baseline (pp), paired 95% CI |',
             '|---|---:|---:|---:|---:|---:|']
    for label, summary, rows in zip(['Original', 'Domain-only CPT', '80/20 CPT'], summaries, predictions):
        differences = [int(row['is_correct']) - int(base_row['is_correct']) for row, base_row in zip(rows, predictions[0])]
        samples = sorted(100 * sum(rng.choices(differences, k=564)) / 564 for _ in range(10000))
        delta = 100 * sum(differences) / 564
        lines.append(f"| {label} | {summary['correct']}/564 | {summary['accuracy']:.2%} | "
                     f"{summary['invalid_rate']:.2%} | {summary['truncation_rate']:.2%} | "
                     f"{delta:+.2f} [{samples[249]:+.2f}, {samples[9749]:+.2f}] |")
    lines += ['', 'Single-seed results. These measurements do not establish general capability retention.']
    Path(output).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'evaluate', 'publish', 'compare'])
    parser.add_argument('--config')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--job-tag', default=os.getenv('JOB_TAG'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output')
    parser.add_argument('--runs', nargs=3, metavar=('ORIGINAL', 'DOMAIN', 'MIXED'))
    args = parser.parse_args()
    if args.command == 'compare':
        compare(args.runs, args.output)
        return
    config = load_config(args.config)
    if args.command == 'preflight':
        preflight(args.config, args.baseline)
        print('Pilot preflight passed')
    elif args.command == 'evaluate':
        evaluate(config, args.output or ROOT / 'CPT/pilot_runs' / args.job_tag / 'vm14k',
                 None if args.baseline else ROOT / config['training']['output_dir'], args.limit)
    else:
        publish(config, args.job_tag, args.baseline, args.diagnostics)


if __name__ == '__main__':
    main()
