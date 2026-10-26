"""CPU-safe configuration, packing and artifact helpers for Qwen3.5 CPT."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = 'CPT/train_qwen35.py'
MODEL = 'Qwen/Qwen3.5-4B'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
           'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')
LORA_PATTERN = r'^(?!.*(?:visual|vision|multi_modal_projector)).*\.(?:' + '|'.join(TARGETS) + ')$'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    import yaml
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text())
    if config.get('pipeline', {}).get('implementation') != IMPLEMENTATION:
        raise ValueError('Expected the dedicated Qwen3.5 CPT implementation')
    if config['model']['name'] != MODEL:
        raise ValueError('This path supports Qwen/Qwen3.5-4B only')
    for section in ['model', 'dataset']:
        if not re.fullmatch(r'[a-f0-9]{40}', config[section].get('revision', '')):
            raise ValueError(f'{section}.revision must be an immutable Hub commit')
    if config['training']['num_train_epochs'] != 1 or config['training']['seed'] != 42:
        raise ValueError('These experiments require one epoch and seed 42')
    if config['dataset'].get('add_chat_template') or not config['dataset'].get('preserve_whitespace'):
        raise ValueError('CPT requires raw text with code whitespace preserved')
    if config['evaluation']['dataset'] != 'vm14k':
        raise ValueError('Only VM14K evaluation is supported by this path')
    if config['training'].get('max_steps', -1) > 0 and not config.get('smoke', False):
        raise ValueError('A step-limited run must be labeled smoke')
    if not re.fullmatch(r'[a-z0-9-]+', config['name']):
        raise ValueError('Experiment name must be a lowercase slug')
    return config


def check_dataset(config, token):
    """Check the published data's tokenizer identity and train/validation metadata."""
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(token=token).dataset_info(config['dataset']['name'], revision=config['dataset']['revision'])
    if info.sha != config['dataset']['revision']:
        raise ValueError('Dataset revision mismatch')
    path = hf_hub_download(config['dataset']['name'], 'preparation_manifest.json', repo_type='dataset',
                           revision=config['dataset']['revision'], token=token)
    manifest = json.loads(Path(path).read_text())
    if (manifest['config']['tokenizer']['name'] != config['model']['name'] or
            manifest['config']['tokenizer']['revision'] != config['model']['revision']):
        raise ValueError('Dataset was prepared with a different model/tokenizer')
    buckets = manifest['buckets']
    if config['dataset']['mixture'] == 'tradmed':
        if set(buckets) != {'domain'}:
            raise ValueError('TradMed-only config requires domain-only prepared data')
    elif config['dataset']['mixture'] == 'mixed':
        expected = {'domain': .8, 'fineweb': .12, 'dolma': .04, 'math': .02, 'code': .02}
        if set(buckets) != set(expected) or any(abs(buckets[k]['actual_share'] - v) > .0001 for k, v in expected.items()):
            raise ValueError('Dataset does not match the 80/12/4/2/2 mixture')
    else:
        raise ValueError('Unknown data mixture')
    return manifest


def output_dir(config, job_tag=None):
    tag = job_tag or os.getenv('JOB_TAG') or f'local-{time.time_ns()}'
    if not re.fullmatch(r'[A-Za-z0-9._-]+', tag):
        raise ValueError('Invalid job tag')
    return ROOT / 'CPT/qwen35_outputs' / f"{config['name']}-{tag}"


def packed_examples(tokenized, length):
    """No dropped remainder; a one-token tail gets masked preceding context."""
    buffer = []
    previous = None
    for row in tokenized:
        buffer.extend(row['input_ids'])
        while len(buffer) >= length:
            part, buffer = buffer[:length], buffer[length:]
            previous = part[-1]
            yield {'input_ids': part, 'labels': part.copy()}
    if len(buffer) == 1:
        if previous is None:
            raise ValueError('At least two corpus tokens are needed for causal LM training')
        yield {'input_ids': [previous, buffer[0]], 'labels': [-100, buffer[0]]}
    elif buffer:
        yield {'input_ids': buffer, 'labels': buffer.copy()}


def publish(config, directory, diagnostics=False):
    from huggingface_hub import HfApi
    api = HfApi(token=os.getenv('HF_TOKEN'))
    repo = config['publish']['repo']
    tag = os.getenv('JOB_TAG') or directory.name
    branch = f"{config['name']}-{tag}"
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError('Qwen3.5 run artifacts must use the configured private model repository')
    api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
    if not diagnostics:
        for name in ['adapter_config.json', 'adapter_model.safetensors', 'resolved_config.json',
                     'training_summary.json', 'vm14k/summary.json']:
            if not (directory / name).is_file():
                raise ValueError(f'Required output missing: {name}')
    api.upload_folder(repo_id=repo, revision=branch, folder_path=str(directory),
                      ignore_patterns=['checkpoint-*/*'])
    log = Path('/host/train.log')
    if log.is_file():
        api.upload_file(repo_id=repo, revision=branch, path_or_fileobj=str(log), path_in_repo='train.log')
    elif os.getenv('JOB_TAG'):
        raise ValueError('Lambda train.log is missing')
    print(f'[qwen35] artifacts: https://huggingface.co/{repo}/tree/{branch}', flush=True)
