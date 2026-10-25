#!/usr/bin/env python3
"""Fetch pinned upstream CPT sources on Lambda and write frozen JSONL exports."""
from __future__ import annotations
from contextlib import closing
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'CPT'))
from pilot_data import SHARES, digest


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get('sources'), list):
        raise ValueError('Acquisition config requires sources')
    if len(config['sources']) != 5 or {s['bucket'] for s in config['sources']} != set(SHARES):
        raise ValueError('Acquisition requires the five original source buckets')
    for source in config['sources']:
        if not re.fullmatch(r'[a-f0-9]{40}', source.get('revision', '')):
            raise ValueError('Each upstream source requires an immutable revision')
        if not source.get('repo'):
            raise ValueError('Each source requires a repository')
    if config.get('seed') != 42:
        raise ValueError('Acquisition seed must be 42')
    return config


def require_code_credentials():
    # The Stack v2 provides IDs; its documented content store is Software Heritage S3.
    missing = [key for key in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY') if not os.getenv(key)]
    if missing:
        raise ValueError('The Stack v2 content download requires AWS_ACCESS_KEY_ID and '
                         'AWS_SECRET_ACCESS_KEY (plus AWS_SESSION_TOKEN for temporary credentials). '
                         'Set these in .env; they will be forwarded to Lambda. No corpus is substituted.')


def check_access(config):
    """Metadata and gated-file access checks only, before GPU allocation."""
    from huggingface_hub import HfApi, hf_hub_url
    import requests
    require_code_credentials()
    token = os.getenv('HF_TOKEN')
    api = HfApi(token=token)
    for source in config['sources']:
        info = api.dataset_info(source['repo'], revision=source['revision'])
        if info.sha != source['revision']:
            raise ValueError('Upstream source revision mismatch')
        if source['bucket'] == 'code':
            filename = next(item.rfilename for item in info.siblings
                            if item.rfilename.startswith('data/Python/') and item.rfilename.endswith('.parquet'))
            response = requests.head(hf_hub_url(source['repo'], filename, repo_type='dataset',
                                      revision=source['revision']),
                                     headers={'Authorization': f'Bearer {token}'} if token else {},
                                     allow_redirects=True, timeout=30)
            response.raise_for_status()


def hf_rows(source, split='train'):
    from datasets import load_dataset
    options = {'revision': source['revision'], 'split': split, 'streaming': True,
               'token': os.getenv('HF_TOKEN')}
    if source.get('config'):
        options['name'] = source['config']
    if source.get('data_dir'):
        options['data_dir'] = source['data_dir']
    yield from load_dataset(source['repo'], **options)


def dolma_rows(source):
    from huggingface_hub import hf_hub_download
    import requests
    listing = hf_hub_download(source['repo'], source['url_list'], repo_type='dataset',
                              revision=source['revision'], token=os.getenv('HF_TOKEN'))
    urls = Path(listing).read_text().splitlines()
    allowed = {'books': 'books', 'wiki': 'encyclopedia'}
    def component_rows(component):
        for url in urls:
            if f'/{component}/' not in url:
                continue
            # Never execute the upstream dataset loading script.
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=response.raw) as compressed:
                    for line in io.TextIOWrapper(compressed, encoding='utf-8'):
                        row = json.loads(line)
                        row['_component'] = allowed[component]
                        row['_download_url'] = url
                        yield row
    # Interleave books and encyclopedia rather than exhaust one component first.
    iterators = [component_rows(component) for component in allowed]
    try:
        for pair in itertools.zip_longest(*iterators):
            for row in pair:
                if row is not None:
                    yield row
    finally:
        for iterator in iterators:
            iterator.close()


def hydrate_code(row, client):
    body = client.get_object(Bucket='softwareheritage', Key=f"content/{row['blob_id']}")['Body']
    with closing(body):
        data = body.read()
    if data.startswith(b'\x1f\x8b'):
        data = gzip.decompress(data)
    return data.decode(row.get('src_encoding') or 'utf-8')


def normalize(source, row, code_client=None):
    bucket = source['bucket']
    extra = {}
    if bucket == 'fineweb' and float(row.get('score', 0)) < source['min_quality_score']:
        return None
    if bucket == 'code':
        licenses = row.get('detected_licenses') or []
        if (row.get('language') not in source['allowed_languages'] or not licenses or
                not set(licenses).issubset(source['allowed_licenses']) or
                row.get('license_type') != 'permissive' or row.get('is_generated') or row.get('is_vendor')):
            return None
        try:
            text = hydrate_code(row, code_client)
        except (UnicodeDecodeError, LookupError):
            return None
        identifier = row['blob_id']
        extra.update(license=sorted(licenses)[0], language=row['language'],
                     detected_licenses=licenses, blob_id=identifier)
    else:
        text = row.get('text', '')
        if bucket == 'domain':
            identifier = row.get('source')
            if not identifier or 'chunk_index' not in row:
                raise ValueError('Medical corpus requires source and chunk_index to group original documents')
            extra['chunk_index'] = int(row['chunk_index'])
        else:
            identifier = row.get('id') or row.get('url') or hashlib.sha256(text.encode()).hexdigest()
        if bucket == 'dolma':
            extra.update(component=row['_component'], download_url=row['_download_url'])
    if not isinstance(text, str) or not text.strip():
        return None
    return {'text': text, 'document_id': f"{source['repo']}:{identifier}",
            'source_id': str(row.get('url') or row.get('source') or identifier), **extra}


def fetch(config_path, manifest_path, tokenizer):
    config = load_config(config_path)
    require_code_credentials()
    import boto3
    client = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f'Acquisition manifest already exists: {manifest_path}')
    exports = manifest_path.parent / 'exports'
    exports.mkdir(parents=True, exist_ok=True)
    sources = sorted(config['sources'], key=lambda s: s['bucket'] != 'domain')
    manifest = {'excluded_source_ids': config.get('excluded_source_ids', []),
                'acquisition_config_sha256': digest(config_path), 'sources': []}
    domain_tokens = 0
    for source in sources:
        bucket = source['bucket']
        splits = ['train', 'validation'] if bucket == 'domain' else ['train']
        files, measurements = [], {}
        for split in splits:
            file = exports / f'{bucket}-{split}.jsonl'
            if file.exists():
                raise FileExistsError(f'Acquisition export already exists: {file}')
            target = (config.get('max_domain_tokens') if bucket == 'domain' else
                      max(1, int(domain_tokens * SHARES[bucket] / .8 * config['replay_headroom'])))
            iterator = dolma_rows(source) if bucket == 'dolma' else hf_rows(source, split)
            count = tokens = validation_docs = 0
            seen = set()
            with closing(iterator), file.open('w') as stream:
                for scanned, row in enumerate(iterator, 1):
                    if scanned > config['max_scan_documents']:
                        raise ValueError(f'Acquisition scan limit exceeded: {bucket}/{split}')
                    item = normalize(source, row, client)
                    if item is None:
                        continue
                    identity = (item['document_id'], item.get('chunk_index', 0))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    count += 1
                    tokens += len(tokenizer.encode(item['text'], add_special_tokens=False)) + 1
                    validation_docs += int(int(hashlib.sha256(f"42:{item['document_id']}".encode()).hexdigest(), 16) % 100 == 0)
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                    if target and tokens >= target and (bucket == 'domain' or
                            (count >= config['min_replay_documents'] and validation_docs >= 2)):
                        break
            if (not count or (target and tokens < target) or (bucket != 'domain' and
                    (count < config['min_replay_documents'] or validation_docs < 2))):
                raise ValueError(f'Insufficient source data: {bucket}/{split}')
            files.append({'path': str(file.relative_to(manifest_path.parent)), 'sha256': digest(file),
                          'split': split if bucket == 'domain' else 'unsplit'})
            measurements[split] = {'documents': count, 'tokens_before_cleaning': tokens}
            if bucket == 'domain' and split == 'train':
                domain_tokens = tokens
        manifest['sources'].append({**source, 'files': files, 'acquisition_counts': measurements,
            'selection_notes': source['selection_notes']})
        print(f'[fetch] {bucket}: {measurements}', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
