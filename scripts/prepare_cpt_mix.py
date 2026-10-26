#!/usr/bin/env python3
"""Standalone CPU preparation of medical/general CPT text; never starts training."""
from __future__ import annotations
import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import lru_cache
import gzip
import io
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {'domain': .80, 'fineweb': .12, 'dolma': .04, 'math': .02, 'code': .02}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(text):
    if not isinstance(text, str):
        raise ValueError('CPT text must be a string')
    text = unicodedata.normalize('NFC', text)
    return '\n'.join(line for line in (' '.join(line.split()) for line in text.strip().splitlines()) if line)


def clean_for_bucket(text, bucket):
    if bucket == 'code':
        if not isinstance(text, str):
            raise ValueError('Code content must be a string')
        return unicodedata.normalize('NFC', text)
    return clean(text)


def key(text):
    return hashlib.sha256(' '.join(re.findall(r'\w+', text.lower())).encode()).hexdigest()


def split_for(document_id, fraction, seed):
    number = int(hashlib.sha256(f'{seed}:{document_id}'.encode()).hexdigest()[:16], 16)
    return 'validation' if number / 2**64 < fraction else 'train'


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config.get('sources', {})) != set(WEIGHTS):
        raise ValueError('Config requires domain, fineweb, dolma, math and code sources')
    if not 0 < config['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    if config['sequence_length'] < 8 or config['seed'] != 42:
        raise ValueError('Use sequence_length >= 8 and seed 42')
    output = Path(config['output_dir'])
    if output.is_absolute() or '..' in output.parts:
        raise ValueError('output_dir must be repository-relative')
    return config


def input_rows(spec, snapshot, token):
    """Local JSONL/TXT/Parquet or streamed, revision-resolved HF datasets."""
    if spec['kind'] == 'source_groups':
        path = (ROOT / spec['path']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != spec['sha256']:
            raise ValueError('Source checkpoint path or hash mismatch')
        with path.open() as stream:
            saved = json.load(stream)
        snapshot.extend(saved['snapshot'])
        snapshot.append({'kind': 'source_groups', 'path': spec['path'], 'sha256': spec['sha256']})
        yield from saved['documents']
    elif spec['kind'] == 'local':
        path = (ROOT / spec['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Local inputs must be inside the repository')
        snapshot.append({'kind': 'local', 'path': spec['path'], 'sha256': sha256(path)})
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        elif path.suffix == '.txt':
            yield {'text': path.read_text(encoding='utf-8'), 'document_id': spec['path']}
        elif path.suffix == '.parquet':
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                yield from batch.to_pylist()
        else:
            raise ValueError('Local inputs must be .jsonl, .txt or .parquet')
    elif spec['kind'] == 'hf':
        from huggingface_hub import HfApi
        from datasets import load_dataset
        revision = HfApi(token=token).dataset_info(spec['repo'], revision=spec.get('revision', 'main')).sha
        snapshot.append({'kind': 'hf', 'repo': spec['repo'], 'revision': revision,
                         'config': spec.get('config'), 'split': spec.get('split', 'train')})
        options = {'name': spec.get('config'), 'revision': revision,
                   'split': spec.get('split', 'train'), 'streaming': True, 'token': token}
        if spec.get('data_dir'):
            options['data_dir'] = spec['data_dir']
        yield from load_dataset(spec['repo'], **options)
    elif spec['kind'] == 'dolma_urls':
        from huggingface_hub import HfApi, hf_hub_download
        import requests
        revision = HfApi(token=token).dataset_info(spec['repo'], revision=spec.get('revision', 'main')).sha
        listing = hf_hub_download(spec['repo'], spec['url_list'], repo_type='dataset', revision=revision, token=token)
        urls = Path(listing).read_text().splitlines()
        snapshot.append({'kind': 'dolma_urls', 'repo': spec['repo'], 'revision': revision,
                         'url_list': spec['url_list'], 'url_list_sha256': sha256(listing),
                         'note': 'External component contents are identified by exported text hashes, not pinned by the URL listing alone.'})
        def component_rows(component, label):
            for url in urls:
                if f'/{component}/' not in url:
                    continue
                print(f'[prepare] Streaming Dolma {component}: {url}', flush=True)
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    response.raw.decode_content = True
                    decoded = io.BufferedReader(response.raw)
                    for _ in range(3):
                        if not decoded.peek(2).startswith(b'\x1f\x8b'):
                            break
                        decoded = io.BufferedReader(gzip.GzipFile(fileobj=decoded, mode='rb'))
                    with io.TextIOWrapper(decoded, encoding='utf-8') as lines:
                        for line in lines:
                            row = json.loads(line)
                            row['document_id'] = f"dolma:{row['id']}"
                            row['component'] = label
                            row['download_url'] = url
                            yield row
        streams = [component_rows(component, label) for component, label in spec['components'].items()]
        try:
            for group in itertools.zip_longest(*streams):
                for row in group:
                    if row is not None:
                        yield row
        finally:
            for stream in streams:
                stream.close()
    else:
        raise ValueError('Input kind must be local, hf or dolma_urls')


@lru_cache(maxsize=1)
def stack_s3_client():
    """Use configured AWS credentials; never ask a local machine for instance credentials."""
    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError
    try:
        session = boto3.Session()
        if session.get_credentials() is None:
            raise ValueError('The Stack v2 requires AWS credentials for Software Heritage S3: '
                             'set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env '
                             '(AWS_SESSION_TOKEN for temporary keys), or configure an AWS profile')
        return session.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                              config=Config(connect_timeout=30, read_timeout=60,
                                            retries={'max_attempts': 3, 'mode': 'standard'}))
    except BotoCoreError as error:
        raise ValueError(f'Cannot initialize The Stack v2 S3 access: {type(error).__name__}') from error


def hydrate_stack_v2(row):
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        body = stack_s3_client().get_object(Bucket='softwareheritage',
                                            Key=f"content/{row['blob_id']}")['Body']
        try:
            raw = body.read()
        finally:
            body.close()
        if raw.startswith(b'\x1f\x8b'):
            raw = gzip.decompress(raw)
        text = raw.decode(row.get('src_encoding') or 'utf-8')
        return text, hashlib.sha256(raw).hexdigest()
    except (UnicodeDecodeError, LookupError, gzip.BadGzipFile):
        return None, None
    except (BotoCoreError, ClientError) as error:
        code = error.response.get('Error', {}).get('Code', '') if isinstance(error, ClientError) else type(error).__name__
        raise ValueError(f'The Stack v2 content download failed ({code}); check SWH S3 access') from error


def code_allowed(row, source):
    licenses = row.get(source.get('license_field', 'license'))
    licenses = [licenses] if isinstance(licenses, str) else licenses
    return (bool(licenses) and {license.lower() for license in licenses}.issubset(license.lower() for license in source['allowed_licenses']) and
            row.get(source.get('language_field', 'language')) in source['allowed_languages'])


def _collect_uncached(bucket, source, config, snapshot, token, token_limit=None, tokenizer=None):
    """Group original-document chunks before deduplication and splitting."""
    if all(spec['kind'] == 'source_groups' for spec in source['inputs']):
        docs = [doc for spec in source['inputs'] for doc in input_rows(spec, snapshot, token)]
        if any(doc.get('bucket') != bucket or doc.get('split') not in ('train', 'validation') for doc in docs):
            raise ValueError('Source checkpoint bucket/split mismatch')
        return docs
    grouped = {}
    count = scanned_tokens = 0
    # Process established validation first; quarantine overlapping training groups.
    specs = sorted(source['inputs'], key=lambda item: item.get('role') != 'validation')
    for spec in specs:
        for row in input_rows(spec, snapshot, token):
            count += 1
            if count > source.get('max_scan_rows', 2_000_000):
                raise ValueError(f'{bucket}: max_scan_rows exceeded')
            provenance = None
            if bucket == 'code':
                if not code_allowed(row, source):
                    continue
                if source.get('hydrate_stack_v2'):
                    if (row.get('license_type') != 'permissive' or row.get('is_vendor') or row.get('is_generated') or
                            (row.get('length_bytes') or 0) > source.get('max_content_bytes', 10_000_000)):
                        continue
                    text, content_hash = hydrate_stack_v2(row)
                    if text is None:
                        continue
                    provenance = {field: row.get(field) for field in
                                  ['blob_id', 'content_id', 'repo_name', 'path', 'revision_id', 'detected_licenses']}
                    provenance['content_sha256'] = content_hash
                else:
                    text = row.get(source.get('text_field', 'text'), '')
                    if source.get('provenance_fields'):
                        provenance = {field: row.get(field) for field in source['provenance_fields']}
                        provenance['content_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            else:
                text = row.get(source.get('text_field', 'text'), '')
            text = clean_for_bucket(text, bucket)
            if not text.strip():
                continue
            if source.get('min_score') is not None and float(row.get(source.get('score_field', 'score'), 0)) < source['min_score']:
                continue
            if bucket == 'dolma' and row.get(source.get('component_field', 'component')) not in source['allowed_components']:
                continue
            identifier = row.get(source.get('document_field', 'document_id'))
            if identifier is None:
                # Safe for complete documents only. Chunked sources must provide their group field.
                if 'chunk_index' in row:
                    raise ValueError(f'{bucket}: chunked rows require the configured document_field')
                identifier = key(text)
            identifier = str(identifier)
            role = spec.get('role', 'unsplit')
            if role not in ('train', 'validation', 'unsplit'):
                raise ValueError('Input role must be train, validation or unsplit')
            split = split_for(identifier, config['validation_fraction'], config['seed']) if role == 'unsplit' else role
            record = grouped.setdefault(identifier, {'id': identifier, 'bucket': bucket, 'split': split, 'parts': {}})
            if provenance is not None:
                record.setdefault('provenance', provenance)
            if record['split'] != split:
                record['split'] = 'validation'
            index = int(row.get(source.get('chunk_field', 'chunk_index'), 0))
            if index in record['parts'] and record['parts'][index] != text:
                raise ValueError(f'{bucket}: conflicting text for document/chunk {identifier}/{index}')
            if index not in record['parts']:
                record['parts'][index] = text
                if tokenizer is not None and record['split'] == 'train':
                    scanned_tokens += len(tokenizer.encode(text)) + 1
            # HF replay streams are bounded, with extra rows for filtering/validation.
            if token_limit and scanned_tokens >= token_limit and len(grouped) >= 256 and any(d['split'] == 'validation' for d in grouped.values()):
                break
        else:
            continue
        break
    return [{**{k: v for k, v in doc.items() if k != 'parts'},
             'text': '\n'.join(text for _, text in sorted(doc['parts'].items()))}
            for doc in grouped.values()]


def collect(bucket, source, config, snapshot, token, token_limit=None, tokenizer=None):
    """Checkpoint collected groups; filtering/chunking can then be retried locally."""
    local_hashes = {spec['path']: sha256(ROOT / spec['path']) for spec in source['inputs'] if spec['kind'] == 'local'}
    fingerprint = {'collector_version': 2 if bucket == 'code' else 1, 'source': source, 'local_hashes': local_hashes,
                   'seed': config['seed'], 'validation_fraction': config['validation_fraction'],
                   'tokenizer': config.get('tokenizer'), 'token_limit': token_limit}
    cacheable = all(spec['kind'] == 'local' or spec.get('revision', 'main') != 'main' for spec in source['inputs'])
    cache = ROOT / '.cache' / 'cpt-source-groups'
    name = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    path = cache / f'{bucket}-{name}.json'
    if cacheable and path.is_file():
        with path.open() as stream:
            saved = json.load(stream)
        snapshot.extend(saved['snapshot'])
        print(f'[prepare] Reusing {bucket} source checkpoint ({len(saved["documents"]):,} groups)', flush=True)
        return saved['documents']
    offset = len(snapshot)
    documents = _collect_uncached(bucket, source, config, snapshot, token, token_limit, tokenizer)
    if cacheable:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.collect-', dir=cache) as directory:
            temporary = Path(directory) / 'groups.json'
            with temporary.open('w') as stream:
                json.dump({'documents': documents, 'snapshot': snapshot[offset:]}, stream, ensure_ascii=False)
            temporary.replace(path)
    return documents


class Deduper:
    def __init__(self, threshold):
        from datasketch import MinHashLSH
        self.hashes = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.index = 0

    def duplicate(self, text):
        from datasketch import MinHash
        exact = key(text)
        if exact in self.hashes:
            return True
        words = re.findall(r'\w+', text.lower())
        signature = MinHash(num_perm=128, seed=42)
        for start in range(0, max(1, len(words) - 4), 1024):
            signature.update_batch([' '.join(words[i:i + 5]).encode()
                                    for i in range(start, min(start + 1024, max(1, len(words) - 4)))])
        if self.lsh.query(signature):
            return True
        self.hashes.add(exact)
        self.lsh.insert(str(self.index), signature)
        self.index += 1
        return False


def fit_prefix(text, limit, tokenizer, suffix='', preserve_whitespace=False):
    """Cut on Unicode character boundaries, then verify actual token length."""
    if hasattr(tokenizer, 'fit_prefix'):
        return tokenizer.fit_prefix(text, limit, suffix, preserve_whitespace)
    if len(tokenizer.encode(text + suffix)) <= limit:
        return text + suffix, ''
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenizer.encode(text[:middle] + suffix)) <= limit:
            low = middle
        else:
            high = middle - 1
    while low and len(tokenizer.encode(text[:low] + suffix)) > limit:
        low -= 1
    if not low:
        return '', text
    return text[:low] + suffix, text[low:]


def chunk_document(doc, length, tokenizer, eos):
    if hasattr(tokenizer, 'document_chunks'):
        yield from tokenizer.document_chunks(doc, length, eos)
        return
    remaining = doc['text']
    index = 0
    while remaining:
        candidate = remaining[:length * 16]
        final = len(candidate) == len(remaining) and len(tokenizer.encode(candidate + eos)) <= length
        text, rest = fit_prefix(candidate, length if final else length - 1, tokenizer, eos if final else '', doc['bucket'] == 'code')
        rest += remaining[len(candidate):]
        if not text:
            raise ValueError('Unable to fit a Unicode character within the token limit')
        # Match the cleaning performed by the restored CPT loader.
        text = clean_for_bucket(text, doc['bucket'])
        if text:
            yield {'text': text, 'source': doc['id'], 'bucket': doc['bucket'],
                   'chunk_index': index, 'tokens': len(tokenizer.encode(text)),
                   **({'provenance': doc['provenance']} if 'provenance' in doc else {})}
            index += 1
        remaining = rest


def budgets(domain_tokens, total=None):
    total = total if total is not None else math.ceil(domain_tokens / .8)
    if total < 10:
        raise ValueError('Insufficient tokens for a five-source mixture')
    result = {bucket: int(total * fraction) for bucket, fraction in WEIGHTS.items()}
    if total == math.ceil(domain_tokens / .8):
        result['domain'] = domain_tokens
        result['fineweb'] += total - sum(result.values())
    else:
        result['domain'] += total - sum(result.values())
    return result


def select(rows, target, tokenizer, eos, rng, repeat=False):
    rows = list(rows)
    rng.shuffle(rows)
    available = sum(row['tokens'] for row in rows)
    if not available or (available < target and not repeat):
        raise ValueError(f'Not enough cleaned tokens: need {target}, have {available}; supply more data or explicitly enable repetition')
    selected, used, index = [], 0, 0
    while used < target and (repeat or index < len(rows)):
        row = dict(rows[index % len(rows)])
        index += 1
        remaining = target - used
        if row['tokens'] > remaining:
            content = row['text'][:-len(eos)] if row['text'].endswith(eos) else row['text']
            text, _ = fit_prefix(content, remaining, tokenizer, suffix=eos, preserve_whitespace=row['bucket'] == 'code')
            if not text:
                break
            row.update(text=clean_for_bucket(text, row['bucket']))
            row['tokens'] = len(tokenizer.encode(row['text']))
        if not row['tokens']:
            break
        selected.append(row)
        used += row['tokens']
    return selected, used, available


def prepare(config, tokenizer, eos, token=None):
    output = ROOT / config['output_dir']
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    snapshot, dropped = [], Counter()
    print('[prepare] Reading and grouping domain documents', flush=True)
    domain = collect('domain', config['sources']['domain'], config, snapshot, token)
    raw_domain_tokens = sum(len(tokenizer.encode(row['text'])) + 1 for row in domain if row['split'] == 'train')
    targets = budgets(raw_domain_tokens, config.get('total_tokens'))
    documents = domain
    for bucket in WEIGHTS:
        if bucket != 'domain':
            print(f'[prepare] Reading {bucket} replay', flush=True)
            documents += collect(bucket, config['sources'][bucket], config, snapshot, token,
                                 max(1, targets[bucket] * 2), tokenizer)
    excluded_ids = set(config.get('exclude_document_ids', []))
    questions = []
    for name in config.get('exclude_benchmarks', []):
        file = ROOT / name
        snapshot.append({'kind': 'benchmark', 'path': name, 'sha256': sha256(file)})
        for line in file.read_text().splitlines():
            row = json.loads(line)
            question = ' '.join(re.findall(r'\w+', row['question'].lower()))
            if question:
                questions.append(question)
    print(f'[prepare] Deduplicating and chunking {len(documents):,} document groups', flush=True)
    deduper = Deduper(config['near_duplicate_threshold'])
    # Identified validation groups take precedence across sources as well.
    heldout = {doc['id'] for doc in documents if doc['split'] == 'validation'}
    chunks = {bucket: {'train': [], 'validation': []} for bucket in WEIGHTS}
    retained = []
    for document_index, doc in enumerate(sorted(documents, key=lambda item: (item['split'] != 'validation', item['bucket'], item['id'])), 1):
        if document_index % 500 == 0:
            print(f'[prepare] Processed {document_index:,}/{len(documents):,} document groups', flush=True)
        normalized = ' '.join(re.findall(r'\w+', doc['text'].lower()))
        if doc['id'] in excluded_ids or any(question in normalized for question in questions):
            dropped['benchmark_or_source'] += 1
            continue
        if doc['split'] == 'train' and doc['id'] in heldout:
            dropped['heldout_group'] += 1
            continue
        if deduper.duplicate(doc['text']):
            dropped['duplicate'] += 1
            continue
        retained.append(doc)
    print(f'[prepare] Chunking {len(retained):,} retained document groups using {config.get("workers", 4)} CPU workers', flush=True)
    def encode_document(doc):
        return list(chunk_document(doc, config['sequence_length'], tokenizer, eos))
    with ThreadPoolExecutor(max_workers=config.get('workers', 4)) as pool:
        for document_index, (doc, rows) in enumerate(zip(retained, pool.map(encode_document, retained)), 1):
            chunks[doc['bucket']][doc['split']].extend(rows)
            if document_index % 500 == 0:
                print(f'[prepare] Chunked {document_index:,}/{len(retained):,} document groups', flush=True)
    domain_tokens = sum(row['tokens'] for row in chunks['domain']['train'])
    targets = budgets(domain_tokens, config.get('total_tokens'))
    rng = random.Random(config['seed'])
    train, validation, counts = [], [], {}
    for bucket in WEIGHTS:
        rows, actual, available = select(chunks[bucket]['train'], targets[bucket], tokenizer, eos,
                                         rng, config['allow_repetition'])
        if not chunks[bucket]['validation']:
            raise ValueError(f'{bucket}: no validation documents survived; supply more documents or an explicit validation input')
        train.extend(rows)
        validation.extend(chunks[bucket]['validation'])
        counts[bucket] = {'target_tokens': targets[bucket], 'actual_tokens': actual,
                          'available_train_tokens': available, 'effective_passes': actual / available,
                          'train_rows': len(rows), 'validation_rows': len(chunks[bucket]['validation'])}
    rng.shuffle(train)
    total = sum(row['tokens'] for row in train)
    for info in counts.values():
        info['actual_share'] = info['actual_tokens'] / total
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cpt-mix-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        stage.mkdir()
        for split, rows in [('train', train), ('validation', validation)]:
            with (stage / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        report = {'preparation_script_sha256': sha256(Path(__file__)), 'config': config, 'sources': snapshot, 'filters': dict(dropped), 'buckets': counts,
                  'actual_train_tokens': total, 'train_rows': len(train), 'validation_rows': len(validation),
                  'files': {split: sha256(stage / f'{split}.jsonl') for split in ['train', 'validation']},
                  'notes': ['Corpus token counts precede trainer packing and epoch repetition.',
                            'Final source boundaries may undershoot targets; actual shares are reported.',
                            'MinHash near-duplicate detection is approximate; source article exclusion needs known IDs.']}
        (stage / 'manifest.json').write_text(json.dumps(report, indent=2))
        if output.exists():
            raise FileExistsError(f'Output created concurrently: {output}')
        stage.rename(output)
    return output, report


class Tokenizer:
    """Fast tokenizer only: no torch, Transformers, Unsloth or GPU dependency."""
    def __init__(self, backend):
        self.backend = backend
    def encode(self, text):
        return self.backend.encode_batch([text], add_special_tokens=False)[0].ids

    def document_chunks(self, doc, length, eos):
        text = doc['text']
        encoded = self.backend.encode_batch([text], add_special_tokens=False)[0]
        offsets = encoded.offsets
        cursor = token_cursor = index = 0
        stride = max(1, length - 8)
        while cursor < len(text):
            stop = min(token_cursor + stride, len(offsets))
            end = len(text) if stop == len(offsets) else offsets[stop][0]
            while end <= cursor and stop < len(offsets):
                stop += 1
                end = len(text) if stop == len(offsets) else offsets[stop][0]
            suffix = eos if end == len(text) else ''
            raw = text[cursor:end]
            piece = clean_for_bucket(raw + suffix, doc['bucket'])
            if len(self.encode(piece)) > length:
                fitted, rest = self.fit_prefix(raw, length, suffix, doc['bucket'] == 'code')
                if rest and suffix:
                    suffix = ''
                    fitted, rest = self.fit_prefix(raw, length - 1, preserve_whitespace=doc['bucket'] == 'code')
                consumed = len(raw) - len(rest)
                if consumed <= 0:
                    raise ValueError('Cannot fit document character in the token limit')
                end = cursor + consumed
                piece = clean_for_bucket(fitted, doc['bucket'])
            if piece:
                ids = self.encode(piece)
                if len(ids) > length:
                    raise ValueError('Chunk exceeded sequence length after cleaning')
                yield {'text': piece, 'source': doc['id'], 'bucket': doc['bucket'],
                       'chunk_index': index, 'tokens': len(ids),
                       **({'provenance': doc['provenance']} if 'provenance' in doc else {})}
                index += 1
            cursor = end
            token_cursor = bisect_left(offsets, (cursor, -1))

    def fit_prefix(self, text, limit, suffix='', preserve_whitespace=False):
        encoded = self.backend.encode_batch([text], add_special_tokens=False)[0]
        suffix_length = len(self.encode(suffix))
        if len(encoded.ids) + suffix_length <= limit:
            return text + suffix, ''
        capacity = limit - suffix_length
        if capacity <= 0:
            return '', text
        boundary = min(len(text), encoded.offsets[min(capacity, len(encoded.ids) - 1)][0])
        if boundary and len(self.encode(clean_for_bucket(text[:boundary] + suffix, 'code' if preserve_whitespace else 'domain'))) > limit:
            low, high = 0, boundary
            while low < high:
                middle = (low + high + 1) // 2
                if len(self.encode(clean_for_bucket(text[:middle] + suffix, 'code' if preserve_whitespace else 'domain'))) <= limit:
                    low = middle
                else:
                    high = middle - 1
            boundary = low
        return (text[:boundary] + suffix, text[boundary:]) if boundary else ('', text)


def publish(output, repo, private=True):
    from datasets import Dataset, DatasetDict, Features, Value
    from huggingface_hub import HfApi, hf_hub_download
    token = os.getenv('HF_TOKEN')
    manifest = json.loads((output / 'manifest.json').read_text())
    features = Features({field: Value(dtype) for field, dtype in {
        'text': 'string', 'source': 'string', 'bucket': 'string',
        'chunk_index': 'int64', 'tokens': 'int64', 'provenance': 'string'}.items()})
    def records(path):
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                row['provenance'] = json.dumps(row['provenance'], ensure_ascii=False) if row.get('provenance') else None
                yield row
    data = DatasetDict({split: Dataset.from_generator(records, gen_kwargs={'path': str(output / f'{split}.jsonl')},
                          features=features, cache_dir=str(ROOT / '.cache/huggingface/datasets'))
                        for split in ['train', 'validation']})
    data.push_to_hub(repo, private=private, token=token)
    api = HfApi(token=token)
    if api.dataset_info(repo).private != private:
        raise ValueError('Uploaded repository visibility does not match the requested setting')
    api.upload_file(repo_id=repo, repo_type='dataset', path_or_fileobj=str(output / 'manifest.json'),
                    path_in_repo='preparation_manifest.json')
    lines = ['---', 'language:', '- vi', '- en', 'task_categories:', '- text-generation', '---', '',
             ('# Traditional-medicine CPT with general replay' if len(manifest['buckets']) > 1 else '# Traditional-medicine CPT only'), '',
             'Raw-text continued-pretraining corpus. No instruction examples or model evaluations are included.', '',
             f"Tokenizer: `{manifest['config']['tokenizer']['name']}` at `{manifest['config']['tokenizer']['revision']}`.",
             f"Training corpus: {manifest['actual_train_tokens']:,} tokens in {manifest['train_rows']:,} rows.",
             f"Validation: {manifest['validation_rows']:,} rows. Counts precede trainer packing and epoch repetition.", '',
             '| Source | Training tokens | Actual share |', '|---|---:|---:|']
    for bucket, info in manifest['buckets'].items():
        lines.append(f"| {bucket} | {info['actual_tokens']:,} | {info['actual_share']:.6%} |")
    lines += ['', 'See `preparation_manifest.json` for source revisions, filtering, hashes and token accounting.',
              'The `provenance` column contains JSON-encoded source metadata when available.', '',
              '## Source terms', '',
              'Each source retains its own terms and licenses. Code license/provenance metadata is retained.',
              ('Code derives from [The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup); its terms are included in `THE_STACK_TERMS.md` and further sharing must preserve them and require agreement.' if 'code' in manifest['buckets'] else 'This dataset contains only the medical rows of the matching cleaned corpus.'), '',
              '## Limitations', '',
              'No claim of preserved reasoning or instruction-following performance is made.',
              'VM14K question matching and known-ID exclusion are incomplete contamination checks.',
              'MinHash deduplication is approximate. HF source revisions and exported corpus hashes are recorded; external Dolma objects are not cryptographically pinned by their URL listing.']
    # Preserve split/file/feature metadata generated by DatasetDict.push_to_hub.
    import yaml
    automatic_card = Path(hf_hub_download(repo, 'README.md', repo_type='dataset', token=token)).read_text()
    metadata = yaml.safe_load(automatic_card.split('---', 2)[1]) or {}
    metadata.update(language=['vi', 'en'], task_categories=['text-generation'])
    body_start = lines.index('---', 1) + 1
    card = '---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n' + '\n'.join(lines[body_start:]) + '\n'
    api.upload_file(repo_id=repo, repo_type='dataset', path_or_fileobj=card.encode(), path_in_repo='README.md')
    for source in manifest['sources']:
        if 'code' in manifest['buckets'] and source.get('repo') == 'bigcode/the-stack-dedup':
            card = Path(hf_hub_download(source['repo'], 'README.md', repo_type='dataset',
                                       revision=source['revision'], token=token)).read_text()
            marker = '## Terms of Use for The Stack'
            position = card.rfind(marker)
            if position < 0:
                raise ValueError('Could not locate upstream code redistribution terms')
            api.upload_file(repo_id=repo, repo_type='dataset', path_or_fileobj=card[position:].encode(),
                            path_in_repo='THE_STACK_TERMS.md')
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data/cpt_mix.yaml')
    parser.add_argument('--push-to-hub', metavar='OWNER/DATASET', help='Upload the newly prepared dataset')
    parser.add_argument('--public', action='store_true', help='Use a public repository for --push-to-hub')
    parser.add_argument('--upload-only', type=Path, help='Upload an existing prepared directory without rebuilding')
    args = parser.parse_args()
    # Keep all caches/temp files in the repository.
    for name, suffix in [('HF_HOME', 'huggingface'), ('TMPDIR', 'tmp'), ('PIP_CACHE_DIR', 'pip')]:
        path = ROOT / '.cache' / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    try:
        if args.upload_only:
            if not args.push_to_hub:
                parser.error('--upload-only requires --push-to-hub')
            output = (ROOT / args.upload_only).resolve()
            if not output.is_relative_to(ROOT):
                raise ValueError('Upload directory must be inside the repository')
            report = json.loads((output / 'manifest.json').read_text())
            for split, checksum in report['files'].items():
                if sha256(output / f'{split}.jsonl') != checksum:
                    raise ValueError('Prepared dataset hash mismatch')
        else:
            config = load_config(args.config)
            # Report missing source files and credentials before downloading corpora.
            missing = []
            for source in config['sources'].values():
                for spec in source['inputs']:
                    if spec['kind'] == 'local' and not (ROOT / spec['path']).is_file():
                        missing.append(f"Missing local input: {spec['path']}")
                if source.get('hydrate_stack_v2'):
                    try:
                        stack_s3_client()
                    except ValueError as error:
                        missing.append(str(error))
            if missing:
                raise ValueError('Missing prerequisites:\n- ' + '\n- '.join(missing))
            from huggingface_hub import HfApi, hf_hub_download
            from tokenizers import Tokenizer as FastTokenizer
            spec = config['tokenizer']
            revision = HfApi(token=os.getenv('HF_TOKEN')).model_info(spec['name'], revision=spec.get('revision', 'main')).sha
            config['tokenizer']['revision'] = revision
            file = hf_hub_download(spec['name'], 'tokenizer.json', revision=revision, token=os.getenv('HF_TOKEN'))
            backend = FastTokenizer.from_file(file)
            backend.no_padding()
            backend.no_truncation()
            eos = spec['eos_token']
            if backend.token_to_id(eos) is None:
                raise ValueError('Configured EOS token is absent from the tokenizer')
            output, report = prepare(config, Tokenizer(backend), eos, os.getenv('HF_TOKEN'))
        print(json.dumps({'output': str(output), 'train_tokens': report['actual_train_tokens'], 'buckets': report['buckets']}, indent=2))
        if args.push_to_hub:
            publish(output, args.push_to_hub, private=not args.public)
            print(f'https://huggingface.co/datasets/{args.push_to_hub}')
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
