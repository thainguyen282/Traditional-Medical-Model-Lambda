"""CPU-only preparation for the token-controlled CPT pilot.

Inputs are reviewed JSONL exports, not unrestricted downloads of entire web/code
corpora. File hashes and upstream commits form the reproducible data snapshot.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VM14K = ROOT / 'EVAL/Data/eastern_medicine_VM14K.jsonl'
VM14K_SHA = '314006d439047ac9b98a890fe14ee1954025b28040c3e162ef71dc771af56574'
SHARES = {'domain': .8, 'fineweb': .12, 'dolma': .04, 'math': .02, 'code': .02}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).lower()))


def shingles(text, width=8):
    words = canonical(text).split()
    return {hashlib.blake2b(' '.join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(max(1, len(words) - width + 1))}


class Deduper:
    """Exact hashes plus 64-value MinHash / 16-band LSH; verify candidates."""
    def __init__(self):
        self.exact = set()
        self.bands = defaultdict(set)
        self.documents = []

    def duplicate(self, text):
        normalized = canonical(text)
        key = hashlib.sha256(normalized.encode()).hexdigest()
        if key in self.exact:
            return True
        grams = shingles(normalized, 5)
        # Deterministic independent hashes, independent of Python hash seed.
        sig = tuple(min(int.from_bytes(hashlib.blake2b(g, digest_size=8,
                      person=i.to_bytes(8, 'little')).digest(), 'little') for g in grams)
                    for i in range(64))
        bands = [(i, sig[i * 4:(i + 1) * 4]) for i in range(16)]
        candidates = set().union(*(self.bands[b] for b in bands))
        if any(len(grams & self.documents[i]) / len(grams | self.documents[i]) >= .8
               for i in candidates):
            return True
        self.exact.add(key)
        index = len(self.documents)
        self.documents.append(grams)
        for band in bands:
            self.bands[band].add(index)
        return False


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    sources = manifest['sources']
    if {s['bucket'] for s in sources} != set(SHARES) or len(sources) != 5:
        raise ValueError('Manifest must contain exactly domain, fineweb, dolma, math, code')
    for source in sources:
        if not re.fullmatch(r'[a-f0-9]{40}', source.get('revision', '')):
            raise ValueError(f"Immutable upstream commit required: {source['bucket']}")
        if not source.get('repo') or not source.get('selection_notes'):
            raise ValueError('Record upstream repo and quality/component selection notes')
        if not source.get('files'):
            raise ValueError('Each source must have at least one frozen export')
        for field in ({'code': ['allowed_licenses', 'allowed_languages'],
                       'dolma': ['allowed_components']}.get(source['bucket'], [])):
            if not source.get(field):
                raise ValueError(f'{source["bucket"]}.{field} must be nonempty')
        for item in source['files']:
            file = (path.parent / item['path']).resolve()
            if not file.is_relative_to(ROOT):
                raise ValueError('Pilot exports must be inside the repository for Lambda bundling')
            if digest(file) != item['sha256']:
                raise ValueError(f'Export hash mismatch: {file}')
            if item.get('split') not in ('train', 'validation', 'unsplit'):
                raise ValueError('File split must be train, validation, or unsplit')
    return manifest


def prepare(manifest_path, tokenizer, output, seed=42):
    """Group chunks, filter, decontaminate, split, deduplicate, and tokenize."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if digest(VM14K) != VM14K_SHA:
        raise ValueError('VM14K frozen subset hash mismatch')
    benchmark = [json.loads(line) for line in VM14K.read_text().splitlines() if line.strip()]
    short_questions = [canonical(row['question']) for row in benchmark
                       if 4 <= len(canonical(row['question']).split()) < 8]
    forbidden = set().union(*(shingles(row['question']) for row in benchmark))
    forbidden.update(set().union(*(shingles(str(option)) for row in benchmark
                                  for option in row['options'] if len(canonical(str(option)).split()) >= 8)))
    excluded_ids = set(manifest.get('excluded_source_ids', []))
    groups = {}
    counts = Counter()
    for source in manifest['sources']:
        bucket = source['bucket']
        for item in source['files']:
            with open(manifest_path.parent / item['path']) as stream:
                for line in stream:
                    row = json.loads(line)
                    text = row.get('text', '').strip()
                    if not text:
                        counts['empty'] += 1
                        continue
                    group = str(row['document_id'])
                    # ID must identify the original article, not an individual chunk.
                    key = (bucket, group)
                    if key not in groups:
                        groups[key] = {'bucket': bucket, 'id': group, 'parts': {},
                                       'split': item['split'], 'reject': False}
                    doc = groups[key]
                    if doc['split'] != item['split']:
                        raise ValueError(f'Original document crosses established splits: {key}')
                    chunk_index = int(row.get('chunk_index', 0))
                    if chunk_index in doc['parts'] and doc['parts'][chunk_index] != text:
                        raise ValueError(f'Conflicting chunk index in original document: {key}/{chunk_index}')
                    doc['parts'][chunk_index] = text
                    if group in excluded_ids or str(row.get('source_id', '')) in excluded_ids:
                        doc['reject'] = True
                    if bucket == 'code' and (row.get('license') not in source['allowed_licenses'] or
                                             row.get('language') not in source['allowed_languages']):
                        doc['reject'] = True
                    if bucket == 'dolma' and row.get('component') not in source['allowed_components']:
                        doc['reject'] = True
    documents = []
    for doc in groups.values():
        if doc['reject']:
            counts['policy_or_source_excluded'] += 1
            continue
        doc['text'] = '\n'.join(text for _, text in sorted(doc.pop('parts').items()))
        normalized = ' ' + canonical(doc['text']) + ' '
        if shingles(doc['text']) & forbidden or any(' ' + question + ' ' in normalized for question in short_questions):
            counts['benchmark_overlap'] += 1
            continue
        if doc['split'] == 'unsplit':
            number = int(hashlib.sha256(f"{seed}:{doc['id']}".encode()).hexdigest(), 16)
            doc['split'] = 'validation' if number % 100 == 0 else 'train'
        documents.append(doc)
    # Validation wins over matching training documents across sources.
    documents.sort(key=lambda d: (d['split'] != 'validation', d['bucket'], d['id']))
    validation_ids = {doc['id'] for doc in documents if doc['split'] == 'validation'}
    deduper = Deduper()
    buckets = {name: {'train': [], 'validation': []} for name in SHARES}
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('Tokenizer must define an EOS document boundary')
    for doc in documents:
        if doc['split'] == 'train' and doc['id'] in validation_ids:
            counts['cross_source_validation_id'] += 1
            continue
        if deduper.duplicate(doc['text']):
            counts['duplicate'] += 1
            continue
        ids = tokenizer.encode(doc['text'], add_special_tokens=False) + [eos]
        buckets[doc['bucket']][doc['split']].append({'id': doc['id'], 'input_ids': ids})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = {'seed': seed, 'manifest_sha256': digest(manifest_path), 'vm14k_sha256': VM14K_SHA,
             'filters': dict(counts), 'sources': manifest['sources'], 'buckets': {},
             'decontamination_limit': '8-word overlap, full 4-7 word questions and supplied source IDs; unidentified source articles may remain'}
    for bucket, splits in buckets.items():
        stats['buckets'][bucket] = {}
        for split, rows in splits.items():
            if not rows:
                raise ValueError(f'No documents remain for {bucket}/{split}')
            path = output / f'{bucket}-{split}.jsonl'
            with path.open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            stats['buckets'][bucket][split] = {'documents': len(rows),
                'tokens': sum(len(row['input_ids']) for row in rows), 'sha256': digest(path)}
    (output / 'preparation.json').write_text(json.dumps(stats, indent=2))
    return stats


def allocate(domain_tokens, mode):
    total = math.ceil(domain_tokens / .8)
    if mode == 'domain-only':
        return {'domain': total}
    if mode != 'mixed':
        raise ValueError('mode must be mixed or domain-only')
    budgets = {name: int(total * fraction) for name, fraction in SHARES.items()}
    budgets['domain'] = domain_tokens
    budgets['fineweb'] += total - sum(budgets.values())
    return budgets


def packed_rows(directory, budgets, length, seed):
    """Yield exact token budgets, including one final partial sequence (no drop)."""
    if length < 2 or sum(budgets.values()) < 2:
        raise ValueError('Causal LM packing requires at least two tokens')
    directory = Path(directory)
    rng = random.Random(seed)
    streams = {}
    for bucket, target in budgets.items():
        rows = [json.loads(line)['input_ids'] for line in
                (directory / f'{bucket}-train.jsonl').read_text().splitlines()]
        rng.shuffle(rows)
        streams[bucket] = {'rows': rows, 'doc': 0, 'pos': 0, 'remaining': target}
    packed = []
    target_length = length - 1 if sum(budgets.values()) == length + 1 else length
    contributions = {bucket: 0 for bucket in budgets}
    while any(stream['remaining'] for stream in streams.values()):
        active = [key for key, stream in streams.items() if stream['remaining']]
        key = rng.choices(active, weights=[streams[k]['remaining'] for k in active])[0]
        stream = streams[key]
        doc = stream['rows'][stream['doc'] % len(stream['rows'])]
        take = min(len(doc) - stream['pos'], stream['remaining'], target_length - len(packed))
        packed.extend(doc[stream['pos']:stream['pos'] + take])
        contributions[key] += take
        stream['remaining'] -= take
        stream['pos'] += take
        if stream['pos'] == len(doc):
            stream['pos'] = 0
            stream['doc'] += 1
        if len(packed) == target_length:
            yield {'input_ids': packed, 'attention_mask': [1] * len(packed), 'labels': packed.copy(), 'bucket_counts': contributions}
            packed = []
            remaining = sum(stream['remaining'] for stream in streams.values())
            target_length = length - 1 if remaining == length + 1 else length
            contributions = {bucket: 0 for bucket in budgets}
    if packed:
        yield {'input_ids': packed, 'attention_mask': [1] * len(packed), 'labels': packed.copy(), 'bucket_counts': contributions}
