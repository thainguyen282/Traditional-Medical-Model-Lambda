"""Standalone preparation tests; no model weights, network, training or upload."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_cpt_mix as prep


class CharTokenizer:
    def encode(self, text):
        return list(text.replace('<eos>', '#'))


class PreparationTests(unittest.TestCase):
    def test_budget_keeps_one_domain_pass_after_rounding(self):
        for size in [29_000_000, 101, 127, 1001]:
            budget = prep.budgets(size)
            self.assertEqual(budget['domain'], size)
            self.assertEqual(sum(budget.values()), __import__('math').ceil(size / .8))

    def test_unicode_chunks_preserve_text_and_final_eos(self):
        text = 'Y học cổ truyền Việt Nam 漢字 và thuốc'
        rows = list(prep.chunk_document({'text': text, 'id': 'book', 'bucket': 'domain'}, 8, CharTokenizer(), '<eos>'))
        self.assertTrue(rows[-1]['text'].endswith('<eos>'))
        self.assertTrue(all(row['tokens'] <= 8 for row in rows))
        self.assertNotIn('\ufffd', ''.join(row['text'] for row in rows))
        boundary = list(prep.chunk_document({'text': 'abcdefgh', 'id': 'book', 'bucket': 'domain'}, 8, CharTokenizer(), '<eos>'))
        self.assertTrue(boundary[-1]['text'].endswith('<eos>'))

    def test_exact_and_near_duplicates(self):
        deduper = prep.Deduper(.85)
        text = ' '.join(f'medical{i}' for i in range(200))
        self.assertFalse(deduper.duplicate(text))
        self.assertTrue(deduper.duplicate(text))
        self.assertTrue(deduper.duplicate(text + ' another word'))
        self.assertFalse(deduper.duplicate('completely different mathematical material'))

    def test_selection_requires_explicit_repetition(self):
        rows = [{'text': 'abc<eos>', 'tokens': 4, 'source': 'd', 'chunk_index': 0, 'bucket': 'code'}]
        with self.assertRaisesRegex(ValueError, 'explicitly enable repetition'):
            prep.select(rows, 8, CharTokenizer(), '<eos>', random.Random(42))
        selected, tokens, available = prep.select(rows, 8, CharTokenizer(), '<eos>', random.Random(42), repeat=True)
        self.assertEqual((len(selected), tokens, available), (2, 8, 4))

    def fixture(self, root):
        sources = {}
        rng = random.Random(17)
        for bucket in prep.WEIGHTS:
            inputs = []
            for role in ['train', 'validation']:
                path = root / f'{bucket}-{role}.jsonl'
                count = 4 if role == 'train' else 1
                with path.open('w') as stream:
                    for i in range(count):
                        text = ' '.join(''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=7)) for _ in range(30))
                        stream.write(json.dumps({'text': text, 'document_id': f'{bucket}-{role}-{i}',
                                                 'license': 'MIT', 'language': 'Python', 'component': 'books'}) + '\n')
                inputs.append({'kind': 'local', 'path': path.name, 'role': role})
            sources[bucket] = {'inputs': inputs, 'allowed_licenses': ['MIT'],
                               'allowed_languages': ['Python'], 'allowed_components': ['books']}
        return {'sources': sources, 'output_dir': 'prepared', 'seed': 42, 'sequence_length': 32,
                'validation_fraction': .01, 'near_duplicate_threshold': .85,
                'allow_repetition': False, 'total_tokens': None}

    def test_complete_output_is_reproducible_and_matches_loader_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with patch.object(prep, 'ROOT', root):
                output, report = prep.prepare(config, CharTokenizer(), '<eos>')
                train = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
                validation = [json.loads(line) for line in (output / 'validation.jsonl').read_text().splitlines()]
                self.assertFalse({row['source'] for row in train} & {row['source'] for row in validation})
                self.assertTrue(all(isinstance(row['text'], str) and row['tokens'] <= 32 for row in train + validation))
                self.assertEqual(sum(row['tokens'] for row in train), report['actual_train_tokens'])
                self.assertAlmostEqual(report['buckets']['domain']['actual_share'], .8, delta=.01)
                for bucket in prep.WEIGHTS:
                    self.assertAlmostEqual(report['buckets'][bucket]['actual_share'], prep.WEIGHTS[bucket], delta=.01)
                second = copy.deepcopy(config); second['output_dir'] = 'prepared-second'
                _, repeated = prep.prepare(second, CharTokenizer(), '<eos>')
                self.assertEqual(report['files'], repeated['files'])
                with self.assertRaises(FileExistsError):
                    prep.prepare(config, CharTokenizer(), '<eos>')

    def test_document_group_overlap_is_quarantined_to_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train.jsonl').write_text(json.dumps({'text': 'training chunk', 'source': 'book', 'chunk_index': 0}) + '\n')
            (root / 'validation.jsonl').write_text(json.dumps({'text': 'validation chunk', 'source': 'book', 'chunk_index': 1}) + '\n')
            spec = {'document_field': 'source', 'inputs': [
                {'kind': 'local', 'path': 'train.jsonl', 'role': 'train'},
                {'kind': 'local', 'path': 'validation.jsonl', 'role': 'validation'}]}
            with patch.object(prep, 'ROOT', root):
                rows = prep.collect('domain', spec, {'seed': 42, 'validation_fraction': .01}, [], None)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['split'], 'validation')
            self.assertEqual(rows[0]['text'], 'training chunk\nvalidation chunk')

    def test_benchmark_question_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            question = 'specific held out benchmark medical question'
            benchmark = root / 'benchmark.jsonl'; benchmark.write_text(json.dumps({'question': question}) + '\n')
            config['exclude_benchmarks'] = ['benchmark.jsonl']
            path = root / 'domain-train.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps({'document_id': 'leak', 'text': 'prefix ' + question + ' suffix'}) + '\n')
            with patch.object(prep, 'ROOT', root):
                output, report = prep.prepare(config, CharTokenizer(), '<eos>')
            self.assertEqual(report['filters']['benchmark_or_source'], 1)
            self.assertNotIn('leak', {json.loads(line)['source'] for line in (output / 'train.jsonl').read_text().splitlines()})

    def test_stack_license_filter_precedes_content_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'stack.jsonl'
            row = {'blob_id': 'blob', 'detected_licenses': ['GPL-3.0'], 'language': 'Python',
                   'license_type': 'permissive', 'src_encoding': 'utf-8', 'length_bytes': 12}
            path.write_text(json.dumps(row) + '\n')
            spec = {'hydrate_stack_v2': True, 'document_field': 'blob_id',
                    'license_field': 'detected_licenses', 'allowed_licenses': ['MIT'],
                    'allowed_languages': ['Python'],
                    'inputs': [{'kind': 'local', 'path': 'stack.jsonl', 'role': 'train'}]}
            with patch.object(prep, 'ROOT', root), patch.object(prep, 'hydrate_stack_v2') as hydrate:
                self.assertEqual(prep.collect('code', spec, {'seed': 42, 'validation_fraction': .01}, [], None), [])
                hydrate.assert_not_called()
            row['detected_licenses'] = ['MIT']; row['repo_name'] = 'owner/repo'; row['path'] = 'file.py'
            path.write_text(json.dumps(row) + '\n')
            with patch.object(prep, 'ROOT', root), patch.object(prep, 'hydrate_stack_v2', return_value=('print(1)', 'checksum')):
                documents = prep.collect('code', spec, {'seed': 42, 'validation_fraction': .01}, [], None)
            self.assertEqual(documents[0]['provenance']['repo_name'], 'owner/repo')
            chunks = list(prep.chunk_document(documents[0], 32, CharTokenizer(), '<eos>'))
            self.assertEqual(chunks[0]['provenance']['content_sha256'], 'checksum')

    def test_stack_content_is_decompressed_and_body_closed(self):
        import gzip
        import io
        from unittest.mock import MagicMock
        body = io.BytesIO(gzip.compress('print("Việt Nam")'.encode()))
        client = MagicMock()
        client.get_object.return_value = {'Body': body}
        with patch.object(prep, 'stack_s3_client', return_value=client):
            text, checksum = prep.hydrate_stack_v2({'blob_id': 'abc', 'src_encoding': 'utf-8'})
        self.assertEqual(text, 'print("Việt Nam")')
        self.assertEqual(checksum, __import__('hashlib').sha256(text.encode()).hexdigest())
        self.assertTrue(body.closed)
        client.get_object.assert_called_once_with(Bucket='softwareheritage', Key='content/abc')

    def test_stack_s3_access_failure_is_not_silently_skipped(self):
        from botocore.exceptions import ClientError
        from unittest.mock import MagicMock
        client = MagicMock()
        client.get_object.side_effect = ClientError({'Error': {'Code': 'AccessDenied'}}, 'GetObject')
        with patch.object(prep, 'stack_s3_client', return_value=client):
            with self.assertRaisesRegex(ValueError, 'AccessDenied'):
                prep.hydrate_stack_v2({'blob_id': 'abc'})

    def test_missing_aws_credentials_are_reported(self):
        from unittest.mock import MagicMock
        session = MagicMock()
        session.get_credentials.return_value = None
        prep.stack_s3_client.cache_clear()
        try:
            with patch('boto3.Session', return_value=session):
                with self.assertRaisesRegex(ValueError, 'AWS credentials'):
                    prep.stack_s3_client()
            session.client.assert_not_called()
        finally:
            prep.stack_s3_client.cache_clear()

    def test_direct_dolma_reader_handles_nested_gzip(self):
        import gzip
        import io
        import types
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'urls.txt'
            listing.write_text('https://example.com/books/book.json.gz\n')
            payload = json.dumps({'id': 'book', 'text': 'sample book text'}).encode() + b'\n'
            response = MagicMock()
            response.__enter__.return_value = response
            response.raw = io.BytesIO(gzip.compress(gzip.compress(payload)))
            api = MagicMock()
            api.dataset_info.return_value.sha = 'a' * 40
            with patch('huggingface_hub.HfApi', return_value=api), \
                 patch('huggingface_hub.hf_hub_download', return_value=str(listing)), \
                 patch('requests.get', return_value=response):
                rows = list(prep.input_rows({'kind': 'dolma_urls', 'repo': 'test/dolma',
                            'url_list': 'urls.txt', 'components': {'books': 'books'}}, [], None))
            self.assertEqual(rows[0]['text'], 'sample book text')
            self.assertEqual(rows[0]['component'], 'books')

    def test_fast_tokenizer_prefix_respects_unicode_and_budget(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator(['Y học cổ truyền Việt Nam 漢字 và thuốc'] * 3,
                trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                    special_tokens=['[UNK]', '<eos>'], show_progress=False))
        tokenizer = prep.Tokenizer(backend)
        text = 'Y học cổ truyền Việt Nam 漢字 và thuốc'
        for limit in range(2, 20):
            prefix, rest = prep.fit_prefix(text, limit, tokenizer)
            self.assertEqual(prefix + rest, text)
            self.assertLessEqual(len(tokenizer.encode(prefix)), limit)
        doc = {'text': prep.clean((text + '\n') * 100), 'id': 'unicode-book', 'bucket': 'domain'}
        rows = list(prep.chunk_document(doc, 32, tokenizer, '<eos>'))
        self.assertTrue(rows[-1]['text'].endswith('<eos>'))
        self.assertEqual(sum(row['text'].count('<eos>') for row in rows), 1)
        self.assertTrue(all(row['tokens'] <= 32 for row in rows))
        restored = ''.join(row['text'].replace('<eos>', '') for row in rows)
        self.assertEqual(''.join(restored.split()), ''.join(doc['text'].split()))

    def test_code_indentation_survives_collection_and_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = 'def example():\n    value = 1\n    return value\n'
            (root / 'code.jsonl').write_text(json.dumps({'text': text, 'document_id': 'code', 'license': 'MIT', 'language': 'Python'}) + '\n')
            spec = {'allowed_licenses': ['MIT'], 'allowed_languages': ['Python'],
                    'inputs': [{'kind': 'local', 'path': 'code.jsonl', 'role': 'train'}]}
            with patch.object(prep, 'ROOT', root):
                documents = prep.collect('code', spec, {'seed': 42, 'validation_fraction': .01}, [], None)
            self.assertIn('\n    return value', documents[0]['text'])
            chunks = list(prep.chunk_document(documents[0], 128, CharTokenizer(), '<eos>'))
            self.assertIn('\n    return value', chunks[0]['text'])

    def test_unlicensed_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'code.jsonl'
            path.write_text(json.dumps({'text': 'code contents', 'document_id': 'code', 'license': 'GPL-3.0', 'language': 'Python'}) + '\n')
            spec = {'allowed_licenses': ['MIT'], 'allowed_languages': ['Python'],
                    'inputs': [{'kind': 'local', 'path': 'code.jsonl'}]}
            with patch.object(prep, 'ROOT', root):
                self.assertEqual(prep.collect('code', spec, {'seed': 42, 'validation_fraction': .01}, [], None), [])


if __name__ == '__main__':
    unittest.main()
