"""CPU tests for token budgets, split isolation, pinned artifacts and Lambda routing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'CPT'), str(ROOT / 'scripts')]
import pilot_data as data
import cpt_pilot as pilot
import lambda_launch as launch
import prepare_cpt_data as preparation
import fetch_cpt_sources as acquisition

CONFIG = ROOT / 'configs/experiments/qwen3.5-4B/cpt-pilot/mixed.yaml'


class DataTests(unittest.TestCase):
    def test_exact_allocation_and_reproducible_packing(self):
        budget = data.allocate(29_000_000, 'mixed')
        self.assertEqual(budget, {'domain': 29_000_000, 'fineweb': 4_350_000,
                                 'dolma': 1_450_000, 'math': 725_000, 'code': 725_000})
        self.assertEqual(data.allocate(29_000_000, 'domain-only'), {'domain': 36_250_000})
        with tempfile.TemporaryDirectory() as directory:
            for bucket in ['domain', 'fineweb']:
                (Path(directory) / f'{bucket}-train.jsonl').write_text(json.dumps({'input_ids': [1, 2, 3, 9]}) + '\n')
            budgets = {'domain': 13, 'fineweb': 7}
            rows = list(data.packed_rows(directory, budgets, 6, 42))
            self.assertEqual(rows, list(data.packed_rows(directory, budgets, 6, 42)))
            self.assertEqual(sum(len(row['input_ids']) for row in rows), 20)
            self.assertEqual([len(row['input_ids']) for row in rows], [6, 6, 6, 2])
            tail = list(data.packed_rows(directory, {'domain': 7}, 6, 42))
            self.assertEqual([len(row['input_ids']) for row in tail], [5, 2])
            for bucket, count in budgets.items():
                self.assertEqual(sum(row['bucket_counts'][bucket] for row in rows), count)

    def test_exact_and_near_deduplication(self):
        deduper = data.Deduper()
        text = ' '.join(f'word{i}' for i in range(200))
        self.assertFalse(deduper.duplicate(text))
        self.assertTrue(deduper.duplicate(text.upper()))
        self.assertTrue(deduper.duplicate(text + ' another word'))
        self.assertFalse(deduper.duplicate('distinct unrelated document'))

    def make_manifest(self, directory):
        sources = []
        for bucket in data.SHARES:
            files = []
            for split in ['train', 'validation']:
                path = directory / f'{bucket}-{split}.jsonl'
                row = {'text': f'{bucket} {split} unique sample document for testing',
                       'document_id': f'{bucket}-{split}', 'component': 'books',
                       'license': 'MIT', 'language': 'Python'}
                path.write_text(json.dumps(row) + '\n')
                files.append({'path': path.name, 'sha256': data.digest(path), 'split': split})
            sources.append({'bucket': bucket, 'repo': 'test/source', 'revision': 'a' * 40,
                            'selection_notes': 'Test fixture', 'files': files,
                            'allowed_licenses': ['MIT'], 'allowed_languages': ['Python'],
                            'allowed_components': ['books']})
        path = directory / 'manifest.json'
        path.write_text(json.dumps({'sources': sources}))
        return path

    def test_preparation_and_manifest_integrity(self):
        class Tokenizer:
            eos_token_id = 9
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            directory = Path(directory)
            manifest = self.make_manifest(directory)
            stats = data.prepare(manifest, Tokenizer(), directory / 'prepared')
            self.assertEqual(set(stats['buckets']), set(data.SHARES))
            for bucket in data.SHARES:
                rows = [json.loads(line) for line in (directory / 'prepared' / f'{bucket}-train.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['input_ids'][-1], 9)
            (directory / 'domain-train.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                data.load_manifest(manifest)

    def test_benchmark_and_identified_source_documents_are_removed(self):
        class Tokenizer:
            eos_token_id = 9
            def encode(self, text, add_special_tokens=False):
                return [1] * len(text.split())
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            directory = Path(directory)
            path = self.make_manifest(directory)
            manifest = json.loads(path.read_text())
            question = json.loads(data.VM14K.read_text().splitlines()[0])['question']
            file = directory / 'domain-train.jsonl'
            with file.open('a') as stream:
                for row in [
                    {'document_id': 'leaked', 'text': 'Some preceding material ' + question + ' following material'},
                    {'document_id': 'excluded-article', 'text': 'otherwise unrelated text from held out source'}]:
                    stream.write(json.dumps(row) + '\n')
            manifest['excluded_source_ids'] = ['excluded-article']
            manifest['sources'][0]['files'][0]['sha256'] = data.digest(file)
            path.write_text(json.dumps(manifest))
            stats = data.prepare(path, Tokenizer(), directory / 'prepared')
            self.assertEqual(stats['filters']['benchmark_overlap'], 1)
            self.assertEqual(stats['filters']['policy_or_source_excluded'], 1)
            self.assertEqual(stats['buckets']['domain']['train']['documents'], 1)

    def test_group_cannot_cross_established_splits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            directory = Path(directory)
            path = self.make_manifest(directory)
            manifest = json.loads(path.read_text())
            row = json.loads((directory / 'domain-validation.jsonl').read_text())
            row['document_id'] = 'domain-train'
            (directory / 'domain-validation.jsonl').write_text(json.dumps(row) + '\n')
            manifest['sources'][0]['files'][1]['sha256'] = data.digest(directory / 'domain-validation.jsonl')
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'crosses established splits'):
                data.prepare(path, None, directory / 'out')


class PreparationTests(unittest.TestCase):
    def make_config(self, directory):
        import yaml
        manifest = DataTests().make_manifest(directory)
        output = directory / 'prepared'
        config = {'seed': 42, 'tokenizer': {'name': 'test/tokenizer', 'revision': 'a' * 40},
                  'data': {'source_manifest': str(manifest.relative_to(ROOT)),
                           'output_dir': str(output.relative_to(ROOT))}}
        path = directory / 'prepare.yaml'
        path.write_text(yaml.safe_dump(config))
        return path, output

    def test_preparation_config_has_no_training_dependency(self):
        config = preparation.load_config(ROOT / 'configs/data/cpt-pilot/prepare.yaml')
        self.assertNotIn('training', config)
        self.assertNotIn('evaluation', config)
        self.assertNotIn('model', config)
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            config_path, output = self.make_config(Path(directory))
            with patch.dict(sys.modules, {'transformers': None, 'unsloth': None}):
                preparation.run(config_path, check=True)
            self.assertFalse(output.exists())

    def test_standalone_preparation_produces_training_contract(self):
        import types
        from unittest.mock import MagicMock
        tokenizer = MagicMock(eos_token_id=9)
        tokenizer.encode.side_effect = lambda text, **kwargs: [1] * len(text.split())
        auto = MagicMock()
        auto.from_pretrained.return_value = tokenizer
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            config_path, output = self.make_config(Path(directory))
            with patch.dict(sys.modules, {'transformers': types.SimpleNamespace(AutoTokenizer=auto)}):
                stats = preparation.run(config_path)
            self.assertEqual(stats['tokenizer_revision'], 'a' * 40)
            self.assertEqual(set(stats['buckets']), set(data.SHARES))
            self.assertTrue((output / 'preparation.json').is_file())
            self.assertTrue((output / 'prepare_config.json').is_file())
            with self.assertRaises(FileExistsError):
                preparation.run(config_path)

    def test_failed_preparation_does_not_publish_partial_snapshot(self):
        import types
        from unittest.mock import MagicMock
        auto = MagicMock()
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            config_path, output = self.make_config(Path(directory))
            with patch.dict(sys.modules, {'transformers': types.SimpleNamespace(AutoTokenizer=auto)}), \
                 patch.object(preparation, 'prepare', side_effect=ValueError('filter failure')):
                with self.assertRaisesRegex(ValueError, 'filter failure'):
                    preparation.run(config_path)
            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob('.prepare-*')))


class AcquisitionTests(unittest.TestCase):
    def test_code_license_filter_runs_before_content_download(self):
        source = next(s for s in acquisition.load_config(ROOT / 'configs/data/cpt-pilot/upstream.yaml')['sources']
                      if s['bucket'] == 'code')
        row = {'blob_id': 'blob', 'language': 'Python', 'detected_licenses': ['GPL-3.0'],
               'license_type': 'permissive'}
        with patch.object(acquisition, 'hydrate_code', return_value='print(1)') as hydrate:
            self.assertIsNone(acquisition.normalize(source, row))
            hydrate.assert_not_called()
            row['detected_licenses'] = ['MIT']
            normalized = acquisition.normalize(source, row)
            self.assertEqual(normalized['license'], 'MIT')
            self.assertEqual(normalized['text'], 'print(1)')
            hydrate.assert_called_once()

    def test_medical_grouping_and_fineweb_quality(self):
        config = acquisition.load_config(ROOT / 'configs/data/cpt-pilot/upstream.yaml')
        domain = next(s for s in config['sources'] if s['bucket'] == 'domain')
        row = acquisition.normalize(domain, {'text': 'medical text', 'source': 'article', 'chunk_index': 2})
        self.assertTrue(row['document_id'].endswith(':article'))
        self.assertEqual(row['chunk_index'], 2)
        with self.assertRaisesRegex(ValueError, 'source and chunk_index'):
            acquisition.normalize(domain, {'text': 'medical text'})
        fineweb = next(s for s in config['sources'] if s['bucket'] == 'fineweb')
        self.assertIsNone(acquisition.normalize(fineweb, {'score': 2, 'text': 'low quality'}))

    def test_missing_aws_credentials_fail_before_network(self):
        with patch.dict('os.environ', {}, clear=True), patch('requests.head') as head:
            with self.assertRaisesRegex(ValueError, 'AWS_ACCESS_KEY_ID'):
                acquisition.check_access(acquisition.load_config(ROOT / 'configs/data/cpt-pilot/upstream.yaml'))
            head.assert_not_called()

    def test_lambda_bundle_needs_configs_but_no_local_corpus(self):
        config = pilot.preflight(CONFIG, launch=True)
        self.assertIn('preparation_config', config['pilot'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bundle.tar.gz'
            launch.build_cpt_pilot_bundle(str(CONFIG), False, path)
            with tarfile.open(path) as archive:
                self.assertIn('scripts/fetch_cpt_sources.py', archive.getnames())
                self.assertIn('scripts/prepare_cpt_data.py', archive.getnames())
                self.assertIn('configs/data/cpt-pilot/upstream.yaml', archive.getnames())
                self.assertFalse(any(name.startswith('CPT/pilot_data/') for name in archive.getnames()))

    def test_fetch_writes_hashed_five_source_manifest(self):
        import types
        import yaml
        from unittest.mock import MagicMock
        config = acquisition.load_config(ROOT / 'configs/data/cpt-pilot/upstream-smoke.yaml')
        config.update(max_domain_tokens=4, min_replay_documents=1, replay_headroom=1)
        def rows(source, split='train'):
            for i in range(5000):
                row = {'id': str(i), 'text': f'unique {source["bucket"]} document {i}',
                       'source': f'{split}-article-{i}', 'chunk_index': 0, 'score': 3,
                       'blob_id': str(i), 'language': 'Python', 'detected_licenses': ['MIT'],
                       'license_type': 'permissive', '_component': 'books', '_download_url': 'https://example.com/book'}
                yield row
        with tempfile.TemporaryDirectory(dir=ROOT / 'CPT') as directory:
            directory = Path(directory)
            spec = directory / 'upstream.yaml'; spec.write_text(yaml.safe_dump(config))
            tokenizer = MagicMock()
            tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
            with patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test', 'AWS_SECRET_ACCESS_KEY': 'test'}), \
                 patch.dict(sys.modules, {'boto3': types.SimpleNamespace(client=MagicMock())}), \
                 patch.object(acquisition, 'hf_rows', side_effect=rows), \
                 patch.object(acquisition, 'dolma_rows', side_effect=rows), \
                 patch.object(acquisition, 'hydrate_code', side_effect=lambda row, client: f'print({row["blob_id"]})'):
                manifest_path = directory / 'sources.json'
                acquisition.fetch(spec, manifest_path, tokenizer)
            manifest = data.load_manifest(manifest_path)
            self.assertEqual({source['bucket'] for source in manifest['sources']}, set(data.SHARES))
            for source in manifest['sources']:
                for file in source['files']:
                    self.assertEqual(data.digest(directory / file['path']), file['sha256'])

    def test_same_job_success_orders_fetch_train_evaluate_publish(self):
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        function = source[source.index('run_cpt_pilot() {'):source.index('\nrun_cpt() {')]
        with tempfile.TemporaryDirectory() as directory:
            calls = Path(directory) / 'calls'
            shell = f'''REPO_DIR={directory}
JOB_TAG=test-job
CONFIG_LIST=configs/test.yaml
CPT_PILOT_PREPARATION_CONFIG=configs/data/cpt-pilot/prepare.yaml
in_docker() {{ printf '%s\\n' "$1" >> '{calls}'; return 0; }}
{function}
run_cpt_pilot
'''
            result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = calls.read_text()
            positions = [commands.index(value) for value in ['prepare_cpt_data.py', '/CPT/CPT.py',
                         'cpt_pilot.py evaluate', 'cpt_pilot.py publish']]
            self.assertEqual(positions, sorted(positions))

    def test_lambda_preparation_failure_prevents_training(self):
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        function = source[source.index('run_cpt_pilot() {'):source.index('\nrun_cpt() {')]
        with tempfile.TemporaryDirectory() as directory:
            calls = Path(directory) / 'calls'
            shell = f'''REPO_DIR={directory}
JOB_TAG=test-job
CONFIG_LIST=configs/test.yaml
CPT_PILOT_PREPARATION_CONFIG=configs/data/cpt-pilot/prepare.yaml
in_docker() {{ printf '%s\\n' "$1" >> '{calls}'; case "$1" in *prepare_cpt_data.py*) return 9;; esac; return 0; }}
{function}
run_cpt_pilot
'''
            result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True)
            self.assertEqual(result.returncode, 9, result.stderr)
            commands = calls.read_text()
            self.assertIn('--fetch-sources', commands)
            self.assertNotIn('/CPT/CPT.py', commands)
            self.assertNotIn('cpt_pilot.py evaluate', commands)
            self.assertIn('--diagnostics', commands)


class EvaluationTests(unittest.TestCase):
    def test_only_final_completed_answers_score(self):
        self.assertEqual(pilot.extract_final('Final Answer: B</think>Final Answer: A<|im_end|>', False), 'A')
        self.assertIsNone(pilot.extract_final('</think>Final Answer: A', True))
        self.assertIsNone(pilot.extract_final('Final Answer: A', False))
        self.assertIsNone(pilot.extract_final('</think>unparseable', False))

    def test_baseline_preflight_does_not_require_training_data(self):
        config = pilot.preflight(CONFIG, baseline=True)
        self.assertEqual(config['evaluation']['dataset'], 'vm14k')
        with patch.object(pilot, 'ROOT', Path('/nonexistent')):
            with self.assertRaisesRegex(FileNotFoundError, 'Lambda preparation stage'):
                pilot.preflight(CONFIG)

    def test_compare_rejects_smoke_and_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            summary = {'smoke_only': True, 'total': 2}
            (directory / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'complete'):
                pilot.compare([directory] * 3, directory / 'report.md')

    def test_full_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for label, correct in [('original', True), ('domain', True), ('mixed', False)]:
                path = root / label; path.mkdir(); paths.append(path)
                summary = {'smoke_only': False, 'total': 564, 'correct': 564 if correct else 0,
                           'accuracy': float(correct), 'invalid_rate': 0, 'truncation_rate': 0,
                           'condition': {'original': 'original', 'domain': 'domain-only', 'mixed': 'mixed'}[label], 'preparation_sha256': 'shared-cleaned-snapshot' if label != 'original' else None,
                           'dataset_sha256': 'data', 'protocol_sha256': 'protocol', 'model': {}, 'generation': {}}
                (path / 'summary.json').write_text(json.dumps(summary))
                (path / 'predictions.jsonl').write_text('\n'.join(json.dumps({'id': str(i), 'answer': 'A', 'is_correct': correct}) for i in range(564)))
            pilot.compare(paths, root / 'report.md')
            report = (root / 'report.md').read_text()
            self.assertIn('-100.00 [-100.00, -100.00]', report)
            summary = json.loads((paths[2] / 'summary.json').read_text()); summary['generation'] = {'temperature': .8}
            (paths[2] / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'generation'):
                pilot.compare(paths, root / 'bad.md')


class LambdaTests(unittest.TestCase):
    def test_cpt_config_forwarding(self):
        self.assertEqual(launch.parse_configs(argparse.Namespace(config=str(CONFIG), config_list=None, task='cpt')), [str(CONFIG)])

    def test_baseline_bundle_contains_exact_sources_and_no_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bundle.tar.gz'
            manifest = launch.build_cpt_pilot_bundle(str(CONFIG), True, path)
            self.assertEqual(manifest['bundle_sha256'], data.digest(path))
            with tarfile.open(path) as archive:
                self.assertIn('CPT/CPT.py', archive.getnames())
                self.assertIn('CPT/pilot_source.json', archive.getnames())
                self.assertNotIn('.env', archive.getnames())
                self.assertNotIn('scripts/prepare_cpt_data.py', archive.getnames())
                self.assertNotIn('configs/data/cpt-pilot/prepare.yaml', archive.getnames())
                content = json.load(archive.extractfile('CPT/pilot_source.json'))
                for name, checksum in content['files'].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), checksum)

    def test_failed_remote_job_is_backed_up_then_terminated(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        api.launch.return_value = 'instance-test'
        events = []
        def remote(*args, **kwargs):
            command = args[2]
            return subprocess.CompletedProcess([], 0, stdout='DONE_FAILED' if 'ls /home/ubuntu/DONE' in command else '0', stderr='')
        def backup(*args):
            events.append('backup')
        api.terminate.side_effect = lambda *args: events.append('terminate')
        with tempfile.TemporaryDirectory() as directory, patch('os.getcwd', return_value=directory):
            # Route run metadata into a temporary logs directory without changing cwd.
            original_path = launch.Path
            def local_path(value, *args):
                return original_path(directory) / 'logs' if value == 'logs' else original_path(value, *args)
            with patch.object(launch, 'find_region_for', return_value='test'), \
                 patch.object(launch, 'wait_for_active', return_value={'ip': '127.0.0.1'}), \
                 patch.object(launch, 'wait_for_ssh'), patch.object(launch, 'scp_to'), \
                 patch.object(launch, 'ssh_cmd', side_effect=remote), \
                 patch.object(launch, 'download_cpt_pilot_backup', side_effect=backup), \
                 patch.object(launch, 'Path', side_effect=local_path):
                result = launch.run_job(api, 'eval', [str(CONFIG)], 'gpu-test', 'key-name', '/key',
                    {'CPT_PILOT': '1', 'CPT_PILOT_BASELINE': '1', 'HF_CPT_REPO': 'test/repo'},
                    'https://example.com/repo', 'main', False)
        self.assertEqual(result, 2)
        self.assertEqual(events, ['backup', 'terminate'])

    def test_bootstrap_vm14k_failure_is_fatal_and_diagnostics_attempted(self):
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        function = source[source.index('run_cpt_pilot() {'):source.index('\nrun_cpt() {')]
        with tempfile.TemporaryDirectory() as directory:
            calls = Path(directory) / 'calls'
            shell = f'''REPO_DIR={directory}
JOB_TAG=test-job
CONFIG_LIST=configs/test.yaml
CPT_PILOT_SMOKE=1
in_docker() {{ printf '%s\\n' "$1" >> '{calls}'; case "$1" in *cpt_pilot.py\\ evaluate*) return 7;; esac; return 0; }}
{function}
run_cpt_pilot
'''
            result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            commands = calls.read_text()
            self.assertIn('CPT.py --config configs/test.yaml', commands)
            self.assertIn('--diagnostics', commands)
            self.assertIn('--limit 2', commands)
            self.assertNotIn('Unsloth_QA_CPT_new', commands)


if __name__ == '__main__':
    unittest.main()
