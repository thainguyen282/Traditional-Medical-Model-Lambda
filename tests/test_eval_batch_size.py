"""Validate CLI forwarding and notebook batch limits without a GPU or cloud job."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import lambda_launch as launcher

NOTEBOOKS = sorted((ROOT / 'EVAL').glob('Unsloth_QA_*.ipynb'))


class EvalBatchSizeTests(unittest.TestCase):
    def run_cli(self, extra):
        env = {
            'LAMBDA_API_KEY': 'test', 'LAMBDA_SSH_KEY_NAME': 'test',
            'LAMBDA_SSH_PRIVATE_KEY': str(Path(__file__).resolve()), 'HF_TOKEN': 'test',
        }
        with patch.dict(os.environ, env, clear=True), patch.object(launcher, 'load_dotenv'), \
                patch.object(launcher, 'LambdaAPI'), \
                patch.object(launcher, 'run_job', return_value=0) as run, \
                patch.object(sys, 'argv', ['lambda_launch.py', '--task', 'eval', *extra]):
            result = launcher.main()
        self.assertEqual(result, 0)
        return run.call_args.kwargs['env_vars']

    def test_cli_forwards_requested_cap(self):
        self.assertEqual(self.run_cli(['--eval-batch-size', '1'])['EVAL_BATCH_SIZE'], '1')

    def test_omitted_flag_keeps_notebook_defaults(self):
        self.assertEqual(self.run_cli([])['EVAL_BATCH_SIZE'], '')

    def test_invalid_cli_values_fail_before_api_initialization(self):
        for value in ['0', '-1', 'abc']:
            with self.subTest(value=value), patch.object(launcher, 'load_dotenv'), \
                    patch.object(launcher, 'LambdaAPI') as api, \
                    patch.object(sys, 'argv', ['lambda_launch.py', '--task', 'eval', '--eval-batch-size', value]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 2)
                api.assert_not_called()

    def test_docker_receives_cap(self):
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        function = 'in_docker() {' + source.split('in_docker() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        script = '''
set -u
DOCKER=mock_docker
REPO_DIR=/repo
HF_TOKEN=test
HF_CPT_REPO=cpt
HF_SFT_REPO=sft
JOB_TAG=test
IMAGE=test-image
EVAL_BATCH_SIZE=1
mock_docker() { printf '%s\\n' "$@"; }
'''
        result = subprocess.run(['bash', '-c', script + function + '\nin_docker true'],
                                text=True, capture_output=True, check=True)
        self.assertIn('EVAL_BATCH_SIZE=1', result.stdout.splitlines())

    def notebook_function(self, path, cap, infer):
        source = '\n'.join(''.join(c.get('source', [])) for c in
                           json.loads(path.read_text())['cells'] if c['cell_type'] == 'code')
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'batch_inference_unsloth')
        namespace = {
            'EVAL_BATCH_SIZE': cap, 'batch_inference': infer,
            'torch': SimpleNamespace(cuda=SimpleNamespace(
                OutOfMemoryError=MemoryError, empty_cache=Mock())),
            'gc': SimpleNamespace(collect=Mock()), 'tqdm': Mock(return_value=Mock()),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['batch_inference_unsloth']

    def samples(self, count):
        return [{'question': str(i), 'options': {'A': 'yes', 'B': 'no'}} for i in range(count)]

    def test_every_notebook_obeys_cap_after_successful_batches(self):
        for notebook in NOTEBOOKS:
            for cap in [1, 3, 8]:
                with self.subTest(notebook=notebook.name, cap=cap):
                    infer = Mock(side_effect=lambda batch: ['A'] * len(batch))
                    result = self.notebook_function(notebook, cap, infer)(self.samples(20), '')
                    self.assertEqual(len(result), 20)
                    self.assertTrue(all(len(call.args[0]) <= cap for call in infer.call_args_list))

    def test_every_notebook_preserves_default_adaptive_sizes(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook.name):
                infer = Mock(side_effect=lambda batch: ['A'] * len(batch))
                self.notebook_function(notebook, None, infer)(self.samples(220), '')
                self.assertEqual([len(c.args[0]) for c in infer.call_args_list], [100, 104, 16])

    def test_every_notebook_retries_oom_with_smaller_batch(self):
        def infer(batch):
            if len(batch) > 1:
                raise MemoryError('mock OOM')
            return ['A']
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook.name):
                result = self.notebook_function(notebook, 3, infer)(self.samples(5), '')
                self.assertEqual(result, ['A'] * 5)

    def test_every_notebook_stops_if_batch_one_cannot_fit(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook.name):
                infer = Mock(side_effect=MemoryError('mock OOM'))
                with self.assertRaisesRegex(RuntimeError, 'smallest batch size'):
                    self.notebook_function(notebook, 1, infer)(self.samples(2), '')
                infer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
