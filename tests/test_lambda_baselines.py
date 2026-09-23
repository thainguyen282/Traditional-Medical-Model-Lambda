"""Baseline integration checks without a GPU, cloud instance, or network access."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_medical_baselines as evaluator
import lambda_launch as launcher


class BaselineCLITests(unittest.TestCase):
    def invoke(self, argv, access_error=None):
        env = {'LAMBDA_API_KEY': 'test', 'LAMBDA_SSH_KEY_NAME': 'test',
               'LAMBDA_SSH_PRIVATE_KEY': str(Path(__file__).resolve()), 'HF_TOKEN': 'test'}
        with patch.dict(os.environ, env, clear=True), patch.object(launcher, 'load_dotenv'), \
                patch.object(launcher, 'LambdaAPI'), \
                patch.object(launcher, 'run_job', return_value=0) as run, \
                patch.object(evaluator, 'check_access', side_effect=access_error) as access, \
                patch.object(sys, 'argv', ['lambda_launch.py', *argv]), \
                contextlib.redirect_stderr(io.StringIO()):
            if access_error:
                with self.assertRaises(SystemExit):
                    launcher.main()
                run.assert_not_called()
                return
            self.assertEqual(launcher.main(), 0)
            return run.call_args.kwargs, access

    def test_medreason_defaults_to_only_564_rows(self):
        args, access = self.invoke(['--task', 'eval', '--baselines', 'medreason'])
        self.assertEqual(args['instance_type'], launcher.DEFAULT_INSTANCE_TYPE)
        self.assertFalse(args['keep_alive'])
        self.assertEqual(args['env_vars']['EVAL_MEDICAL_MODELS'], 'medreason')
        self.assertEqual(args['env_vars']['EVAL_DATASETS'], 'vm14k')
        self.assertEqual(args['env_vars']['EVAL_LIMIT'], '0')
        self.assertEqual(args['env_vars']['EVAL_BATCH_SIZE'], '4')
        access.assert_called_once_with(['medreason'], 'test')
        rows, _, metadata = evaluator.prepare_dataset('vm14k')
        self.assertEqual(len(rows), 564)
        self.assertEqual(metadata['samples'], 564)

    def test_explicit_options_and_duplicates(self):
        args, access = self.invoke(['--task', 'eval', '--medical-baselines', 'medreason', 'medreason',
                                   '--eval-datasets', 'vm14k', 'vm14k', '--eval-limit', '4',
                                   '--eval-batch-size', '1', '--eval-generation-mode', 'notebook'])
        self.assertEqual(args['env_vars']['EVAL_DATASETS'], 'vm14k')
        self.assertEqual(args['env_vars']['EVAL_LIMIT'], '4')
        self.assertEqual(args['env_vars']['EVAL_BATCH_SIZE'], '1')
        self.assertEqual(args['env_vars']['EVAL_GENERATION_MODE'], 'notebook')
        access.assert_called_once_with(['medreason'], 'test')

    def test_existing_tasks_do_not_preflight_baselines_or_forward_new_environment(self):
        for argv in [['--task', 'cpt'], ['--task', 'sft', '--config', 'config.yaml'],
                     ['--task', 'inference'], ['--task', 'eval'],
                     ['--task', 'eval', '--eval-only', 'name|repo|rev|sft']]:
            with self.subTest(argv=argv):
                args, access = self.invoke(argv)
                access.assert_not_called()
                self.assertNotIn('EVAL_MEDICAL_MODELS', args['env_vars'])
                self.assertNotIn('EVAL_BATCH_SIZE', args['env_vars'])

    def test_inaccessible_model_fails_before_launch(self):
        self.invoke(['--task', 'eval', '--baselines', 'medreason'], RuntimeError('access denied'))

    def test_invalid_options_fail_before_api_initialization(self):
        cases = [
            ['--task', 'sft', '--baselines', 'medreason'],
            ['--task', 'eval', '--baselines', 'missing'],
            ['--task', 'eval', '--baselines', 'medreason', '--eval-datasets', 'missing'],
            ['--task', 'eval', '--baselines', 'medreason', '--eval-only', 'x'],
            ['--task', 'eval', '--baselines', 'medreason', '--config', 'x'],
            ['--task', 'eval', '--baselines', 'medreason', '--eval-limit', '-1'],
            ['--task', 'eval', '--baselines', 'medreason', '--eval-batch-size', '0'],
            ['--task', 'eval', '--eval-datasets', 'vm14k'],
        ]
        for argv in cases:
            with self.subTest(argv=argv), patch.object(launcher, 'load_dotenv'), \
                    patch.object(launcher, 'LambdaAPI') as api, \
                    patch.object(sys, 'argv', ['lambda_launch.py', *argv]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.main()
                api.assert_not_called()

    def test_bundle_contains_exact_selected_local_data(self):
        def inspect_bundle(key, ip, local, remote):
            with tarfile.open(local) as archive:
                names = set(archive.getnames())
                self.assertEqual(names, {'scripts/eval_medical_baselines.py',
                                         'EVAL/Unsloth_QA_goc_VM14K.ipynb',
                                         'EVAL/Data/eastern_medicine_VM14K.jsonl'})
                for name in names:
                    self.assertEqual(archive.extractfile(name).read(), (ROOT / name).read_bytes())
        with patch.object(launcher, 'scp_to', side_effect=inspect_bundle) as scp:
            launcher.copy_medical_eval_bundle('key', 'ip', ['vm14k'])
        scp.assert_called_once()


class LifecycleTests(unittest.TestCase):
    def run_mock_job(self, baseline=True, sentinel='DONE_SUCCESS', download_error=None, keep=False):
        events = []
        api = Mock()
        api.launch.return_value = 'instance-test'
        api.terminate.side_effect = lambda *args: events.append('terminate')
        def ssh(key, ip, command, **kwargs):
            if command.startswith('ls '):
                return subprocess.CompletedProcess([], 0, stdout=sentinel)
            return subprocess.CompletedProcess([], 0, stdout='0')
        def download(*args):
            events.append('download')
            if download_error:
                raise download_error
        env = {'EVAL_MEDICAL_MODELS': 'medreason', 'EVAL_DATASETS': 'vm14k'} if baseline else {}
        with patch.object(launcher, 'find_region_for', return_value='region-test'), \
                patch.object(launcher, 'wait_for_active', return_value={'ip': '127.0.0.1'}), \
                patch.object(launcher, 'wait_for_ssh'), patch.object(launcher, 'scp_to'), \
                patch.object(launcher, 'copy_medical_eval_bundle') as bundle, \
                patch.object(launcher, 'ssh_cmd', side_effect=ssh), \
                patch.object(launcher, 'download_medical_eval', side_effect=download), \
                contextlib.redirect_stdout(io.StringIO()):
            code = launcher.run_job(api, 'eval', [], launcher.DEFAULT_INSTANCE_TYPE,
                                    'key', 'keyfile', env, 'repo', 'main2', keep)
        api.launch.assert_called_once()
        self.assertEqual(bundle.call_count, int(baseline))
        return code, events

    def test_success_downloads_before_normal_termination(self):
        self.assertEqual(self.run_mock_job(), (0, ['download', 'terminate']))

    def test_remote_failure_still_downloads_and_terminates(self):
        self.assertEqual(self.run_mock_job(sentinel='DONE_FAILED'), (2, ['download', 'terminate']))

    def test_failed_or_interrupted_download_does_not_leave_billable_instance(self):
        for error in [RuntimeError('scp failed'), KeyboardInterrupt()]:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.run_mock_job(download_error=error), (2, ['download', 'terminate']))

    def test_keep_alive_still_requires_explicit_flag(self):
        self.assertEqual(self.run_mock_job(keep=True), (0, ['download']))

    def test_existing_eval_does_not_bundle_or_download(self):
        self.assertEqual(self.run_mock_job(baseline=False), (0, ['terminate']))


class BootstrapTests(unittest.TestCase):
    def test_baseline_dispatch_and_failure_propagation(self):
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        functions = '\n'.join(re.search(r'^' + name + r'\(\) \{.*?^\}', source, re.M | re.S).group()
                              for name in ['run_medical_eval', 'run_eval'])
        for failure in ['', 'eval_medical_baselines.py', 'lambda_push_to_hf.py']:
            with self.subTest(failure=failure):
                script = '''set -u
EVAL_MEDICAL_MODELS=medreason
EVAL_DATASETS=vm14k
JOB_TAG=test-job
HF_SFT_REPO=test/repo
in_docker() {
    printf '%s\\n' "$1"
    if [ -n "$MOCK_FAILURE" ] && [[ "$1" == *"$MOCK_FAILURE"* ]]; then return 1; fi
}
'''
                result = subprocess.run(['bash', '-c', script + functions + '\nrun_eval'],
                                        env={**os.environ, 'MOCK_FAILURE': failure},
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, int(bool(failure)), result.stderr)
                self.assertIn("--model 'medreason' --dataset 'vm14k'", result.stdout)
                self.assertIn("--limit '0' --batch-size '4'", result.stdout)
                self.assertIn("--revision 'eval-medical-test-job'", result.stdout)
                self.assertIn('lambda_push_to_hf.py', result.stdout)
                self.assertNotIn('nbconvert', result.stdout)


if __name__ == '__main__':
    unittest.main()
