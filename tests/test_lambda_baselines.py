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
                patch.object(launcher, 'check_git_access'), \
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

    def test_qwen_gemma_pair_uses_vm14k_direct_and_all_rows(self):
        args, access = self.invoke(['--task', 'eval', '--baselines', 'qwen3-4b', 'gemma4-e2b'])
        self.assertEqual(args['env_vars']['EVAL_MEDICAL_MODELS'], 'qwen3-4b gemma4-e2b')
        self.assertEqual(args['env_vars']['EVAL_DATASETS'], 'vm14k')
        self.assertEqual(args['env_vars']['EVAL_GENERATION_MODE'], 'direct')
        self.assertEqual(args['env_vars']['EVAL_LIMIT'], '0')
        self.assertEqual(args['env_vars']['EVAL_BATCH_SIZE'], '4')
        access.assert_called_once_with(['qwen3-4b', 'gemma4-e2b'], 'test')

    def test_region_is_forwarded_without_changing_gpu_or_image_defaults(self):
        args, _ = self.invoke(['--task', 'eval', '--baselines', 'medreason',
                               '--region', 'us-east-1', '--git-branch', 'thai-dev'])
        self.assertEqual(args['region'], 'us-east-1')
        self.assertEqual(args['git_branch'], 'thai-dev')
        self.assertEqual(args['instance_type'], launcher.DEFAULT_INSTANCE_TYPE)
        self.assertEqual(args['env_vars']['DOCKER_IMAGE'], '')

    def test_boot_timeout_is_forwarded(self):
        args, _ = self.invoke(['--task', 'eval', '--baselines', 'medreason', '--boot-timeout', '1200'])
        self.assertEqual(args['boot_timeout'], 1200)

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
            ['--task', 'eval', '--baselines', 'medreason', '--boot-timeout', '0'],
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
                                         'scripts/setup_baseline_env.sh',
                                         'requirements-baselines.txt',
                                         'EVAL/Unsloth_QA_goc_VM14K.ipynb',
                                         'EVAL/Data/eastern_medicine_VM14K.jsonl'})
                for name in names:
                    self.assertEqual(archive.extractfile(name).read(), (ROOT / name).read_bytes())
        with patch.object(launcher, 'scp_to', side_effect=inspect_bundle) as scp:
            launcher.copy_medical_eval_bundle('key', 'ip', ['vm14k'])
        scp.assert_called_once()


class GitPreflightTests(unittest.TestCase):
    def test_private_github_auth_stays_out_of_arguments(self):
        result = subprocess.CompletedProcess([], 0, stdout='abc\trefs/heads/thai-dev\n')
        with patch.object(launcher.subprocess, 'run', return_value=result) as run:
            launcher.check_git_access('https://github.com/example/private.git', 'thai-dev', 'secret-token')
        args, kwargs = run.call_args
        self.assertNotIn('secret-token', repr(args))
        self.assertEqual(kwargs['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertIn('refs/heads/thai-dev', args[0])

    def test_git_failure_stops_before_provisioning_without_echoing_secrets(self):
        env = {'LAMBDA_API_KEY': 'test', 'LAMBDA_SSH_KEY_NAME': 'test',
               'LAMBDA_SSH_PRIVATE_KEY': str(Path(__file__).resolve()), 'HF_TOKEN': 'test'}
        result = subprocess.CompletedProcess([], 128, stdout='', stderr='secret-token')
        errors = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.object(launcher, 'load_dotenv'), \
                patch.object(launcher, 'LambdaAPI'), patch.object(launcher, 'run_job') as run, \
                patch.object(launcher.subprocess, 'run', return_value=result), \
                patch.object(sys, 'argv', ['lambda_launch.py', '--task', 'eval', '--baselines', 'medreason']), \
                contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit):
                launcher.main()
        run.assert_not_called()
        self.assertIn('repository/branch is inaccessible', errors.getvalue())
        self.assertNotIn('secret-token', errors.getvalue())

    def test_missing_branch_is_rejected(self):
        result = subprocess.CompletedProcess([], 0, stdout='abc\trefs/heads/main\n')
        with patch.object(launcher.subprocess, 'run', return_value=result):
            with self.assertRaises(RuntimeError):
                launcher.check_git_access('https://github.com/example/repo.git', 'thai-dev', None)

    def test_no_output_directory_skips_archive_and_download(self):
        result = subprocess.CompletedProcess([], 1, stdout='')
        with patch.object(launcher, 'ssh_cmd', return_value=result) as ssh, \
                patch.object(launcher.subprocess, 'run') as scp, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(launcher.download_medical_eval('key', 'ip', 'job'))
        ssh.assert_called_once()
        scp.assert_not_called()


class BootWaitTests(unittest.TestCase):
    def test_longer_timeout_allows_active_after_ten_minutes(self):
        api = Mock()
        api.get_instance.side_effect = [
            {'status': 'booting', 'ip': '127.0.0.1'},
            {'status': 'active', 'ip': '127.0.0.1'},
        ]
        with patch.object(launcher.time, 'time', side_effect=[0, 0, 601]), \
                patch.object(launcher.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.wait_for_active(api, 'instance', timeout=1200)['status'], 'active')
        self.assertEqual(api.get_instance.call_count, 2)

    def test_ip_alone_does_not_bypass_boot_timeout(self):
        api = Mock()
        api.get_instance.return_value = {'status': 'booting', 'ip': '127.0.0.1'}
        with patch.object(launcher.time, 'time', side_effect=[0, 0, 600]), \
                patch.object(launcher.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.wait_for_active(api, 'instance')

    def test_timeout_still_terminates_before_setup(self):
        api = Mock()
        api.launch.return_value = 'instance'
        with patch.object(launcher, 'find_region_for', return_value='us-east-1'), \
                patch.object(launcher, 'wait_for_active', side_effect=SystemExit(1)) as wait, \
                patch.object(launcher, 'scp_to') as scp, \
                patch.object(launcher, 'download_medical_eval') as download, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.run_job(api, 'eval', [], launcher.DEFAULT_INSTANCE_TYPE,
                                 'key', 'keyfile', {}, 'repo', 'thai-dev', False,
                                 region='us-east-1', boot_timeout=1200)
        wait.assert_called_once_with(api, 'instance', timeout=1200)
        api.terminate.assert_called_once_with('instance')
        scp.assert_not_called()
        download.assert_not_called()


class RegionSelectionTests(unittest.TestCase):
    def api(self, regions):
        api = Mock()
        api.instance_types.return_value = {
            launcher.DEFAULT_INSTANCE_TYPE: {
                'regions_with_capacity_available': [{'name': name} for name in regions]
            }
        }
        return api

    def test_explicit_virginia_overrides_first_available(self):
        api = self.api(['asia-south-1', 'us-east-1'])
        self.assertEqual(launcher.find_region_for(api, launcher.DEFAULT_INSTANCE_TYPE, 'us-east-1'),
                         'us-east-1')

    def test_omitted_region_preserves_first_available_selection(self):
        api = self.api(['asia-south-1', 'us-east-1'])
        self.assertEqual(launcher.find_region_for(api, launcher.DEFAULT_INSTANCE_TYPE), 'asia-south-1')

    def test_unavailable_virginia_stops_before_launch_without_fallback(self):
        for regions in [['asia-south-1'], []]:
            api = self.api(regions)
            with self.subTest(regions=regions), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.run_job(api, 'eval', [], launcher.DEFAULT_INSTANCE_TYPE,
                                     'key', 'keyfile', {}, 'repo', 'thai-dev', False,
                                     region='us-east-1')
            api.launch.assert_not_called()


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
