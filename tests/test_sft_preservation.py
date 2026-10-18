"""Exercise launcher cleanup without provisioning a cloud instance."""
import ast
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class SFTPreservationTest(unittest.TestCase):
    def run_job(self, *, backup_error=None, interrupted=False):
        tree = ast.parse((ROOT / 'scripts/lambda_launch.py').read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_job')
        api = Mock()
        api.launch.return_value = 'instance-123'
        ssh = Mock(return_value=types.SimpleNamespace(stdout='DONE_FAILED'))
        if interrupted:
            ssh.side_effect = KeyboardInterrupt
        backup = Mock(side_effect=backup_error)
        env = dict(
            LambdaAPI=object, Path=Path, __file__=str(ROOT / 'scripts/lambda_launch.py'),
            find_region_for=lambda *args: 'region', log=Mock(),
            time=types.SimpleNamespace(time=lambda: 123),
            wait_for_active=lambda *args, **kwargs: {'ip': '127.0.0.1'},
            wait_for_ssh=Mock(), scp_to=Mock(), ssh_cmd=ssh,
            tempfile=tempfile, tarfile=__import__('tarfile'),
            render_env_file=lambda values: '', os=__import__('os'),
            REMOTE_REPO_DIR='repo', sys=types.SimpleNamespace(stdout=Mock()),
            download_sft_backup=backup,
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<launcher>', 'exec'), env)
        result = env['run_job'](api, 'sft', ['config.yaml'], 'gpu', 'key', 'keypath', {}, 'repo', 'main', False)
        return api, backup, result

    def test_failed_job_downloads_before_termination(self):
        api, backup, result = self.run_job()
        backup.assert_called_once_with('keypath', '127.0.0.1', 'tm-sft-123')
        api.terminate.assert_called_once_with('instance-123')
        self.assertEqual(result, 2)

    def test_failed_backup_preserves_instance(self):
        api, backup, result = self.run_job(backup_error=RuntimeError('network unavailable'))
        backup.assert_called_once()
        api.terminate.assert_not_called()
        self.assertEqual(result, 2)

    def test_interrupt_preserves_running_job(self):
        api, backup, result = self.run_job(interrupted=True)
        api.terminate.assert_not_called()
        backup.assert_not_called()
        self.assertNotEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
