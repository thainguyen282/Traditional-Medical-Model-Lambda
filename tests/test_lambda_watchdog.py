"""Verify watchdog deadlines and API retries without cloud access."""
import ast
import shlex
import subprocess
import sys
import types
from unittest.mock import Mock
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lambda_watchdog', ROOT / 'scripts/lambda_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class EndLoop(BaseException):
    pass


class WatchdogTest(unittest.TestCase):
    def test_api_targets_only_own_instance(self):
        response = io.BytesIO(json.dumps({'data': {'terminated_instances': [{'id': 'own-id'}]}}).encode())
        with patch.object(watchdog.urllib.request, 'urlopen', return_value=response) as request:
            watchdog.terminate({'api_key': 'test-key', 'instance_id': 'own-id'})
        req = request.call_args.args[0]
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.full_url, watchdog.TERMINATE_URL)
        self.assertEqual(json.loads(req.data), {'instance_ids': ['own-id']})
        self.assertEqual(req.get_header('Authorization'), 'Basic dGVzdC1rZXk6')
        self.assertEqual(request.call_args.kwargs['timeout'], 30)

    def test_empty_api_result_does_not_confirm_termination(self):
        with patch.object(watchdog.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"data": {}}')):
            with self.assertRaises(RuntimeError):
                watchdog.terminate({'api_key': 'test-key', 'instance_id': 'own-id'})

    def test_waits_until_deadline_then_retries_network_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'deadline': 100}))
            with patch.object(watchdog.time, 'time', side_effect=[40, 100, 160]), \
                    patch.object(watchdog.time, 'sleep', side_effect=[None, None, EndLoop]) as sleep, \
                    patch.object(watchdog, 'terminate', side_effect=[OSError('unreachable'), None]) as terminate:
                with self.assertRaises(EndLoop):
                    watchdog.main(config)
            self.assertEqual(terminate.call_count, 2)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [60, 60, 60])

    def test_reboot_after_deadline_terminates_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'deadline': 100}))
            with patch.object(watchdog.time, 'time', return_value=200), \
                    patch.object(watchdog.time, 'sleep', side_effect=EndLoop), \
                    patch.object(watchdog, 'terminate') as terminate:
                with self.assertRaises(EndLoop):
                    watchdog.main(config)
                terminate.assert_called_once()


class WatchdogInstallTest(unittest.TestCase):
    def test_credentials_use_stdin_and_root_only_permissions(self):
        tree = ast.parse((ROOT / 'scripts/lambda_launch.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'install_termination_watchdog')
        ssh = Mock()
        env = dict(LambdaAPI=object, Path=Path, json=json, shlex=shlex,
                   __file__=str(ROOT / 'scripts/lambda_launch.py'), ssh_cmd=ssh, log=Mock())
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<launcher>', 'exec'), env)
        api = types.SimpleNamespace(session=types.SimpleNamespace(auth=('test-secret', '')))
        env['install_termination_watchdog'](api, 'ssh-key', 'host', 'own-id', 43200)
        command = ssh.call_args.args[2]
        payload = ssh.call_args.kwargs['input_text']
        self.assertNotIn('test-secret', command)
        installer = shlex.split(command)[4]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'systemd').mkdir()
            installer = installer.replace('/etc/lambda-watchdog', str(root / 'watchdog'))
            installer = installer.replace('/etc/systemd/system', str(root / 'systemd'))
            subprocess.run([sys.executable, '-c', installer], input=payload, text=True, check=True)
            config = root / 'watchdog/config.json'
            self.assertEqual(json.loads(config.read_text()),
                             dict(api_key='test-secret', instance_id='own-id', deadline=43200))
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(config.parent.stat().st_mode & 0o777, 0o700)
            service = (root / 'systemd/lambda-watchdog.service').read_text()
            self.assertIn('Restart=always', service)
            self.assertIn('WantedBy=multi-user.target', service)
            self.assertNotIn('test-secret', service)
            self.assertEqual((root / 'watchdog/watchdog.py').read_text(),
                             (ROOT / 'scripts/lambda_watchdog.py').read_text())
