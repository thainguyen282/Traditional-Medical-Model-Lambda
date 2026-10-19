"""Verify SSH retry behavior without connecting to a cloud instance."""
import ast
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import Mock


class SSHRetryTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/lambda_launch.py'
        method = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'ssh_cmd')
        self.run = Mock()
        self.sleep = Mock()
        env = dict(subprocess=types.SimpleNamespace(run=self.run, CompletedProcess=subprocess.CompletedProcess),
                   time=types.SimpleNamespace(sleep=self.sleep), log=Mock(), REMOTE_USER='ubuntu')
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<launcher>', 'exec'), env)
        self.ssh = env['ssh_cmd']

    def results(self, *codes):
        self.run.side_effect = [subprocess.CompletedProcess(['ssh'], code) for code in codes]

    def test_transport_failure_recovers(self):
        self.results(255, 0)
        self.assertEqual(self.ssh('key', 'host', 'chmod 600 job.env', retries=4).returncode, 0)
        self.sleep.assert_called_once_with(10)

    def test_remote_failure_is_not_retried(self):
        self.results(1)
        with self.assertRaises(subprocess.CalledProcessError):
            self.ssh('key', 'host', 'false', retries=4)
        self.assertEqual(self.run.call_count, 1)
        self.sleep.assert_not_called()

    def test_exhausted_retries_raise(self):
        self.results(255, 255)
        with self.assertRaises(subprocess.CalledProcessError):
            self.ssh('key', 'host', 'true', retries=2)
        self.assertEqual(self.run.call_count, 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_unchecked_failure_returns_result(self):
        self.results(255, 255)
        self.assertEqual(self.ssh('key', 'host', 'true', retries=2, check=False).returncode, 255)

    def test_default_does_not_repeat_commands(self):
        self.results(255)
        with self.assertRaises(subprocess.CalledProcessError):
            self.ssh('key', 'host', 'start-job')
        self.assertEqual(self.run.call_count, 1)
        self.sleep.assert_not_called()
