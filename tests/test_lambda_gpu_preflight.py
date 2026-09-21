"""Exercise the bootstrap's CUDA probe without provisioning a GPU instance."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


BOOTSTRAP = Path(__file__).resolve().parents[1] / 'scripts/lambda_bootstrap.sh'
PROBE = BOOTSTRAP.read_text().split("' <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]


class GPUPreflightTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Mock()
        self.cuda.device_count.return_value = 2
        self.cuda.get_device_name.side_effect = lambda i: f'GPU {i}'
        value = Mock()
        value.item.return_value = 2
        tensor = Mock()
        tensor.__add__ = Mock(return_value=value)
        self.torch = SimpleNamespace(
            __version__='test+cu128', version=SimpleNamespace(cuda='12.8'),
            cuda=self.cuda, ones=Mock(return_value=tensor),
        )
        self.output = io.StringIO()

    def run_probe(self):
        with patch.dict(sys.modules, torch=self.torch):
            with contextlib.redirect_stdout(self.output):
                exec(compile(PROBE, str(BOOTSTRAP), 'exec'), {})

    def test_success_exercises_every_visible_gpu(self):
        self.run_probe()
        self.cuda.init.assert_called_once_with()
        self.assertEqual(self.torch.ones.call_count, 2)
        for index in range(2):
            self.torch.ones.assert_any_call(1, device=f'cuda:{index}')
            self.cuda.synchronize.assert_any_call(index)
        self.assertIn('CUDA compute preflight passed', self.output.getvalue())

    def test_cpu_wheel_fails_before_cuda_init(self):
        self.torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
            self.run_probe()
        self.cuda.init.assert_not_called()

    def test_driver_error_is_preserved(self):
        self.cuda.init.side_effect = RuntimeError('CUDA driver initialization failed')
        with self.assertRaisesRegex(RuntimeError, 'CUDA driver initialization failed'):
            self.run_probe()
        self.torch.ones.assert_not_called()

    def test_compute_failure_is_not_treated_as_success(self):
        self.cuda.synchronize.side_effect = RuntimeError('CUDA kernel failed')
        with self.assertRaisesRegex(RuntimeError, 'CUDA kernel failed'):
            self.run_probe()
        self.assertNotIn('preflight passed', self.output.getvalue())

    def test_no_visible_devices_fails(self):
        self.cuda.device_count.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'no devices'):
            self.run_probe()


if __name__ == '__main__':
    unittest.main()
