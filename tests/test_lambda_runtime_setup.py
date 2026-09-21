"""Test host runtime setup with mocked commands, without touching Docker."""
import os
from pathlib import Path
import subprocess
import unittest


BOOTSTRAP = Path(__file__).resolve().parents[1] / 'scripts/lambda_bootstrap.sh'
SETUP = 'ensure_nvidia_runtime() {' + BOOTSTRAP.read_text().split(
    'ensure_nvidia_runtime() {', 1
)[1].split('\nensure_nvidia_runtime\n', 1)[0]
MOCKS = r'''
set -u -o pipefail
DOCKER=mock_docker
registered=${REGISTERED:-0}
mock_docker() {
    [ "${INFO_FAIL:-0}" = 0 ] || return 1
    echo 'runc io.containerd.runc.v2'
    [ "$registered" = 0 ] || echo 'nvidia'
    return 0
}
command() {
    if [ "$*" = '-v nvidia-ctk' ]; then
        [ "${MISSING_CTK:-0}" = 0 ]
    else
        builtin command "$@"
    fi
}
sudo() {
    echo "sudo $*"
    case "$*" in
        'nvidia-ctk runtime configure --runtime=docker')
            [ "${CONFIG_FAIL:-0}" = 0 ] ;;
        'systemctl restart docker')
            [ "${RESTART_FAIL:-0}" = 0 ] || return 1
            registered=${AFTER_RESTART:-1} ;;
        *) return 99 ;;
    esac
}
fail() { echo "FAIL: $*"; exit 1; }
'''
# Match Docker's space-separated template output in the mock.
MOCKS = MOCKS.replace("echo 'runc io.containerd.runc.v2'", "printf 'runc io.containerd.runc.v2 '")


class RuntimeSetupTests(unittest.TestCase):
    def run_setup(self, **settings):
        return subprocess.run(
            ['bash', '-c', MOCKS + SETUP + '\nensure_nvidia_runtime'],
            env={**os.environ, **settings}, text=True, capture_output=True,
        )

    def test_registered_runtime_is_left_alone(self):
        result = self.run_setup(REGISTERED='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('sudo ', result.stdout)

    def test_missing_runtime_is_configured_and_verified(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('sudo nvidia-ctk runtime configure --runtime=docker', result.stdout)
        self.assertIn('sudo systemctl restart docker', result.stdout)
        self.assertIn('NVIDIA runtime registered', result.stdout)

    def test_missing_toolkit_stops_before_mutation(self):
        result = self.run_setup(MISSING_CTK='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nvidia-ctk not found', result.stdout)
        self.assertNotIn('sudo ', result.stdout)

    def test_failed_configuration_does_not_restart_docker(self):
        result = self.run_setup(CONFIG_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('sudo systemctl', result.stdout)

    def test_restart_failure_is_fatal(self):
        result = self.run_setup(RESTART_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Docker restart', result.stdout)

    def test_registration_is_checked_after_restart(self):
        result = self.run_setup(AFTER_RESTART='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('still missing', result.stdout)

    def test_docker_info_failure_stops_before_mutation(self):
        result = self.run_setup(INFO_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('sudo ', result.stdout)


if __name__ == '__main__':
    unittest.main()
