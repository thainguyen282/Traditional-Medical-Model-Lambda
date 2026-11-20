import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sweeper', Path(__file__).resolve().parents[1] / 'scripts/lambda_expiry_sweeper.py')
sweeper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweeper)


class ExpirySweepTest(unittest.TestCase):
    def test_only_expired_matching_instances_are_terminated(self):
        launched = 1700000000
        deadline = launched + sweeper.LIFETIME
        instances = [
            {'id': 'expired', 'name': f'tmm-exp-{deadline}-{launched}', 'status': 'unhealthy'},
            {'id': 'future', 'name': f'tmm-exp-{deadline + 1}-{launched + 1}', 'status': 'active'},
            {'id': 'other', 'name': 'other', 'status': 'active'},
            {'id': 'bad', 'name': f'tmm-exp-{deadline}-{launched + 1}', 'status': 'active'},
            {'id': 'gone', 'name': f'tmm-exp-{deadline}-{launched}', 'status': 'terminating'},
        ]
        response = {'terminated_instances': [{'id': 'expired'}]}
        with patch.object(sweeper, 'request', side_effect=[instances, response]) as request, patch.object(sweeper.time, 'sleep'):
            sweeper.sweep('key', deadline)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args[2], {'instance_ids': ['expired']})

    def test_unconfirmed_termination_fails_run_and_continues(self):
        name = f'tmm-exp-{1700000000 + sweeper.LIFETIME}-1700000000'
        instances = [{'id': i, 'name': name, 'status': 'active'} for i in ('first', 'second')]
        with patch.object(sweeper, 'request', side_effect=[instances, {}, {'terminated_instances': [{'id': 'second'}]}]) as request, patch.object(sweeper.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                sweeper.sweep('key', 1800000000)
        self.assertEqual(request.call_count, 3)


if __name__ == '__main__':
    unittest.main()
