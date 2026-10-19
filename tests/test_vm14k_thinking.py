"""Validate frozen medical data and reasoning boundary scoring without a GPU."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_general_vi_vllm as runner

class Vm14kThinkingTests(unittest.TestCase):
    def test_frozen_data_and_original_prompt(self):
        rows, meta = runner.source_rows('vm14k', None)
        self.assertEqual(len(rows), 564)
        self.assertEqual(meta['dataset_revision'], runner.vm14k_protocol()[2]['dataset_sha256'])
        self.assertIn('Final Answer: X', runner.prompt('vm14k', rows[0]))
        self.assertFalse(meta['constrained_to_options'])

    def test_only_completed_final_answer_is_scored(self):
        raw = 'Final Answer: B</think>Final Answer: A'
        self.assertEqual(runner.final_answer('qwen3-4b-thinking-2507', raw, 'stop', 'vm14k'), 'A')
        self.assertIsNone(runner.final_answer('qwen3-4b-thinking-2507', raw, 'length', 'vm14k'))
        self.assertIsNone(runner.final_answer('qwen3-4b-thinking-2507', 'Final Answer: B', 'stop', 'vm14k'))
