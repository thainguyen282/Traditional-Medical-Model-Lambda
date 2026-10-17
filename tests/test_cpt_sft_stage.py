"""Check CPT adapter continuation without importing the GPU stack."""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/unsloth-qwen3-4B/exp0015/config_sft.yaml'


class CPTSFTStageTest(unittest.TestCase):
    def test_matches_exp0013_sft_recipe(self):
        config = yaml.safe_load(CONFIG.read_text())
        previous = yaml.safe_load((CONFIG.parent.parent / 'exp0013/config_cpt.yaml').read_text())
        self.assertEqual(config['datasets'], previous['datasets'])
        for key, value in previous['training'].items():
            if key != 'output_dir':
                self.assertEqual(config['training'][key], value, key)
        self.assertTrue(config['model']['require_adapter'])
        self.assertNotIn('lora', config)

    def load(self, adapter):
        tree = ast.parse((ROOT / 'src/commands/train_command.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load_training_model')
        model = types.SimpleNamespace(peft_config={'default': {}}) if adapter else object()
        tokenizer = object()
        fast = Mock()
        fast.from_pretrained.return_value = (model, tokenizer)
        env = {'FastLanguageModel': fast, 'get_chat_template': lambda tokenizer, **kwargs: tokenizer}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<loader>', 'exec'), env)
        config = yaml.safe_load(CONFIG.read_text())
        config['model']['revision'] = 'pinned-cpt-revision'
        command = types.SimpleNamespace(config=config, logger=Mock(), lora_config=None)
        return env['_load_training_model'], command, fast, model, tokenizer

    def test_continues_adapter_and_forwards_revision(self):
        load, command, fast, model, tokenizer = self.load(True)
        self.assertEqual(load(command), (model, tokenizer))
        fast.for_training.assert_called_once_with(model)
        fast.get_peft_model.assert_not_called()
        self.assertEqual(fast.from_pretrained.call_args.kwargs['revision'], 'pinned-cpt-revision')

    def test_rejects_base_model_for_cpt_sft(self):
        load, command, fast, _, _ = self.load(False)
        with self.assertRaisesRegex(ValueError, 'Expected a CPT LoRA adapter'):
            load(command)
        fast.get_peft_model.assert_not_called()

    def test_eval_keeps_original_unconstrained_budget(self):
        for path in (ROOT / 'EVAL').glob('Unsloth_QA_*.ipynb'):
            cells = [''.join(c.get('source', [])) for c in json.loads(path.read_text())['cells']]
            config = next(s for s in cells if '# System Prompt and Generation Config' in s)
            env = {}
            exec(config, env)
            self.assertEqual(env['generation_config']['max_new_tokens'], 250)
            self.assertIn('Final Answer: X', env['system_prompt'])
            self.assertNotIn('prefix_allowed_tokens_fn', '\n'.join(cells))


if __name__ == '__main__':
    unittest.main()
