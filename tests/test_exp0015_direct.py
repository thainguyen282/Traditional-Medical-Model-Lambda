"""CPU regression checks for CPT-SFT answer-only evaluation."""
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_medical_baselines as runner

class Exp0015DirectTests(TestCase):
    def test_thinking_prompt_is_closed_before_final_answer(self):
        rows, protocol, _ = runner.prepare_dataset('vm14k', 4, 'direct')
        formatter = Mock()
        formatter.apply_chat_template.return_value = '<|im_start|>assistant\n<think>\n'
        result = runner.format_generation_prompt(formatter, rows[0], protocol)
        self.assertTrue(result.endswith('<think>\n</think>\n\nFinal Answer:'))
        self.assertEqual(protocol['generation_config']['max_new_tokens'], 16)
        self.assertFalse(protocol['generation_config']['do_sample'])

    def test_only_option_and_eos_can_be_generated(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: [10 if label.strip() == 'A' else 11]
        constraint = runner.answer_constraint(tokenizer, [{'options': {'A': 'a', 'B': 'b'}}], 2, [99], 16)
        self.assertEqual(set(constraint(0, Mock(__getitem__=lambda self, key: Mock(tolist=lambda: [])))), {10, 11})
        self.assertEqual(constraint(0, Mock(__getitem__=lambda self, key: Mock(tolist=lambda: [10]))), [99])

    def test_adapter_access_is_pinned(self):
        with patch.object(runner, 'validate_exp0015') as validate:
            self.assertEqual(runner.check_access(['exp0015'], None), {'exp0015': runner.EXP0015_REVISION})
            validate.assert_called_once_with(None)

    def test_final_adapter_is_applied_once_on_pinned_nf4_base(self):
        import types
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), bfloat16='bf16')
        base_loader = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 1
        tokenizer_loader = Mock()
        tokenizer_loader.from_pretrained.return_value = tokenizer
        peft_loader = Mock()
        transformers = types.SimpleNamespace(AutoModelForCausalLM=base_loader,
            AutoTokenizer=tokenizer_loader, BitsAndBytesConfig=Mock(), Gemma3ForConditionalGeneration=Mock())
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers,
                                    'peft': types.SimpleNamespace(PeftModel=peft_loader)}), \
             patch.object(runner, 'validate_exp0015'):
            runner.load_model('exp0015', runner.EXP0015_REVISION, None)
        base_loader.from_pretrained.assert_called_once()
        self.assertEqual(base_loader.from_pretrained.call_args.args, (runner.EXP0015_BASE,))
        self.assertEqual(base_loader.from_pretrained.call_args.kwargs['revision'], runner.EXP0015_BASE_REVISION)
        self.assertTrue(transformers.BitsAndBytesConfig.call_args.kwargs['load_in_4bit'])
        peft_loader.from_pretrained.assert_called_once_with(base_loader.from_pretrained.return_value,
            runner.MODELS['exp0015'], revision=runner.EXP0015_REVISION, token=None, is_trainable=False)
