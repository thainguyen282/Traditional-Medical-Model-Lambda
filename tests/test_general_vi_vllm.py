"""CPU-only regression checks for vLLM benchmark routing and scoring."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_general_vi_vllm as runner


class VllmGeneralTests(unittest.TestCase):
    def test_math_uses_final_box_after_reasoning(self):
        raw = r'reasoning \boxed{1}</think>Final answer: \boxed{42}'
        self.assertEqual(runner.final_answer('qwen3-4b-thinking-2507', raw, 'stop', 'polymath'), '42')
        self.assertIsNone(runner.final_answer('qwen3-4b-thinking-2507', raw, 'length', 'polymath'))
        self.assertIsNone(runner.final_answer('qwen3-4b-thinking-2507', r'\boxed{42}', 'stop', 'polymath'))

    def test_every_benchmark_routes_and_scores(self):
        for benchmark in runner.BENCHMARKS:
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                is_math = benchmark in ('mt-aime24', 'polymath')
                row = {'question': 'Question', 'options': None if is_math else ['a', 'b', 'c', 'd'],
                       'answer': '42' if is_math else 'B'}
                raw = '</think>' + (r'\boxed{42}' if is_math else 'Final Answer: B')
                tokenizer = Mock()
                tokenizer.decode.return_value = raw
                generated = types.SimpleNamespace(token_ids=[1], finish_reason='stop')
                engine = Mock()
                engine.chat.return_value = [types.SimpleNamespace(outputs=[generated])]
                engine.get_tokenizer.return_value = tokenizer
                api = Mock()
                api.model_info.return_value.sha = 'revision'
                hub = types.SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=Mock(), snapshot_download=Mock())
                vllm = types.SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=Mock(), __version__='test')
                args = types.SimpleNamespace(benchmark=benchmark, model='qwen3-4b-thinking-2507',
                    limit=0, output_dir=Path(directory), max_input_tokens=4096, max_new_tokens=8192,
                    max_num_seqs=8, gpu_memory_utilization=.7, enforce_eager=True, protocol='thinking')
                with patch.dict(sys.modules, {'huggingface_hub': hub, 'vllm': vllm}), \
                     patch.object(runner, 'source_rows', return_value=([row], {'total_rows': 1})) as source, \
                     patch.object(runner, 'get_math_scorer', return_value=lambda a, b: a == b) as scorer:
                    self.assertEqual(runner.run(args), 0)
                source.assert_called_once_with(benchmark, runner.os.getenv('HF_TOKEN'))
                self.assertEqual(scorer.called, is_math)
                summary = json.loads((Path(directory) / args.model / benchmark / 'summary.json').read_text())
                self.assertEqual(summary['correct'], 1)
                self.assertEqual(summary['selected_rows'], 1)
                self.assertEqual(summary['benchmark'], benchmark)


    def test_exp0015_uses_final_adapter_once_on_pinned_underlying_base(self):
        self._check_thinking_adapter('exp0015', runner.EXP0015_REVISION)

    def test_cpt_only_uses_pinned_adapter_with_baseline_thinking_settings(self):
        self._check_thinking_adapter('qwen3-4b-thinking-cpt', runner.EXP0015_CPT_REVISION)

    def _check_thinking_adapter(self, model, revision):
        with tempfile.TemporaryDirectory() as directory:
            adapter_config = Path(directory) / 'adapter_config.json'
            adapter_config.write_text(json.dumps({'base_model_name_or_path': runner.EXP0015_BASE,
                                                 'r': 64, 'use_rslora': True, 'lora_alpha': 64}))
            api = Mock()
            api.model_info.side_effect = [types.SimpleNamespace(sha=revision),
                                         types.SimpleNamespace(sha=runner.EXP0015_BASE_REVISION)]
            snapshot = Mock(return_value='/adapter')
            hub = types.SimpleNamespace(HfApi=Mock(return_value=api),
                                        hf_hub_download=Mock(return_value=str(adapter_config)),
                                        snapshot_download=snapshot)
            tokenizer = Mock()
            tokenizer.decode.return_value = '</think>Final Answer: B'
            engine = Mock()
            engine.get_tokenizer.return_value = tokenizer
            engine.chat.return_value = [types.SimpleNamespace(outputs=[
                types.SimpleNamespace(token_ids=[1], finish_reason='stop')])]
            vllm = types.SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=Mock(), __version__='test')
            lora = types.SimpleNamespace(LoRARequest=Mock(return_value='final-sft-adapter'))
            args = types.SimpleNamespace(model=model, benchmark='include', limit=0,
                output_dir=Path(directory), max_input_tokens=4096, max_new_tokens=32768,
                max_num_seqs=8, gpu_memory_utilization=.70, enforce_eager=True, protocol='thinking')
            row = {'question': 'Q', 'options': ['a', 'b', 'c', 'd'], 'answer': 'B'}
            with patch.dict(sys.modules, {'huggingface_hub': hub, 'vllm': vllm,
                                         'vllm.lora.request': lora}), \
                 patch.object(runner, 'source_rows', return_value=([row], {'total_rows': 1})):
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(vllm.LLM.call_args.kwargs['model'], runner.EXP0015_BASE)
            self.assertEqual(vllm.LLM.call_args.kwargs['revision'], runner.EXP0015_BASE_REVISION)
            self.assertEqual(vllm.LLM.call_args.kwargs['max_lora_rank'], 64)
            self.assertTrue(vllm.LLM.call_args.kwargs['enable_lora'])
            snapshot.assert_called_once()
            self.assertEqual(snapshot.call_args.kwargs['revision'], revision)
            lora.LoRARequest.assert_called_once_with(model, 1, '/adapter')
            self.assertEqual(engine.chat.call_args.kwargs['lora_request'], 'final-sft-adapter')
            self.assertEqual(engine.chat.call_args.kwargs['chat_template_kwargs'], {'enable_thinking': True})
            manifest = json.loads((Path(directory) / model / 'include/run_config.json').read_text())
            if model == 'exp0015':
                self.assertEqual(manifest['cpt_source_revision'], runner.EXP0015_CPT_REVISION)
            else:
                self.assertEqual(manifest['model_revision'], runner.EXP0015_CPT_REVISION)
                self.assertEqual(manifest['adapter_composition'], 'CPT-only adapter; apply once')
            self.assertEqual(manifest['decoding']['max_new_tokens'], 32768)
            self.assertEqual(manifest['base_model'], runner.EXP0015_BASE)


if __name__ == '__main__':
    unittest.main()
