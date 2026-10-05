"""Protocol and reporting checks; no model weights, cloud, or GPU required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_medical_baselines as evaluator


class GenerationTests(unittest.TestCase):
    def test_both_models_use_no_thinking_and_the_same_constraint(self):
        import torch

        class Inputs(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        samples = [{"question": "Question", "options": {"A": "One", "B": "Two"}}]
        protocol = {"generation_mode": "direct", "system_prompt": "Answer directly",
                    "generation_config": {"max_new_tokens": 16, "do_sample": False}}
        for gemma in (False, True):
            with self.subTest(gemma=gemma):
                tokenizer = MagicMock()
                tokenizer.pad_token_id = 0
                tokenizer.encode.side_effect = lambda value, **kw: [10 if value == " A" else 11]
                tokenizer.decode.return_value = " A"
                processor = MagicMock() if gemma else None
                formatter = processor if gemma else tokenizer
                formatter.apply_chat_template.return_value = "<assistant>"
                formatter.return_value = Inputs(input_ids=torch.tensor([[1, 2, 3]]))
                model = MagicMock()
                model.generation_config.eos_token_id = [2]
                model.generate.return_value = torch.tensor([[1, 2, 3, 10, 2]])
                result = evaluator.generate_batch(model, tokenizer, samples, protocol, processor)
                self.assertEqual(result, [{"response": "Final Answer: A", "hit_token_limit": False}])
                self.assertFalse(formatter.apply_chat_template.call_args.kwargs["enable_thinking"])
                self.assertEqual(formatter.call_args.kwargs["text"], ["<assistant>Final Answer:"])
                if gemma:
                    tokenizer.apply_chat_template.assert_not_called()
                args = model.generate.call_args.kwargs
                self.assertFalse(args["do_sample"])
                allowed = args["prefix_allowed_tokens_fn"]
                self.assertEqual(allowed(0, torch.tensor([1, 2, 3])), [10, 11])
                self.assertEqual(allowed(0, torch.tensor([1, 2, 3, 10])), [2])
                self.assertEqual(allowed(0, torch.tensor([1, 2, 3, 10, 2])), [2])

    def test_gemma_uses_processor_and_conditional_generation_loader(self):
        names = ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig",
                 "Gemma3ForConditionalGeneration", "Gemma4ForConditionalGeneration", "AutoProcessor")
        fake = SimpleNamespace(**{name: MagicMock() for name in names})
        with patch.dict(sys.modules, {"transformers": fake}), patch("torch.cuda.is_available", return_value=True):
            model, tokenizer, processor = evaluator.load_model("gemma4-e2b", "revision", "token")
        fake.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIs(tokenizer, processor.tokenizer)
        self.assertIs(model, fake.Gemma4ForConditionalGeneration.from_pretrained.return_value)
        self.assertEqual(fake.AutoProcessor.from_pretrained.call_args.args, ("google/gemma-4-E2B-it",))
        self.assertEqual(fake.Gemma4ForConditionalGeneration.from_pretrained.call_args.kwargs["revision"], "revision")

    def test_empty_answer_is_invalid(self):
        samples, protocol, _ = evaluator.prepare_dataset("vm14k")
        row = evaluator.score_response(samples[0], {"response": "", "hit_token_limit": True}, protocol, 1)
        self.assertTrue(row["is_deviation"])
        self.assertFalse(row["is_correct"])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        samples, protocol, metadata = evaluator.prepare_dataset("vm14k")
        self.assertEqual(len(samples), 564)
        for key in evaluator.VM14K_BASELINES:
            directory = self.output / key / "vm14k"
            directory.mkdir(parents=True)
            manifest = {**metadata, "model": evaluator.MODELS[key], "revision": "test-revision",
                        "status": "complete", "thinking_enabled": False, "seed": 3407,
                        "quantization": "nf4-double-quant-bfloat16", "dependencies": {"transformers": "5.5.0"},
                        "chat_template_kwargs": {"enable_thinking": False}}
            rows = [evaluator.score_response(sample, {"response": "Final Answer: " + sample["answer"],
                    "hit_token_limit": False}, protocol, index) for index, sample in enumerate(samples, 1)]
            (directory / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (directory / "run_config.json").write_text(json.dumps(manifest))
            evaluator.write_metrics(directory, {**manifest, **evaluator.summarize(rows)})

    def test_exports_two_complete_rows(self):
        evaluator.export_vm14k_comparison(self.output)
        self.assertEqual(len((self.output / "comparison.csv").read_text().splitlines()), 3)
        self.assertIn("564/564", (self.output / "comparison.md").read_text())
        self.assertIn("constrained answer-only", (self.output / "comparison.md").read_text())

    def test_incomplete_or_smoke_runs_cannot_be_reported(self):
        path = self.output / "gemma4-e2b/vm14k/run_config.json"
        manifest = json.loads(path.read_text())
        for update in ({"status": "failed"}, {"limit": 4}, {"thinking_enabled": True}):
            with self.subTest(update=update):
                path.write_text(json.dumps({**manifest, **update}))
                with self.assertRaises(ValueError):
                    evaluator.export_vm14k_comparison(self.output)
                self.assertFalse((self.output / "comparison.csv").exists())

    def test_different_runtime_is_rejected(self):
        path = self.output / "gemma4-e2b/vm14k/run_config.json"
        manifest = json.loads(path.read_text())
        manifest["dependencies"] = {"transformers": "different-version"}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "different evaluation"):
            evaluator.export_vm14k_comparison(self.output)

    def test_missing_response_is_rejected(self):
        path = self.output / "gemma4-e2b/vm14k/responses.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "incomplete responses"):
            evaluator.export_vm14k_comparison(self.output)


class PairedWorkflowTests(unittest.TestCase):
    def run_pair(self, fail_smoke=False):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            def evaluate(key, dataset, revision, args):
                calls.append((key, dataset, args.limit, revision))
                dest = args.output_dir / key / dataset
                dest.mkdir(parents=True)
                summary = {"samples": args.limit or 564, "deviations": int(fail_smoke), "token_limit_rate": 0}
                (dest / "summary.json").write_text(json.dumps(summary))

            with patch.object(evaluator, "evaluate", side_effect=evaluate), \
                    patch.object(evaluator, "check_runtime"), \
                    patch.object(evaluator, "check_access", return_value=dict.fromkeys(evaluator.VM14K_BASELINES, "pinned")), \
                    patch.object(evaluator, "export_vm14k_comparison") as export, \
                    patch("sys.argv", ["eval", "--vm14k-baselines", "--output-dir", directory]), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = evaluator.main()
        return result, calls, export.call_count

    def test_both_smoke_runs_precede_full_runs_and_export(self):
        result, calls, exports = self.run_pair()
        self.assertEqual(result, 0)
        self.assertEqual([(c[0], c[2]) for c in calls],
                         [("qwen3-4b", 4), ("gemma4-e2b", 4), ("qwen3-4b", 0), ("gemma4-e2b", 0)])
        self.assertTrue(all(c[1] == "vm14k" and c[3] == "pinned" for c in calls))
        self.assertEqual(exports, 1)

    def test_failed_smoke_prevents_full_runs_and_export(self):
        result, calls, exports = self.run_pair(fail_smoke=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(exports, 0)


if __name__ == "__main__":
    unittest.main()
