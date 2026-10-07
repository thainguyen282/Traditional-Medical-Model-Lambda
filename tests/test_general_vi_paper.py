"""Paper-style Vietnamese evaluation checks that need no GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_general_vi as general
import lambda_launch


class PaperStyleTests(unittest.TestCase):
    def test_include_selection_is_shared_and_reproducible(self):
        rows = [{"question": str(i)} for i in range(550)]
        first, metadata = general.select_rows("include", rows, {"total_rows": 550}, "paper-style")
        second, other = general.select_rows("include", rows, {"total_rows": 550}, "paper-style")
        self.assertEqual(len(first), 55)
        self.assertEqual(first, second)
        self.assertEqual(metadata["source_indices"], other["source_indices"])
        self.assertEqual(len(set(metadata["source_indices"])), 55)
        self.assertEqual(general.select_rows("include", rows, {}, "legacy")[0], rows)

    def test_published_non_thinking_decoding_settings(self):
        config = general.decoding_config("mt-aime24", "paper-style")
        self.assertEqual(config["max_new_tokens"], 32768)
        self.assertEqual((config["temperature"], config["top_p"], config["top_k"]),
                         (0.7, 0.8, 20))
        self.assertEqual(config["presence_penalty"], 1.5)
        self.assertFalse(config["thinking_enabled"])

    def test_paper_comparison_rejects_different_include_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for benchmark in general.BENCHMARKS:
                for model in general.PAPER_STYLE_MODELS:
                    folder = output / model / benchmark
                    folder.mkdir(parents=True)
                    selected = [0] if model != "qwen3-4b-cpt" or benchmark != "include" else [1]
                    config = {"status": "complete", "limit": 0, "selected_rows": 1,
                              "protocol": "paper-style", "dataset_revision": "rev",
                              "source_files": {"data": "hash"}, "source_indices": selected,
                              "scorer_revision": "scorer", "decoding": {"do_sample": True},
                              "selection": "test"}
                    summary = {"status": "complete", "evaluated_rows": 1,
                               "correct": 1, "accuracy": 1.0}
                    (folder / "run_config.json").write_text(json.dumps(config))
                    (folder / "summary.json").write_text(json.dumps(summary))
                    (folder / "responses.jsonl").write_text(
                        json.dumps({"index": 0, "is_correct": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomparable"):
                general.export_comparison(output, "paper-style")

    def test_lambda_forwards_paper_profile_without_provisioning(self):
        env = {"LAMBDA_API_KEY": "test", "LAMBDA_SSH_KEY_NAME": "key",
               "LAMBDA_SSH_PRIVATE_KEY": __file__, "HF_TOKEN": "test"}
        argv = ["lambda_launch.py", "--task", "eval", "--general-vi",
                "--eval-protocol", "paper-style"]
        with patch.dict(os.environ, env, clear=True), patch.object(lambda_launch, "load_dotenv"), \
             patch.object(lambda_launch, "LambdaAPI"), patch.object(lambda_launch, "check_git_access"), \
             patch.object(general, "source_rows", return_value=([], {})), \
             patch.object(general, "resolve_models", return_value={}) as resolve, \
             patch.object(lambda_launch, "run_job", return_value=0) as run, \
             patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lambda_launch.main(), 0)
        resolve.assert_called_once_with(list(general.PAPER_STYLE_MODELS), "test")
        values = run.call_args.kwargs["env_vars"]
        self.assertEqual(values["EVAL_PROTOCOL"], "paper-style")
        self.assertEqual(values["EVAL_MEDICAL_MODELS"], " ".join(general.PAPER_STYLE_MODELS))


if __name__ == "__main__":
    unittest.main()
