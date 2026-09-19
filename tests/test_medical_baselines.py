import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_medical_baselines as evaluator
import lambda_launch as launcher


class MedicalEvaluationTests(unittest.TestCase):
    def test_existing_datasets_and_answer_rules(self):
        for dataset, count in [("new", 105), ("vm14k", 564)]:
            samples, protocol, metadata = evaluator.prepare_dataset(dataset)
            self.assertEqual(len(samples), count)
            self.assertEqual(metadata["generation_config"]["max_new_tokens"], 250)
            self.assertFalse(metadata["generation_config"]["do_sample"])
            for response, answer in [
                ("Final Answer: B", "B"), ("### C", "C"),
                ("", "EMPTY_RESPONSE"), ("Final Answer: unknown", "INVALID_FORMAT"),
            ]:
                self.assertEqual(protocol["extract_answer"](response), answer)
            sample = {**samples[0], "answer": "A", "options": {"A": "yes", "B": "no"}}
            rows = [evaluator.score_response(sample, {
                "response": text, "hit_token_limit": False,
            }, protocol, i) for i, text in enumerate(["Final Answer: A", "Final Answer: B", ""], 1)]
            summary = evaluator.summarize(rows)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["deviations"], 1)
            self.assertAlmostEqual(summary["accuracy"], 1 / 3)

    def test_access_resolves_redirect_commit_and_rejects_gated_model(self):
        response = SimpleNamespace(ok=True, headers={}, history=[
            SimpleNamespace(headers={"X-Repo-Commit": "abc123"}),
        ])
        with patch("requests.get", return_value=response):
            self.assertEqual(evaluator.check_access(["huatuo"], "test"), {"huatuo": "abc123"})
        with patch("requests.get", return_value=SimpleNamespace(ok=False, status_code=403)):
            with self.assertRaisesRegex(RuntimeError, "Cannot access google/medgemma"):
                evaluator.check_access(["medgemma"], "test")

    def test_mock_generation_oom_retries_and_writes_complete_outputs(self):
        class OutOfMemoryError(Exception):
            pass

        cuda = SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=Mock())
        batch_sizes = []

        def generate(model, tokenizer, samples, protocol):
            batch_sizes.append(len(samples))
            if len(samples) > 1:
                raise OutOfMemoryError()
            return [{"response": f"Final Answer: {samples[0]['answer']}", "hit_token_limit": False}]

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "torch": SimpleNamespace(cuda=cuda),
            "transformers": SimpleNamespace(set_seed=Mock()),
        }), patch.object(evaluator, "load_model", return_value=(Mock(), Mock())), \
                patch.object(evaluator, "generate_batch", side_effect=generate):
            args = SimpleNamespace(output_dir=Path(directory), limit=3, batch_size=4)
            evaluator.evaluate("huatuo", "new", "test-revision", args)
            output = Path(directory) / "huatuo/new"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(batch_sizes, [3, 2, 1, 1, 1])
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 3)
            with self.assertRaises(FileExistsError):
                evaluator.evaluate("huatuo", "new", "test-revision", args)

    def test_generation_error_is_not_scored_as_empty_response(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                OutOfMemoryError=MemoryError, empty_cache=Mock())),
            "transformers": SimpleNamespace(set_seed=Mock()),
        }), patch.object(evaluator, "load_model", return_value=(Mock(), Mock())), \
                patch.object(evaluator, "generate_batch", side_effect=ValueError("bad template")):
            args = SimpleNamespace(output_dir=Path(directory), limit=1, batch_size=1)
            with self.assertRaisesRegex(ValueError, "bad template"):
                evaluator.evaluate("huatuo", "new", "test-revision", args)
            output = Path(directory) / "huatuo/new"
            self.assertEqual(json.loads((output / "run_config.json").read_text())["status"], "failed")
            self.assertFalse((output / "summary.json").exists())


class LambdaMedicalTests(unittest.TestCase):
    def test_bootstrap_runs_all_pairs_and_reports_upload_failure(self):
        source = (evaluator.ROOT / "scripts/lambda_bootstrap.sh").read_text()
        function = source.split("run_medical_eval() {", 1)[1].split("\nrun_eval() {", 1)[0]
        stub = '''
EVAL_MEDICAL_MODELS='medgemma huatuo'
JOB_TAG=test-job
HF_SFT_REPO=test/results
in_docker() {
    printf '%s\\n' "$1"
    case "$1" in *lambda_push_to_hf.py*) return 1;; esac
    return 0
}
'''
        result = subprocess.run(
            ["bash"], input=stub + "run_medical_eval() {" + function + "\nrun_medical_eval\n",
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 1)
        for model in ["medgemma", "huatuo"]:
            for dataset in ["new", "vm14k"]:
                self.assertIn(f"--model '{model}' --dataset '{dataset}'", result.stdout)
        self.assertIn("--revision 'eval-medical-test-job'", result.stdout)

    def run_mock_job(self, backup_error=None, sentinel="DONE_SUCCESS"):
        api = Mock()
        api.launch.return_value = "test-instance"

        def ssh(key, ip, command, **kwargs):
            if command.startswith("ls "):
                return SimpleNamespace(stdout=sentinel)
            return SimpleNamespace(stdout="0")

        events = Mock()
        events.attach_mock(api.terminate, "terminate")
        with patch.object(launcher, "find_region_for", return_value="test-region"), \
                patch.object(launcher, "wait_for_active", return_value={"ip": "127.0.0.1"}), \
                patch.object(launcher, "wait_for_ssh"), \
                patch.object(launcher, "scp_to"), \
                patch.object(launcher, "copy_medical_eval_bundle"), \
                patch.object(launcher, "ssh_cmd", side_effect=ssh), \
                patch.object(launcher, "download_medical_eval", side_effect=backup_error) as backup:
            events.attach_mock(backup, "backup")
            code = launcher.run_job(
                api, "eval", [], "test-gpu", "test-key", "test-key-path",
                {"EVAL_MEDICAL_MODELS": "huatuo"}, "test-repo", "main2", False,
            )
        return code, api, events

    def test_backup_precedes_termination_even_after_upload_failure(self):
        code, api, events = self.run_mock_job(sentinel="DONE_FAILED")
        self.assertEqual(code, 2)
        api.terminate.assert_called_once_with("test-instance")
        self.assertEqual([call[0] for call in events.mock_calls], ["backup", "terminate"])

    def test_failed_backup_retains_instance(self):
        code, api, _ = self.run_mock_job(backup_error=RuntimeError("scp failed"))
        self.assertEqual(code, 2)
        api.terminate.assert_not_called()

    def test_preflight_denial_never_launches_gpu(self):
        args = ["lambda_launch.py", "--task", "eval", "--medical-baselines", "medgemma"]
        env = {"LAMBDA_API_KEY": "test", "LAMBDA_SSH_KEY_NAME": "test",
               "LAMBDA_SSH_PRIVATE_KEY": __file__, "HF_TOKEN": "test"}
        with patch.object(sys, "argv", args), patch.dict("os.environ", env), \
                patch.object(launcher, "load_dotenv"), \
                patch.object(evaluator, "check_access", side_effect=RuntimeError("HTTP 403")), \
                patch.object(launcher, "run_job") as run:
            with self.assertRaises(SystemExit):
                launcher.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
