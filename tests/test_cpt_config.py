"""Exercise config propagation without GPU allocation or Hub writes."""
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/unsloth-qwen3-4B/exp0015/config_cpt.yaml'


class TrainingReached(Exception):
    pass


class CPTConfigTest(unittest.TestCase):
    def run_cpt(self, configured):
        captured = {}

        class Dataset:
            column_names = ['text']
            def __len__(self):
                return 1
            def map(self, fn):
                return self
            def filter(self, fn):
                return self

        class Model:
            @staticmethod
            def from_pretrained(**kwargs):
                captured['model'] = kwargs
                return object(), object()
            @staticmethod
            def get_peft_model(model, **kwargs):
                captured['lora'] = kwargs
                return model

        class Trainer:
            def __init__(self, **kwargs):
                captured['trainer'] = kwargs
            def train(self):
                raise TrainingReached

        def dataset(name, **kwargs):
            captured.setdefault('datasets', []).append((name, kwargs))
            return Dataset()

        def module(**kwargs):
            return types.SimpleNamespace(**kwargs)

        stubs = {
            'dotenv': module(load_dotenv=lambda *a, **k: None),
            'huggingface_hub': module(get_token=lambda: 'test', HfApi=lambda **k: module(dataset_info=lambda _: None)),
            'torch': module(),
            'datasets': module(load_dataset=dataset),
            'unsloth': module(FastLanguageModel=Model, UnslothTrainer=Trainer,
                              UnslothTrainingArguments=lambda **kwargs: kwargs),
        }
        argv = ['CPT.py'] + (['--config', str(CONFIG)] if configured else [])
        with patch.dict(sys.modules, stubs), patch.object(sys, 'argv', argv):
            with self.assertRaises(TrainingReached):
                runpy.run_path(str(ROOT / 'CPT/CPT.py'), run_name='__main__')
        return captured

    def test_yaml_reaches_model_dataset_and_trainer(self):
        result = self.run_cpt(True)
        args = result['trainer']['args']
        self.assertEqual(result['model']['model_name'], 'Qwen/Qwen3-4B-Thinking-2507')
        self.assertEqual(result['datasets'][0][0], 'venera-ai/TradMed-CPT-53k-2048')
        self.assertEqual(args['num_train_epochs'], 1)
        self.assertEqual(args['per_device_train_batch_size'], 8)
        self.assertEqual(args['gradient_accumulation_steps'], 2)
        self.assertEqual(args['output_dir'], str(ROOT / 'CPT/cpt_qwen3_thinking_2507_medical'))
        self.assertTrue(result['lora']['use_gradient_checkpointing'])
        self.assertEqual(result['trainer']['dataset_kwargs'], {'add_special_tokens': False})

    def test_no_config_retains_existing_defaults(self):
        result = self.run_cpt(False)
        self.assertEqual(result['trainer']['args']['num_train_epochs'], 3)
        self.assertEqual(result['trainer']['args']['per_device_train_batch_size'], 2)
        self.assertNotIn('dataset_kwargs', result['trainer'])

    def test_bootstrap_passes_config_as_one_argument(self):
        import subprocess
        source = (ROOT / 'scripts/lambda_bootstrap.sh').read_text()
        function = source[source.index('run_cpt() {'):source.index('\nrun_sft() {')]
        script = function + '\nin_docker() { printf "%s\\n" "$1"; return 17; }\nrun_cpt\n'
        result = subprocess.run(['bash', '-c', script], env={**os.environ, 'CPT_NEW_MODEL': '',
                                'CONFIG_LIST': 'configs/a config.yaml'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 17)
        import shlex
        command = result.stdout.splitlines()[-1]
        self.assertEqual(shlex.split(command)[-2:], ['--config', 'configs/a config.yaml'])


if __name__ == '__main__':
    unittest.main()
