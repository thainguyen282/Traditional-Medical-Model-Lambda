"""CPU-only contracts and Lambda wiring for the isolated Qwen3.5 experiments."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'CPT'),str(ROOT/'scripts')]
import qwen35_common as common
import lambda_launch as launch


def fixture_config():
    config=yaml.safe_load((ROOT/'configs/experiments/qwen3.5-4B/mixed.yaml').read_text())
    config['dataset']['revision']='a'*40
    return config


class Qwen35Tests(unittest.TestCase):
    def test_frozen_vm14k_schema_and_final_answer_scoring(self):
        from eval_medical_baselines import prepare_dataset
        from eval_qwen35_vm14k import score_final
        rows,protocol,metadata=prepare_dataset('vm14k',generation_mode='notebook')
        self.assertEqual(len(rows),564)
        row=rows[0]
        self.assertNotIn('id',row)
        correct=lambda text:row['answer']
        result=score_final(row,'reasoning</think>final answer',False,correct,0)
        self.assertTrue(result['is_correct'])
        self.assertEqual(len(result['id']),64)
        for raw,truncated,extract in [('reasoning</think>final answer',True,correct),
                                     ('unfinished reasoning',False,correct),
                                     ('reasoning</think>invalid',False,lambda text:'Z')]:
            result=score_final(row,raw,truncated,extract,0)
            self.assertFalse(result['is_correct'])
            self.assertTrue(result['invalid'])

    def test_packing_keeps_tail_with_valid_causal_context(self):
        examples=list(common.packed_examples([{'input_ids':[1,2,3,4,5]}],4))
        self.assertEqual(examples[0],{'input_ids':[1,2,3,4],'labels':[1,2,3,4]})
        self.assertEqual(examples[1],{'input_ids':[4,5],'labels':[-100,5]})
        self.assertEqual(list(common.packed_examples([{'input_ids':[1,2,3]}],4))[0]['input_ids'],[1,2,3])

    def test_adapter_targets_cover_hybrid_layers_and_exclude_vision(self):
        for name in ['model.language_model.layers.0.linear_attn.in_proj_qkv',
                     'model.language_model.layers.0.linear_attn.out_proj',
                     'model.language_model.layers.3.self_attn.q_proj',
                     'model.language_model.layers.0.mlp.down_proj']:
            self.assertIsNotNone(re.fullmatch(common.LORA_PATTERN,name))
        for name in ['model.visual.blocks.0.q_proj','lm_head','model.language_model.embed_tokens']:
            self.assertIsNone(re.fullmatch(common.LORA_PATTERN,name))

    def test_config_rejects_moving_revisions_and_wrong_tokenizer_modes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'configs') as directory:
            path=Path(directory)/'config.yaml';config=fixture_config()
            path.write_text(yaml.safe_dump(config));self.assertEqual(common.load_config(path)['model']['name'],common.MODEL)
            config['dataset']['revision']='main';path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'immutable'):
                common.load_config(path)
            config=fixture_config();config['dataset']['preserve_whitespace']=False;path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'whitespace'):
                common.load_config(path)

    def test_old_prepared_tokenizer_is_rejected_before_training(self):
        config=fixture_config()
        manifest={'config':{'tokenizer':{'name':'Qwen/Qwen3-4B-Thinking-2507','revision':'a'*40}}}
        api=MagicMock();api.dataset_info.return_value.sha='a'*40
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(manifest))
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',return_value=str(path)):
                with self.assertRaisesRegex(ValueError,'different model/tokenizer'):
                    common.check_dataset(config,'dummy-token')

    def test_bundle_has_exact_local_code_no_credentials(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'configs') as directory:
            path=Path(directory)/'config.yaml';path.write_text(yaml.safe_dump(fixture_config()))
            archive=Path(directory)/'bundle.tar.gz'
            info=launch.build_qwen35_bundle(str(path),archive)
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),info['bundle_sha256'])
            with tarfile.open(archive) as saved:
                self.assertIn('CPT/train_qwen35.py',saved.getnames())
                self.assertIn('CPT/eval_qwen35_vm14k.py',saved.getnames())
                self.assertNotIn('.env',saved.getnames())
                metadata=json.load(saved.extractfile('CPT/qwen35_source.json'))
                for name,checksum in metadata['files'].items():
                    self.assertEqual(hashlib.sha256(saved.extractfile(name).read()).hexdigest(),checksum)

    def test_eval_failure_is_fatal_and_triggers_diagnostics(self):
        source=(ROOT/'scripts/lambda_bootstrap.sh').read_text()
        function=source[source.index('run_qwen35_cpt() {'):source.index('\nrun_cpt() {')]
        with tempfile.TemporaryDirectory() as directory:
            calls=Path(directory)/'calls'
            shell=f'''REPO_DIR={directory}
QWEN35_NAME=qwen35-test
JOB_TAG=test
CONFIG_LIST=configs/test.yaml
in_docker() {{ printf '%s\\n' "$1" >> '{calls}'; case "$1" in *eval_qwen35_vm14k.py*--publish*) return 7;; esac; return 0; }}
{function}
run_qwen35_cpt
'''
            result=subprocess.run(['bash','-c',shell],capture_output=True,text=True)
            self.assertEqual(result.returncode,7,result.stderr)
            commands=calls.read_text()
            self.assertIn('train_qwen35.py --config configs/test.yaml',commands)
            self.assertIn('--diagnostics',commands)
            self.assertNotIn('Unsloth_QA_CPT_new',commands)

    def test_legacy_cpt_entrypoint_has_not_changed(self):
        original=subprocess.check_output(['git','show','e5aad10:CPT/CPT.py'],cwd=ROOT)
        self.assertEqual((ROOT/'CPT/CPT.py').read_bytes(),original)


if __name__=='__main__':
    unittest.main()
