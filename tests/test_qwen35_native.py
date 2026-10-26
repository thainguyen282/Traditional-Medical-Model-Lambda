"""Optional real Transformers/PEFT smoke test; no pretrained weights or GPU needed."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'CPT'))


@unittest.skipUnless(os.getenv('QWEN35_NATIVE_TEST')=='1','Run explicitly in .venv-qwen35-check')
class NativeQwen35Test(unittest.TestCase):
    def test_native_lora_training_and_generation(self):
        import torch
        from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
        from peft import get_peft_model,LoraConfig,TaskType
        from datasets import Dataset
        from qwen35_common import LORA_PATTERN
        torch.set_num_threads(2)
        text=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
            partial_rotary_factor=1.0,rope_parameters={'rope_type':'default','rope_theta':10000.0,'mrope_section':[1,1,2]})
        vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
            out_hidden_size=32,patch_size=2,temporal_patch_size=1,spatial_merge_size=1,num_position_embeddings=16)
        config=Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_id=63,
            video_token_id=62,vision_start_token_id=60,vision_end_token_id=61)
        model=AutoModelForImageTextToText.from_config(config)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,lora_alpha=4,
                                             use_rslora=True,target_modules=LORA_PATTERN))
        model.config.use_cache=False;model.config.get_text_config().use_cache=False
        model.enable_input_require_grads()
        x=torch.randint(3,59,(1,8))
        loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=x,use_cache=False).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                            for n,p in model.named_parameters() if 'lora_' in n))
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if 'visual' in n))
        model.zero_grad()
        data=Dataset.from_dict({'input_ids':[x[0].tolist()],'labels':[x[0].tolist()]})
        def collate(rows):
            ids=torch.tensor([row['input_ids'] for row in rows])
            return {'input_ids':ids,'labels':ids.clone(),'attention_mask':torch.ones_like(ids)}
        with tempfile.TemporaryDirectory(dir=ROOT/'CPT') as directory:
            trainer=Trainer(model=model,train_dataset=data,data_collator=collate,args=TrainingArguments(
                output_dir=directory,use_cpu=True,max_steps=1,per_device_train_batch_size=1,
                optim='adamw_torch',report_to='none',save_strategy='no',eval_strategy='no',
                logging_strategy='no',disable_tqdm=True,gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant':False}))
            result=trainer.train()
            self.assertTrue(__import__('math').isfinite(result.training_loss))
        model.eval();model.config.use_cache=True;model.config.get_text_config().use_cache=True
        with torch.inference_mode():
            generated=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=2,
                                     do_sample=False,pad_token_id=0,eos_token_id=2,suppress_tokens=[60,61,62,63])
        self.assertGreater(generated.shape[1],x.shape[1])
        print('Native Qwen3.5: finite loss/backward, frozen vision, Trainer step and cached generation passed.')


if __name__=='__main__':
    unittest.main()
