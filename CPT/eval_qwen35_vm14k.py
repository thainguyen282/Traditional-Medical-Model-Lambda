#!/usr/bin/env python3
"""VM14K-only final-checkpoint evaluation and artifact publication for Qwen3.5."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from qwen35_common import ROOT, load_config, output_dir, publish


def score_final(row, raw, truncated, extract_answer, index):
    answer=None
    if not truncated and '</think>' in raw:
        answer=extract_answer(raw.rsplit('</think>',1)[1])
        if answer not in row['options']:
            answer=None
    return {'id':hashlib.sha256(row['question'].encode('utf-8')).hexdigest(),
            'row_index':index,'answer':row['answer'],'prediction':answer,
            'is_correct':answer==row['answer'],'invalid':answer is None,'truncated':truncated}


def evaluate(config, directory):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from peft import PeftModel
    sys.path.insert(0, str(ROOT/'scripts'))
    from eval_medical_baselines import prepare_dataset, build_prompt
    rows, protocol, metadata = prepare_dataset('vm14k', generation_mode='notebook')
    limit = config['evaluation'].get('limit', 0)
    if limit:
        rows = rows[:limit]
    adapter = json.loads((directory/'adapter_config.json').read_text())
    if adapter['base_model_name_or_path'] != config['model']['name'] or adapter['revision'] != config['model']['revision']:
        raise ValueError('Final adapter base/revision mismatch')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], revision=config['model']['revision'])
    base = AutoModelForImageTextToText.from_pretrained(config['model']['name'], revision=config['model']['revision'],
                                dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, str(directory), is_trainable=False)
    model.config.use_cache = True
    model.config.get_text_config().use_cache = True
    model.eval()
    result_dir = directory/'vm14k'; result_dir.mkdir(exist_ok=True)
    settings = {name:config['evaluation'][name] for name in ['max_new_tokens','temperature','top_p','top_k']}
    predictions=[]
    with (result_dir/'predictions.jsonl').open('w') as stream:
        for index,row in enumerate(rows):
            set_seed(config['evaluation']['seed']+index)
            prompt = tokenizer.apply_chat_template([{'role':'user','content':build_prompt(row,protocol['system_prompt'])}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
            if inputs['input_ids'].shape[1]>config['evaluation']['max_input_tokens']:
                raise ValueError('VM14K input exceeds the fixed prompt budget')
            with torch.inference_mode():
                generated=model.generate(**inputs,**settings,do_sample=True,pad_token_id=tokenizer.eos_token_id)
            generated=generated[0,inputs['input_ids'].shape[1]:]
            raw=tokenizer.decode(generated,skip_special_tokens=False)
            eos=model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            truncated=len(generated)>=settings['max_new_tokens'] and int(generated[-1]) not in eos
            record=score_final(row,raw,truncated,protocol['extract_answer'],index)
            record.update(generated_tokens=len(generated),raw=raw)
            predictions.append(record); stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
            print(f'[vm14k] {index+1}/{len(rows)}',flush=True)
    summary={'dataset':'VM14K traditional-medicine subset (564 questions)',
             'dataset_sha256':metadata['dataset_sha256'],'protocol_sha256':metadata['protocol_sha256'],
             'smoke':config.get('smoke',False),'total':len(rows),'generation':config['evaluation'],
             'correct':sum(r['is_correct'] for r in predictions),'invalid':sum(r['invalid'] for r in predictions),
             'truncated':sum(r['truncated'] for r in predictions)}
    summary['accuracy']=summary['correct']/summary['total']
    summary['invalid_rate']=summary['invalid']/summary['total']
    summary['truncation_rate']=summary['truncated']/summary['total']
    (result_dir/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args(); config=load_config(args.config); directory=output_dir(config)
    if args.diagnostics:
        publish(config,directory,diagnostics=True)
    else:
        evaluate(config,directory)
        if args.publish: publish(config,directory)


if __name__=='__main__':
    main()
