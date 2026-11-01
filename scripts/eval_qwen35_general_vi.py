#!/usr/bin/env python3
"""Matched Qwen3.5-4B / mixed CPT evaluation on Vietnamese general tasks."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

from eval_general_vi import BENCHMARKS, ROOT, SCORER_COMMIT, get_math_scorer, prompt, score, source_rows
from eval_medical_baselines import (MODELS, QWEN35_BASE, QWEN35_BASE_REVISION,
    QWEN35_MIXED_REVISION, load_model, validate_qwen35_mixed)

MODEL_REVISIONS = {'qwen35-base': QWEN35_BASE_REVISION, 'qwen35-mixed': QWEN35_MIXED_REVISION}


def score_response(name, row, raw, truncated, math_equal=None):
    final = raw.rsplit('</think>', 1)[1] if '</think>' in raw and not truncated else ''
    result = score(name, row, final, math_equal)
    return {**result, 'response': raw, 'invalid': result['extracted_answer'] is None,
            'truncated': truncated}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def evaluate(key, datasets, args, versions, math_equal):
    import torch
    from transformers import set_seed
    model, tokenizer, _ = load_model(key, MODEL_REVISIONS[key], os.getenv('HF_TOKEN'))
    model.config.use_cache = True
    model.config.get_text_config().use_cache = True
    summaries = []
    try:
        for name, (all_rows, metadata) in datasets.items():
            rows = all_rows[:args.limit] if args.limit else all_rows
            target = args.output_dir / key / name
            target.mkdir(parents=True, exist_ok=False)
            settings = dict(do_sample=True, temperature=.6, top_p=.95, top_k=20,
                            max_new_tokens=args.max_new_tokens)
            manifest = {**metadata, 'model': MODELS[key], 'model_revision': MODEL_REVISIONS[key],
                        'base_model': QWEN35_BASE, 'base_revision': QWEN35_BASE_REVISION,
                        'benchmark': name, 'limit': args.limit, 'evaluated_rows': len(rows),
                        'seed': args.seed, 'seed_policy': 'seed + row index, reset for each model/benchmark',
                        'decoding': settings, 'thinking_enabled': True,
                        'max_input_tokens': args.max_input_tokens,
                        'quantization': 'nf4-double-quant-bfloat16', 'batch_size': 1,
                        'versions': versions, 'scorer_revision': SCORER_COMMIT if not rows[0]['options'] else None,
                        'status': 'running'}
            write_json(target / 'run_config.json', manifest)
            predictions = []
            try:
                with (target / 'responses.jsonl').open('w', encoding='utf-8') as stream:
                    for index, row in enumerate(rows):
                        set_seed(args.seed + index)
                        text = tokenizer.apply_chat_template(
                            [{'role': 'user', 'content': prompt(name, row)}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=True)
                        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
                        length = inputs['input_ids'].shape[1]
                        if length > args.max_input_tokens:
                            raise ValueError(f'{name}/{index}: input exceeds budget ({length}); refusing truncation')
                        with torch.inference_mode():
                            output = model.generate(**inputs, **settings, pad_token_id=tokenizer.pad_token_id)
                        tokens = output[0, length:]
                        eos = model.generation_config.eos_token_id
                        eos = eos if isinstance(eos, list) else [eos]
                        truncated = len(tokens) >= args.max_new_tokens and int(tokens[-1]) not in eos
                        raw = tokenizer.decode(tokens, skip_special_tokens=False)
                        record = {'index': index, 'generated_tokens': len(tokens),
                                  **score_response(name, row, raw, truncated, math_equal)}
                        predictions.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()
                        print(f'[{key}/{name}] {index + 1}/{len(rows)}', flush=True)
                summary = {'model': key, 'benchmark': name, 'status': 'complete',
                           'total': len(rows), 'correct': sum(r['is_correct'] for r in predictions),
                           'invalid': sum(r['invalid'] for r in predictions),
                           'truncated': sum(r['truncated'] for r in predictions), 'smoke': bool(args.limit)}
                summary['accuracy'] = summary['correct'] / summary['total']
                write_json(target / 'summary.json', summary)
                summaries.append(summary)
                manifest['status'] = 'complete'
            except Exception as error:
                manifest.update(status='failed', error=str(error))
                raise
            finally:
                write_json(target / 'run_config.json', manifest)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return summaries


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_REVISIONS, default=list(MODEL_REVISIONS))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or min(args.max_input_tokens, args.max_new_tokens) < 1:
        parser.error('limit must be nonnegative and token budgets positive')
    if len(set(args.models)) != len(args.models) or len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error('duplicate models or benchmarks')
    if args.output_dir.exists():
        parser.error('Use a new output directory to preserve previous results')
    from huggingface_hub import HfApi
    token = os.getenv('HF_TOKEN')
    HfApi(token=token).model_info(QWEN35_BASE, revision=QWEN35_BASE_REVISION)
    if 'qwen35-mixed' in args.models:
        validate_qwen35_mixed(token)
    datasets = {name: source_rows(name, token) for name in args.benchmarks}
    math_equal = get_math_scorer(ROOT / 'cache/general_vi') if any(
        name in args.benchmarks for name in ('mt-aime24', 'polymath')) else None
    for name, (rows, _) in datasets.items():
        print(f'[preflight] {name}: {len(rows)} Vietnamese questions', flush=True)
    if args.preflight_only:
        return 0
    from eval_medical_baselines import check_runtime
    versions = check_runtime(args.models)
    summaries = []
    for key in args.models:
        summaries.extend(evaluate(key, datasets, args, versions, math_equal))
    # Only a complete, unlimited run is eligible for a comparison table.
    if not args.limit:
        fields = ['model', 'benchmark', 'total', 'correct', 'accuracy', 'invalid', 'truncated']
        with (args.output_dir / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(summaries)
        lines = ['| Model | Benchmark | Accuracy | Invalid | Truncated |', '|---|---|---:|---:|---:|']
        lines.extend(f"| {r['model']} | {r['benchmark']} | {r['accuracy']:.2%} | {r['invalid']} | {r['truncated']} |" for r in summaries)
        (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
