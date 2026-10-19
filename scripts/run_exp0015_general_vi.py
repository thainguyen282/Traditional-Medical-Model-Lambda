#!/usr/bin/env python3
"""Run the four general benchmarks for the pinned exp0015 SFT adapter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from eval_general_vi import BENCHMARKS, source_rows, get_math_scorer
from eval_general_vi_vllm import (EXP0015_BASE, EXP0015_BASE_REVISION,
                                 EXP0015_REVISION, MODELS)

ROOT = Path(__file__).resolve().parents[1]


def preflight():
    """Check credentials, exact model commits, adapter layout, and datasets."""
    import json
    from huggingface_hub import HfApi, hf_hub_download
    token = os.getenv('HF_TOKEN')
    api = HfApi(token=token)
    info = api.model_info(MODELS['exp0015'], revision=EXP0015_REVISION)
    files = {item.rfilename for item in info.siblings}
    if not {'adapter_config.json', 'adapter_model.safetensors'}.issubset(files):
        raise ValueError('Pinned exp0015 revision is missing its LoRA adapter files')
    config = json.loads(Path(hf_hub_download(MODELS['exp0015'], 'adapter_config.json',
                        revision=EXP0015_REVISION, token=token)).read_text())
    if config['base_model_name_or_path'] != EXP0015_BASE or config['r'] != 64:
        raise ValueError('Unexpected exp0015 base or LoRA rank; review checkpoint metadata')
    api.model_info(EXP0015_BASE, revision=EXP0015_BASE_REVISION)
    for benchmark in BENCHMARKS:
        rows, _ = source_rows(benchmark, token)
        print(f'[preflight] {benchmark}: {len(rows)} rows', flush=True)
    get_math_scorer(ROOT / 'cache' / 'general_vi')
    print(f'[preflight] adapter={MODELS["exp0015"]}@{EXP0015_REVISION}', flush=True)
    print(f'[preflight] underlying base={EXP0015_BASE}@{EXP0015_BASE_REVISION}', flush=True)


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.70)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--upload', action='store_true', help='Upload results/logs to HF_SFT_REPO after evaluation')
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1 or args.max_num_seqs < 1:
        parser.error('limit must be nonnegative; token and sequence counts must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU memory utilization must be between 0 and 1')
    if not args.preflight_only and args.output_dir is None:
        parser.error('--output-dir is required for evaluation')
    preflight()
    if args.preflight_only:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator = ROOT / 'scripts/eval_general_vi_vllm.py'
    failed = []
    for benchmark in BENCHMARKS:
        print(f'[evaluation] exp0015 / {benchmark}', flush=True)
        command = [sys.executable, str(evaluator), '--model', 'exp0015', '--protocol', 'thinking',
                   '--benchmark', benchmark, '--output-dir', str(output), '--limit', str(args.limit),
                   '--max-new-tokens', str(args.max_new_tokens), '--max-num-seqs', str(args.max_num_seqs),
                   '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--enforce-eager']
        if subprocess.run(command, cwd=ROOT).returncode:
            failed.append(benchmark)
    if not failed and not args.limit:
        subprocess.run([sys.executable, str(evaluator), '--export-comparison',
                        '--comparison-models', 'exp0015', '--benchmarks', *BENCHMARKS,
                        '--output-dir', str(output)], cwd=ROOT, check=True)
    print(f'[evaluation] outputs: {output}; failed benchmarks: {failed}', flush=True)
    if args.upload:
        revision = f'eval-general-vi-exp0015-{output.name}'
        subprocess.run([sys.executable, str(ROOT / 'scripts/lambda_push_to_hf.py'),
                        '--task', 'eval', '--repo', os.getenv('HF_SFT_REPO', MODELS['exp0015']),
                        '--revision', revision, '--eval-src', str(output),
                        '--log-file', str(output / 'evaluation.log')], cwd=ROOT, check=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
