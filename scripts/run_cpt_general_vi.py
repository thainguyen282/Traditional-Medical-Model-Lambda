#!/usr/bin/env python3
"""Run the pinned Thinking CPT-only adapter with the saved base evaluation settings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from eval_general_vi import BENCHMARKS, source_rows, get_math_scorer
from eval_general_vi_vllm import (MODELS, EXP0015_CPT_REVISION,
                                 EXP0015_BASE, EXP0015_BASE_REVISION)

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen3-4b-thinking-cpt'


def preflight(benchmarks):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token = os.getenv('HF_TOKEN')
    api = HfApi(token=token)
    info = api.model_info(MODELS[MODEL], revision=EXP0015_CPT_REVISION)
    files = {item.rfilename for item in info.siblings}
    if not {'adapter_config.json', 'adapter_model.safetensors'}.issubset(files):
        raise ValueError('Pinned CPT checkpoint is missing adapter files')
    config = json.loads(Path(hf_hub_download(
        MODELS[MODEL], 'adapter_config.json', revision=EXP0015_CPT_REVISION,
        token=token)).read_text())
    if config['base_model_name_or_path'] != EXP0015_BASE or config['r'] != 64:
        raise ValueError('Unexpected CPT base or rank; review checkpoint metadata')
    api.model_info(EXP0015_BASE, revision=EXP0015_BASE_REVISION)
    snapshot_download(MODELS[MODEL], revision=EXP0015_CPT_REVISION, token=token,
                      allow_patterns=['adapter_config.json', 'adapter_model.safetensors'])
    for benchmark in benchmarks:
        rows, _ = source_rows(benchmark, token)
        print(f'[preflight] {benchmark}: {len(rows)} rows', flush=True)
    if any(b in ('mt-aime24', 'polymath') for b in benchmarks):
        get_math_scorer(ROOT / 'cache/general_vi')
    print(f'[preflight] CPT-only: {MODELS[MODEL]}@{EXP0015_CPT_REVISION}', flush=True)
    print(f'[preflight] base: {EXP0015_BASE}@{EXP0015_BASE_REVISION}', flush=True)


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'EVAL/outputs/general_vi_cpt' /
                        datetime.now().strftime('local-%Y%m%d-%H%M%S'))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.70)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1 or args.max_num_seqs < 1:
        parser.error('limit must be nonnegative; token and sequence counts must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU memory utilization must be between 0 and 1')
    preflight(args.benchmarks)
    if args.preflight_only:
        return 0
    output = args.output_dir.resolve()
    if any((output / MODEL / b).exists() for b in args.benchmarks):
        parser.error('Selected benchmark output already exists; use a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    evaluator = ROOT / 'scripts/eval_general_vi_vllm.py'
    failed = []
    for benchmark in args.benchmarks:
        command = [sys.executable, '-u', str(evaluator), '--model', MODEL,
                   '--protocol', 'thinking', '--benchmark', benchmark,
                   '--output-dir', str(output), '--limit', str(args.limit),
                   '--max-input-tokens', '4096', '--max-new-tokens', str(args.max_new_tokens),
                   '--max-num-seqs', str(args.max_num_seqs),
                   '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--enforce-eager']
        with (output / f'{benchmark}.log').open('w') as log:
            print(f'[evaluation] {benchmark}; log: {log.name}', flush=True)
            if subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode:
                failed.append(benchmark)
    if not failed and not args.limit:
        subprocess.run([sys.executable, str(evaluator), '--export-comparison',
                        '--comparison-models', MODEL, '--benchmarks', *args.benchmarks,
                        '--output-dir', str(output)], cwd=ROOT, check=True)
    print(f'[evaluation] outputs: {output}; failed benchmarks: {failed}', flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
