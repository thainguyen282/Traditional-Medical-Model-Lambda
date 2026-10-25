#!/usr/bin/env python3
"""CPU preparation, standalone or before CPT in the same Lambda job."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'CPT'))
from pilot_data import digest, load_manifest, prepare


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('Preparation config must be a YAML mapping')
    for section in ('tokenizer', 'data'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Preparation config requires {section}')
    if not config['tokenizer'].get('name') or not re.fullmatch(r'[a-f0-9]{40}', config['tokenizer'].get('revision', '')):
        raise ValueError('Tokenizer requires a name and immutable revision')
    if not isinstance(config.get('seed'), int):
        raise ValueError('Preparation requires an integer seed')
    for key in ('source_manifest', 'output_dir'):
        path_value = config['data'].get(key)
        if not isinstance(path_value, str):
            raise ValueError(f'data.{key} is required')
        value = Path(path_value)
        if value.is_absolute() or '..' in value.parts or not str(value).startswith('CPT/'):
            raise ValueError(f'data.{key} must be a repository-relative CPT/ path')
    acquisition = config['data'].get('acquisition_config')
    if acquisition and (not isinstance(acquisition, str) or Path(acquisition).is_absolute() or '..' in Path(acquisition).parts):
        raise ValueError('data.acquisition_config must be repository-relative')
    return config


def run(config_path, manifest_override=None, check=False, fetch_sources=False):
    config = load_config(config_path)
    manifest = Path(manifest_override) if manifest_override else ROOT / config['data']['source_manifest']
    output = ROOT / config['data']['output_dir']
    if output.exists() and not check:
        raise FileExistsError(f'Prepared snapshot already exists: {output}; choose a new data.output_dir')
    if fetch_sources:
        from fetch_cpt_sources import load_config as load_acquisition, check_access, fetch
        acquisition_path = ROOT / config['data']['acquisition_config']
        acquisition = load_acquisition(acquisition_path)
        if check:
            check_access(acquisition)
            print('Pinned upstream access checked; no corpus or model weights downloaded.')
            return None
    elif not manifest.is_file():
        raise FileNotFoundError(
            f'Source manifest missing: {manifest}. Supply curated exports, or use '
            '--fetch-sources to fetch the original datasets directly.'
        )
    elif check:
        load_manifest(manifest)
        print('Preparation inputs verified; no training or GPU allocation performed.')
        return None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'], revision=config['tokenizer']['revision'])
    if fetch_sources:
        fetch(acquisition_path, manifest, tokenizer)
    load_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete snapshot; failed preparation leaves no final output.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'snapshot'
        stats = prepare(manifest, tokenizer, stage, seed=config['seed'])
        stats.update(tokenizer_name=config['tokenizer']['name'],
                     tokenizer_revision=config['tokenizer']['revision'],
                     preparation_config=config, preparation_config_sha256=digest(config_path))
        (stage / 'preparation.json').write_text(json.dumps(stats, indent=2))
        (stage / 'prepare_config.json').write_text(json.dumps(config, indent=2))
        if output.exists():
            raise FileExistsError(f'Output was created concurrently: {output}')
        os.rename(stage, output)
    print(f'Prepared snapshot: {output}')
    print('Prepared artifacts are ready for CPT.')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data/cpt-pilot/prepare.yaml')
    parser.add_argument('--manifest', type=Path, help='Optional source-manifest override')
    parser.add_argument('--fetch-sources', action='store_true', help='Fetch original pinned upstream datasets before preparation')
    parser.add_argument('--check', action='store_true', help='Validate exports without tokenizing or writing')
    args = parser.parse_args()
    try:
        run(args.config, args.manifest, args.check, args.fetch_sources)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
