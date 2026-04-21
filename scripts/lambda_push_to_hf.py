#!/usr/bin/env python3
"""
Push a trained artifact (LoRA adapter + tokenizer) to a Hugging Face repo.

Runs on the Lambda instance after training finishes. Two modes:

  --task cpt  --src <dir> --repo huyennkai/Venera-CPT-TMModel --revision <tag>
      Uploads <dir> (expected to be CPT/cpt_qwen3_medical) as-is.

  --task sft  --config <yaml> --repo huyennkai/Venera-Traditional-Model --revision <tag>
      Reads the config, reproduces train_command.py's output layout
      (group_outputs/outputs/{SFT|CPT_SFT}/<config-name>/model), uploads that.
      The revision name is prefixed with cpt-sft- or sft- based on whether
      config["model"]["name"] references a CPT checkpoint.

Optional extras (both tasks):
  --eval-src <dir>   Upload a directory under eval/<basename>/ in the same
                     revision. Can be passed multiple times. Missing paths
                     are skipped (non-fatal).
  --log-file <path>  Upload a single file as train.log at the revision root.

The script creates the repo if it does not exist, creates the branch
(revision) if it does not exist, and uploads via HfApi.upload_folder.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, create_branch, create_repo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


def log(msg: str) -> None:
    print(f"[push_to_hf] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[push_to_hf] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s or "run"


def locate_sft_artifact(config_path: Path) -> tuple[Path, str, Path]:
    """
    Mirror src/commands/train_command.py:
        pipeline_label = "CPT_SFT" if "cpt" in model_name.lower() else "SFT"
        output_dir = <train.output_dir>/<pipeline_label>
        model_output_dir = <output_dir>/model
        eval_output_dir = <output_dir>/benchmark_evaluations
    Returns (model_dir, pipeline_label, benchmark_dir).
    """
    cfg = yaml.safe_load(config_path.read_text())
    model_name = cfg.get("model", {}).get("name", "")
    pipeline_label = "CPT_SFT" if "cpt" in model_name.lower() else "SFT"
    base_output = cfg.get("training", {}).get("output_dir", "./outputs")
    pipeline_dir = Path(base_output) / pipeline_label
    model_dir = pipeline_dir / "model"
    benchmark_dir = pipeline_dir / "benchmark_evaluations"
    return model_dir, pipeline_label, benchmark_dir


def ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, token=token)
        log(f"repo exists: {repo_id}")
    except RepositoryNotFoundError:
        log(f"creating repo: {repo_id}")
        create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)


def ensure_branch(repo_id: str, revision: str, token: str) -> None:
    try:
        create_branch(repo_id=repo_id, branch=revision, token=token,
                      exist_ok=True)
        log(f"branch ready: {revision}")
    except HfHubHTTPError as e:
        die(f"create_branch({revision}) failed: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["cpt", "sft"], required=True)
    p.add_argument("--repo", required=True, help="HF repo id e.g. user/name")
    p.add_argument("--revision", required=True,
                   help="Branch/tag to push to (created if missing)")
    p.add_argument("--src", help="Source dir (required for --task cpt)")
    p.add_argument("--config",
                   help="Path to training config yaml (required for --task sft)")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--eval-src", action="append", default=[],
                   help="Directory to upload under eval/<basename>/. Repeatable. "
                        "Missing paths are skipped.")
    p.add_argument("--log-file",
                   help="File to upload as train.log at the revision root. "
                        "Skipped if missing.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        die("HF_TOKEN not set")

    auto_eval_srcs: list[Path] = []
    if args.task == "cpt":
        if not args.src:
            die("--src required for cpt")
        artifact_dir = Path(args.src)
        revision = f"cpt-{slugify(args.revision)}"
        pipeline_label = "CPT"
    else:
        if not args.config:
            die("--config required for sft")
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            die(f"config not found: {cfg_path}")
        artifact_dir, pipeline_label, benchmark_dir = locate_sft_artifact(cfg_path)
        prefix = "cpt-sft" if pipeline_label == "CPT_SFT" else "sft"
        cfg_slug = slugify(f"{cfg_path.parent.parent.name}-{cfg_path.parent.name}")
        revision = f"{prefix}-{cfg_slug}-{slugify(args.revision)}"
        auto_eval_srcs.append(benchmark_dir)

    if not artifact_dir.exists():
        die(f"artifact dir missing: {artifact_dir}")
    if not any(artifact_dir.iterdir()):
        die(f"artifact dir empty: {artifact_dir}")

    log(f"task={args.task} label={pipeline_label} src={artifact_dir}")
    log(f"target: {args.repo}@{revision}")

    api = HfApi(token=token)
    ensure_repo(api, args.repo, token)
    ensure_branch(args.repo, revision, token)

    commit_msg = args.commit_message or f"Upload {pipeline_label} artifact ({args.revision})"
    log(f"uploading {artifact_dir} -> {args.repo}@{revision}")
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=args.repo,
        revision=revision,
        commit_message=commit_msg,
        token=token,
        ignore_patterns=["*.pyc", "__pycache__/*", ".git/*"],
    )

    eval_srcs = auto_eval_srcs + [Path(s) for s in args.eval_src]
    for src_path in eval_srcs:
        if not src_path.exists():
            log(f"eval src missing, skip: {src_path}")
            continue
        if not any(src_path.iterdir()):
            log(f"eval src empty, skip: {src_path}")
            continue
        dest = f"eval/{src_path.name}"
        log(f"uploading eval: {src_path} -> {args.repo}@{revision}:{dest}")
        api.upload_folder(
            folder_path=str(src_path),
            repo_id=args.repo,
            revision=revision,
            path_in_repo=dest,
            commit_message=f"{commit_msg} [eval:{src_path.name}]",
            token=token,
            ignore_patterns=["*.pyc", "__pycache__/*", ".git/*",
                             ".ipynb_checkpoints/*"],
        )

    if args.log_file:
        log_path = Path(args.log_file)
        if log_path.exists() and log_path.stat().st_size > 0:
            log(f"uploading log: {log_path} -> {args.repo}@{revision}:train.log")
            api.upload_file(
                path_or_fileobj=str(log_path),
                path_in_repo="train.log",
                repo_id=args.repo,
                revision=revision,
                commit_message=f"{commit_msg} [train.log]",
                token=token,
            )
        else:
            log(f"log file missing/empty, skip: {log_path}")

    log(f"DONE. https://huggingface.co/{args.repo}/tree/{revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
