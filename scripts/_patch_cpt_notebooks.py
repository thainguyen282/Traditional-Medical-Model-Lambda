#!/usr/bin/env python3
"""
One-shot editor: patch CPT eval notebooks so they can load the adapter
either from a local path (legacy) or from a HF repo+revision when the env
vars CPT_HF_REPO / CPT_HF_REVISION are set by CPT.py before nbconvert.

Idempotent — running twice is a no-op.
"""
import json
import sys
from pathlib import Path

NOTEBOOKS = [
    "EVAL/Unsloth_QA_CPT_new.ipynb",
    "EVAL/Unsloth_QA_CPT_VM14K.ipynb",
]

# New model_path block (replaces the hardcoded local path).
# Keeps variable name `model_path` for backward compat with later cells.
MODEL_PATH_BLOCK = '''# =========================
# Model path — prefer HF repo (set by CPT.py / bootstrap), else local adapter
# =========================
CPT_HF_REPO = os.environ.get("CPT_HF_REPO")
CPT_HF_REVISION = os.environ.get("CPT_HF_REVISION")
if CPT_HF_REPO:
    model_path = CPT_HF_REPO
    model_revision = CPT_HF_REVISION
    print(f"[eval] loading CPT adapter from HF: {CPT_HF_REPO}@{CPT_HF_REVISION}")
else:
    model_path = os.path.join(
        BASE_DIR,
        "..",
        "CPT",
        "cpt_qwen3_medical")
    model_revision = None
    print(f"[eval] loading CPT adapter from local: {model_path}")'''

# Old model_path block markers we need to replace. We do this by detecting
# the first line of the old block and swallowing until the blank line before
# `subset_size`. Simpler: do a string replace on the known exact substring.
OLD_MODEL_PATH_VARIANTS = [
    '''# Path tới checkpoint CPT đã train
model_path = os.path.join(
    BASE_DIR,
    "..",
    "CPT",
    "cpt_qwen3_medical")''',
    '''# Path tới checkpoint CPT đã train
model_path = os.path.join(
    BASE_DIR,
    "..",
    "CPT",
    "cpt_qwen3_medical"
)''',
]

OLD_FROM_PRETRAINED = '''model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    attn_implementation="sdpa",
    trust_remote_code=True,
)'''

NEW_FROM_PRETRAINED = '''_fp_kwargs = dict(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    attn_implementation="sdpa",
    trust_remote_code=True,
)
if model_revision:
    _fp_kwargs["revision"] = model_revision
model, tokenizer = FastLanguageModel.from_pretrained(**_fp_kwargs)'''


def patch_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = "".join(cell["source"])
    changed = False

    for variant in OLD_MODEL_PATH_VARIANTS:
        if variant in src:
            src = src.replace(variant, MODEL_PATH_BLOCK)
            changed = True
            break

    if OLD_FROM_PRETRAINED in src:
        src = src.replace(OLD_FROM_PRETRAINED, NEW_FROM_PRETRAINED)
        changed = True

    if changed:
        # Keep line endings correctly split for nbformat.
        lines = src.splitlines(keepends=True)
        cell["source"] = lines
    return changed


def patch_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n_patched = 0
    for cell in nb["cells"]:
        if patch_cell(cell):
            n_patched += 1
    if n_patched:
        path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return n_patched


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    for rel in NOTEBOOKS:
        p = repo_root / rel
        if not p.exists():
            print(f"SKIP (missing): {p}")
            continue
        n = patch_notebook(p)
        print(f"patched {n} cell(s) in {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
