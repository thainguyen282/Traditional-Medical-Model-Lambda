#!/usr/bin/env python3
"""
Patch SFT eval notebooks to prefer an HF repo+revision over the local
fallback path. Intended to be used together with bootstrap.sh run_sft,
which pushes the freshly-trained adapter to HF and sets the env vars
SFT_HF_REPO / SFT_HF_REVISION before running each notebook.

Fallback order when env vars missing:
    1. SFT_HF_REPO (+ SFT_HF_REVISION) — load from Hub
    2. SFT_MODEL_PATH — load from that local path
    3. The hardcoded group_outputs/... path that was in the notebook

Idempotent.
"""
import json
import re
import sys
from pathlib import Path

NOTEBOOKS = [
    "EVAL/Unsloth_QA_SFT_new.ipynb",
    "EVAL/Unsloth_QA_SFT_VM14K.ipynb",
    "EVAL/Unsloth_QA_CPT_SFT_new.ipynb",
    "EVAL/Unsloth_QA_CPT_SFT_VM14K.ipynb",
]

# Match `model_path = os.environ.get("SFT_MODEL_PATH") or os.path.join(\n ... \n)`
# capture the os.path.join(...) payload (inside parens) so we can reuse it.
MODEL_PATH_RE = re.compile(
    r'model_path\s*=\s*os\.environ\.get\("SFT_MODEL_PATH"\)\s*or\s*os\.path\.join\(([^)]*)\)',
    re.DOTALL,
)

NEW_MODEL_PATH_TEMPLATE = '''SFT_HF_REPO = os.environ.get("SFT_HF_REPO")
SFT_HF_REVISION = os.environ.get("SFT_HF_REVISION")
if SFT_HF_REPO:
    model_path = SFT_HF_REPO
    model_revision = SFT_HF_REVISION
    print(f"[eval] loading SFT adapter from HF: {{SFT_HF_REPO}}@{{SFT_HF_REVISION}}")
elif os.environ.get("SFT_MODEL_PATH"):
    model_path = os.environ["SFT_MODEL_PATH"]
    model_revision = None
    print(f"[eval] loading SFT adapter from SFT_MODEL_PATH: {{model_path}}")
else:
    model_path = os.path.join({payload})
    model_revision = None
    print(f"[eval] loading SFT adapter from local: {{model_path}}")'''

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

    m = MODEL_PATH_RE.search(src)
    if m:
        new_block = NEW_MODEL_PATH_TEMPLATE.format(payload=m.group(1).strip())
        src = src[:m.start()] + new_block + src[m.end():]
        changed = True

    if OLD_FROM_PRETRAINED in src:
        src = src.replace(OLD_FROM_PRETRAINED, NEW_FROM_PRETRAINED)
        changed = True

    if changed:
        cell["source"] = src.splitlines(keepends=True)
    return changed


def patch_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = sum(patch_cell(c) for c in nb["cells"])
    if n:
        path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return n


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
