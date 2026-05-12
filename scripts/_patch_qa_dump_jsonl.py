"""Patch all EVAL/Unsloth_QA_*.ipynb notebooks to dump per-sample JSONL with
the model's raw response + the extracted answer.

Each notebook gets a `responses.jsonl` written into its `output_dir/` (alongside
the existing `model_output.txt`), one row per sample, schema:

    {q_idx, question, options, valid_option_keys,
     raw_response,        # model output BEFORE format-extract
     extracted_answer,    # ONE of: a letter A-E, or sentinel
                          # (EMPTY_RESPONSE / INVALID_FORMAT / NO_VALID_ANSWER)
     correct_answer, is_correct, is_deviation, topics, difficulty}

Idempotent — re-running skips already-patched notebooks (marker comment).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "EVAL"

ANCHOR_REQUIRES = "extracted_answer = extract_answer("
ANCHOR_INSERT_BEFORE = "output_buffer.append(output_text)"
MARKER = "# === dump JSONL: raw + extracted answer"

DUMP_TEMPLATE = '''\
# === dump JSONL: raw + extracted answer (parser-friendly per-sample log)
_dump_path = os.path.join(output_dir, "responses.jsonl")
with open(_dump_path, "a", encoding="utf-8") as _fdump:
    _fdump.write(json.dumps({
        "q_idx": total_count,
        "question": sample["question"],
        "options": sample["options"],
        "valid_option_keys": valid_option_keys,
        "raw_response": model_generated_text,
        "extracted_answer": extracted_answer,
        "correct_answer": correct_option_letter,
        "is_correct": bool(is_correct),
        "is_deviation": bool(is_deviation),
        "topics": medical_topics,
        "difficulty": difficulty_level,
    }, ensure_ascii=False) + "\\n")
'''


def patch_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
        joined = "".join(src)
        if ANCHOR_REQUIRES not in joined:
            continue
        if MARKER in joined:
            return "SKIP (already patched)"
        # Locate the anchor line. Must match ALL lines that contain it; insert
        # before the FIRST occurrence (the main loop one).
        anchor_idx: int | None = None
        for i, line in enumerate(src):
            if ANCHOR_INSERT_BEFORE in line:
                anchor_idx = i
                break
        if anchor_idx is None:
            return "FAIL (anchor missing)"
        # Detect indent of the anchor line and apply same indent to every dump line.
        anchor_line = src[anchor_idx]
        indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip())]
        dump_lines = [
            (indent + line + "\n") if line else (indent.rstrip() + "\n")
            for line in DUMP_TEMPLATE.split("\n")[:-1]  # trailing "" from final \n
        ]
        src[anchor_idx:anchor_idx] = dump_lines
        cell["source"] = src
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return f"PATCHED (+{len(dump_lines)} lines)"
    return "FAIL (no main-loop cell)"


def main() -> int:
    notebooks = sorted(EVAL_DIR.glob("Unsloth_QA_*.ipynb"))
    if not notebooks:
        print(f"no notebooks found in {EVAL_DIR}", file=sys.stderr)
        return 1
    width = max(len(p.name) for p in notebooks)
    rc = 0
    for nb in notebooks:
        result = patch_notebook(nb)
        print(f"  {nb.name:<{width}}  {result}")
        if result.startswith("FAIL"):
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
