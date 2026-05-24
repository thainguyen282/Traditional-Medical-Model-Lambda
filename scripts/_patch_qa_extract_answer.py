"""Patch the `extract_answer` cell in all EVAL/Unsloth_QA_*.ipynb notebooks
with v2 logic.

v2 changes vs original:
- Strip Qwen reasoning artifacts (<tool_call>, <think>, ...) BEFORE matching.
- Add explicit Vietnamese patterns: "Đáp án [đúng]", "Trả lời", "Câu trả lời",
  "Chọn [đáp án]".
- Add "X: Đúng" / "X: Correct" semantic match (option marked correct in
  analysis blocks like "**B**: Đúng. ...").
- Fallback `\\b[A-E]\\b` takes FIRST match (model usually states the answer
  first, then analyzes all options). Old code took LAST, which mis-extracted
  responses like "B  Phân tích: **A**: Sai. **B**: Đúng. **C**: Sai. **D**: Sai"
  as D.

Idempotent — re-running skips notebooks that already contain the v2 marker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "EVAL"

OLD_ANCHOR = "_FINAL_ANSWER_PATTERN"   # present in every old notebook
# Anchor accepts v2 too — re-run upgrades v2 → v3.
NEW_MARKER = "# === extract_answer v3"
V2_MARKER = "# === extract_answer v2"

NEW_CELL = r'''# === extract_answer v3: v2 + Markdown-heading recognition (`### A` / `#### B`)
_REASONING_TAG_RE = re.compile(
    r"<(?:tool_call|think|tool_use|reasoning|thought|scratchpad)\b[^>]*>?",
    re.IGNORECASE,
)
# Markdown heading style answer (HIGHEST priority — common in datasets where
# the trained assistant trailer is `### A`, e.g. Maverfrick/New-*-multiplechoice).
# Matches `## A` ... `##### A`, optional bold, at end of line.
_MARKDOWN_HEADING_RE = re.compile(
    r"(?m)^#{2,5}\s+\*{0,2}([A-E])\*{0,2}\s*$",
)
# Explicit answer markers in EN and VN:
#   "Final Answer: B", "Final answer is B", "the answer is B",
#   "Đáp án: B", "Đáp án đúng là B", "Trả lời: B", "Câu trả lời là B",
#   "Chọn B", "Chọn đáp án B", "**B**" right after one of these triggers.
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+answer|đáp\s*án(?:\s+đúng)?|trả\s*lời|câu\s*trả\s*lời|"
    r"chọn(?:\s+đáp\s*án)?|the\s+answer\s+is|correct\s+answer|answer\s+is)"
    r"\s*(?:[:=]|\bis\b|\blà\b)?\s*"
    r"\*{0,2}\s*\(?\s*([A-E])(?:\b|\))",
    re.IGNORECASE,
)
# Analysis-block pattern: "**X**: Đúng" / "X: Đúng" / "X: correct" — model
# explicitly marks an option as correct while enumerating all options.
_CORRECT_OPTION_RE = re.compile(
    r"\*{0,2}([A-E])\*{0,2}\s*[:.)]\s*(?:Đúng\b|đúng\b|Correct\b)",
    re.IGNORECASE,
)
# Fallback patterns — tried in order; take FIRST match (NOT last) because
# models usually state the answer upfront then explain.
_ORDERED_FALLBACKS = [
    re.compile(
        r"(?:the|correct|final)\s*(?:answer|option|choice|letter)\s*(?:is)?\s*"
        r"[:=]?\s*\*{0,2}([A-E])\*{0,2}",
        re.IGNORECASE,
    ),
    re.compile(r"[\(\[]([A-E])[\)\]]"),
    re.compile(r"\b([A-E])\b"),
]

def extract_answer(model_response):
    """Extract answer letter from model response."""
    if not isinstance(model_response, str) or not model_response.strip():
        return "EMPTY_RESPONSE"

    cleaned = _REASONING_TAG_RE.sub("", model_response).strip()
    if not cleaned:
        return "EMPTY_RESPONSE"

    # 1. Markdown heading `### A` — trained Maverfrick/New-* format. Highest
    # priority because heading-style answers are unambiguous and intentional.
    m = _MARKDOWN_HEADING_RE.search(cleaned)
    if m:
        return m.group(1).upper()

    m = _EXPLICIT_ANSWER_RE.search(cleaned)
    if m:
        return m.group(1).upper()

    m = _CORRECT_OPTION_RE.search(cleaned)
    if m:
        return m.group(1).upper()

    # "final answer" / "đáp án" appeared but the letter slot wasn't parseable.
    if re.search(r"final\s*answer|đáp\s*án", cleaned, re.IGNORECASE):
        return "INVALID_FORMAT"

    for pattern in _ORDERED_FALLBACKS:
        matches = pattern.findall(cleaned)
        if matches:
            return matches[0].upper()
    return "NO_VALID_ANSWER"

def evaluate_answer(extracted_answer, correct_answer):
    """Evaluate if extracted answer is correct"""
    return extracted_answer.strip().upper() == correct_answer.strip().upper()
'''


def patch_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
        joined = "".join(src)
        if NEW_MARKER in joined:
            return "SKIP (already v3)"
        # Match the original anchor OR the v2 marker (re-run upgrades v2→v3).
        if (OLD_ANCHOR not in joined and V2_MARKER not in joined) \
                or "def extract_answer" not in joined:
            continue
        # Replace this entire cell — old extract_answer block.
        cell["source"] = [l + "\n" for l in NEW_CELL.split("\n")[:-1]]
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return "PATCHED"
    return "FAIL (anchor not found)"


def main() -> int:
    notebooks = sorted(EVAL_DIR.glob("Unsloth_QA_*.ipynb"))
    if not notebooks:
        print(f"no notebooks in {EVAL_DIR}", file=sys.stderr)
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
