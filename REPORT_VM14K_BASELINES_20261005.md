# VM14K baseline analysis — 2026-10-05

The latest Lambda job, `tm-eval-1791210745`, evaluated the pinned
`Qwen/Qwen3-4B-Instruct-2507` and `google/gemma-4-E2B-it` checkpoints on the
same 564 VM14K traditional-medicine items. Both used no thinking, greedy
answer-only decoding restricted to option labels, NF4 quantization, and batch
size 4. Both runs completed with zero invalid answers and zero token-limit hits.

| Model | Correct / 564 | Accuracy |
| --- | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 264 / 564 | 46.81% |
| Gemma-4-E2B-it | 225 / 564 | 39.89% |

**Qwen led by 39 items, or 6.91 percentage points.** Of the 564 paired items,
both models got 125 right and 200 wrong; Qwen alone got 139 right, versus 100
for Gemma alone. The two models selected the same answer on 239 items. The new
run's per-item predictions exactly match the previous run (`tm-eval-1791208642`)
for both checkpoints, confirming deterministic reproduction under this protocol.

## Where the gap appears

| Difficulty | Items | Qwen correct | Gemma correct | Qwen − Gemma |
| --- | ---: | ---: | ---: | ---: |
| Easy | 112 | 46 (41.07%) | 56 (50.00%) | −8.93 pp |
| Moderate | 339 | 157 (46.31%) | 125 (36.87%) | +9.44 pp |
| Hard | 39 | 17 (43.59%) | 16 (41.03%) | +2.56 pp |
| Challenging | 74 | 44 (59.46%) | 28 (37.84%) | +21.62 pp |

Qwen's lead comes mainly from Moderate and Challenging items. Gemma does better
on Easy items. The Hard subset is small, so its one-item difference is not
informative.

The gap is also sensitive to the correct option letter. Qwen got 145/248 items
whose answer was B or C, while Gemma got 82/248. For A or D, Gemma got 143/316
and Qwen got 119/316. Qwen chose B or C on 358/564 items (63.48%), although
those letters were correct on 248/564 items (43.97%). This is evidence of an
answer-position tendency worth checking; it does not by itself explain whether
the models differ in traditional-medicine knowledge.

## Repeated items and interpretation

The 564 rows represent **232 distinct question/choice/answer-content groups**:
35 appear once, 65 twice, 131 three times, and one six times. In 164 groups,
the correct answer's letter changes across versions of the same item. Within
those groups, Qwen is correct on 118/201 B-or-C versions versus Gemma's 66/201;
on A-or-D versions, Qwen is correct on 98/248 versus Gemma's 103/248. This
reinforces the need to check position sensitivity before interpreting the
raw accuracy gap as a knowledge gap.

Weighting each distinct group equally gives 47.77% for Qwen and 40.88% for
Gemma, so the point-estimate gap remains about 6.9 points. A paired bootstrap
that resamples the 232 groups (20,000 draws, seed 3407) gives a 95% interval
of **−0.18 to +13.99 percentage points** for the row-weighted Qwen − Gemma
difference. This interval includes a tie; the 564 rows should not be treated
as 564 independent questions.

These numbers describe constrained multiple-choice answering on this VM14K
subset. They do not measure free-form explanation quality, ROUGE/BLEU,
LLM-as-judge scores, or broader clinical performance. The useful next check is
to score one balanced answer-position version per distinct item, retaining the
same prompts and checkpoint revisions.

The complete [uploaded run](https://huggingface.co/Maverfrick/Venera-Traditional-Model/tree/eval-medical-tm-eval-1791210745/eval/tm-eval-1791210745)
includes both per-item response files, manifests, summaries, and comparison
tables. A local copy is under
`EVAL/outputs/medical_baselines/tm-eval-1791210745/`.
