# Vietnamese general benchmark results — October 6, 2026

These are **pilot results under this repository's protocol**, not a
reproduction of Table 33 in the [Qwen3 technical report](https://arxiv.org/html/2505.09388).
The report's Qwen3-4B scores are 48.8% / 65.5% / 20.0% / 19.0% in
non-thinking mode, or 68.8% / 63.6% / 60.0% / 42.2% in thinking mode
(MLogiQA / INCLUDE / MT-AIME24 / PolyMath). Direct comparison is invalid
because this run used the later
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
which supports only non-thinking mode. The paper also samples 10% of
INCLUDE and allows up to 32,768 output tokens; this run used all 550
Vietnamese INCLUDE rows, greedy decoding, and a 2,048-token math cap.

Job `tm-eval-1791333722` ended before the three-model, four-benchmark
comparison completed. The saved archive is
`logs/tm-eval-1791333722/medical-eval.tar.gz`. It contains complete results
for three Qwen3-4B-Instruct-2507 benchmarks and 92 of 500 PolyMath responses.
The Lambda instance has terminated. No completed Hub upload or comparison
table exists for this job.

| Model | Benchmark | Correct / Total | Accuracy | Status |
| --- | --- | ---: | ---: | --- |
| Qwen3-4B-Instruct-2507 | MLogiQA (vi) | 20 / 80 | 25.00% | Complete |
| Qwen3-4B-Instruct-2507 | INCLUDE (vi) | 311 / 550 | 56.55% | Complete |
| Qwen3-4B-Instruct-2507 | MT-AIME24 (vi) | 3 / 30 | 10.00% | Complete |
| Qwen3-4B-Instruct-2507 | PolyMath (vi) | — | — | Interrupted after 92 / 500 |
| Gemma-4-E2B-it | All four | — | — | Not started |
| Qwen3-4B + CPT | All four | — | — | Not started |

Qwen revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.
The Qwen runner recorded 33 unparsed answers on MLogiQA, 8 on INCLUDE,
and 26 on MT-AIME24; these count as incorrect. Several multiple-choice
answers were printed as a first-line letter followed by an explanation,
which the original parser missed. That parser is fixed for future runs,
but the archived scores above remain unchanged. Many MT-AIME responses
end mid-solution at the token cap. The first 92 PolyMath rows
are an incomplete prefix, so they must not be reported as a benchmark score.

The job reloads weights at the start of each benchmark. Earlier logs therefore
show repeated weight loading even after a benchmark finishes. The runner now
logs model load completion and progress after each batch. A new launch is
required to finish the remaining model and benchmark runs.
