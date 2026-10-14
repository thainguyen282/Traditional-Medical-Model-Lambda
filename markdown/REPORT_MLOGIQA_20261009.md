# MLogiQA Vietnamese evaluation — October 9, 2026

## Completed baseline run

Measured results from Lambda job `tm-eval-1791574404`, evaluated on all 80
Vietnamese test questions. These are the archived scores, without rescoring.

| Model | Correct / total | Accuracy | Invalid answers | Token-limit hits | Model evaluation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Thinking-2507 | 47 / 80 | 58.75% | 19 | 10 | 22m 43s |
| Gemma-4-E2B-it (thinking enabled) | 43 / 80 | 53.75% | 7 | 0 | 12m 43s |

Invalid answers count as incorrect. Responses stopped by the token limit are
always invalid; other invalid responses did not yield a choice using the
archived parser. Thus, Qwen has 9 additional parsing failures and Gemma has 7.
Times include model initialization and inference, but exclude instance setup,
environment installation, and result upload.

### Configuration and provenance

- GPU: Lambda `gpu_1x_a100_sxm4` (A100 SXM4 40 GB), one model at a time.
- Backend: vLLM 0.31.0, BF16 weights, eager execution.
- Maximum concurrent sequences: 8; GPU memory utilization: 0.60.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, seed 3407.
- Output limit: 8,192 tokens; engine context limit: 12,288 tokens.
- Thinking enabled for both baselines; Gemma uses text-only initialization.
- Dataset: `Qwen/P-MMEval`, `mlogiqa/test/vi.jsonl`.
- Dataset revision: `47bb647f35fdd6f5374826b3f5d4f84eb5b5afce`.
- Source SHA256: `9ec344f486c64d9a86328daa9268a03210105a4b51fb6f24fa7ffaf30226ca17`.
- Qwen revision: `768f209d9ea81521153ed38c47d515654e938aea`.
- Gemma revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.

The run does not establish reproduction of a published paper score: the
checkpoint, output budget, prompting, and scoring must all match for that claim.

### Saved outputs

- [Hugging Face run folder](https://huggingface.co/Maverfrick/Venera-Traditional-Model/tree/eval-medical-tm-eval-1791574404/eval/tm-eval-1791574404)
- [Qwen results](https://huggingface.co/Maverfrick/Venera-Traditional-Model/tree/eval-medical-tm-eval-1791574404/eval/tm-eval-1791574404/qwen3-4b-thinking-2507/mlogiqa)
- [Gemma results](https://huggingface.co/Maverfrick/Venera-Traditional-Model/tree/eval-medical-tm-eval-1791574404/eval/tm-eval-1791574404/gemma4-e2b/mlogiqa)
- Local archive: `logs/tm-eval-1791574404/medical-eval.tar.gz`.

## CPT evaluation

The existing `huyennkai/Venera-CPT-TMModel` adapter was trained on
`unsloth/Qwen3-4B-Instruct-2507`. It is **not** a Thinking-2507 CPT adapter.
Its evaluation uses the instruction model's final-answer format and must not
be interpreted as the effect of CPT on the thinking baseline above.

Status: launched as `tm-eval-1791580264`; no CPT score is reported yet.

- Instance: `93cac22c16db4ebbb9d5cc6e5806098a`, A100 SXM4 40 GB.
- Adapter revision verified before launch: `43053107c4f0931c9f61977923cfba75a61d8606`.
- Base revision verified before launch: `992063681dc2f7de4ee976110199552935cad284`.
- Protocol: `instruct-vllm`, BF16, eager, batch 8, memory utilization 0.60,
  temperature 0.6, top-p 0.95, top-k 20, seed 3407, output cap 8,192.
- Local launcher log: `logs/mlogiqa-cpt-20261009-launch.log`.
- Expected output branch after upload: `eval-medical-tm-eval-1791580264`
  in `Maverfrick/Venera-Traditional-Model`.
- The launcher is configured to terminate the instance after completion.
