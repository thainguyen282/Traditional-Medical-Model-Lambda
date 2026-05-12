# VeneraMedicalM1 — Fine-tuning LLM cho Y Học Cổ Truyền

Dự án nghiên cứu fine-tuning Large Language Model chuyên biệt cho lĩnh vực **Y Học Cổ Truyền** (Đông Y), dùng pipeline hai giai đoạn: **Continued Pre-Training (CPT)** → **Supervised Fine-Tuning (SFT)**.

Hỗ trợ 2 chế độ chạy:
- **Local** — train trên máy có GPU (hoặc kiểm thử dry-run).
- **Lambda Cloud (khuyến nghị)** — launch từ máy local, pre-baked Docker image tự chạy train + auto-eval + push artifact & eval outputs lên Hugging Face, tự terminate khi xong. Xem [scripts/LAMBDA_README.md](scripts/LAMBDA_README.md).

## 🎯 Model output

Adapter LoRA đã train + eval, public trên Hugging Face. Mỗi branch là **snapshot tự chứa**: adapter + eval outputs + train.log.

### Hiện tại (25/04/2026, SFT trên `Maverfrick/TradMedInstruct-multiplechoice`)

| Use case | HF repo @ branch | VM14K Acc | Bench nhỏ Acc | Deviation |
|---|---|:--:|:--:|:--:|
| 🥈 **Cân bằng** (đa chuyên khoa + Đông y) | [`Venera-Traditional-Model` @ `cpt-sft-…-tm-sft-1776945292`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776945292) | **48.05%** | 58.10% | **0% / 0%** |
| 🧠 **Generation tự do** (không cần Q&A format) | [`Venera-CPT-TMModel` @ `cpt-tm-cpt-1776775114`](https://huggingface.co/huyennkai/Venera-CPT-TMModel/tree/cpt-tm-cpt-1776775114) | **49.11%** | 58.10% | 0% |
| ⚠️ SFT-only (regression — không khuyến nghị ship) | [`Venera-Traditional-Model` @ `sft-…-tm-sft-1776949731`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/sft-unsloth-qwen3-4B-exp0001-tm-sft-1776949731) | 34.04% | 30.48% | 3.01% / 7.62% |

### Historical — dataset cũ `Maverfrick/TradMedInstruct-2topics-3546` (22/04/2026)

| Use case | HF repo @ branch | VM14K Acc | Bench nhỏ Acc | Deviation |
|---|---|:--:|:--:|:--:|
| 🥇 **TCM thuần** (Q&A Đông y hẹp) | [`Venera-Traditional-Model` @ `sft-…-tm-sft-1776771092`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/sft-unsloth-qwen3-4B-exp0001-tm-sft-1776771092) | 46.45% | 🥇 **63.81%** | 0% |
| 🥈 Cân bằng (đa chuyên khoa + Đông y) | [`Venera-Traditional-Model` @ `cpt-sft-…-tm-sft-1776784126`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776784126) | 47.16% | 62.86% | 1.06% |

### exp0003 — CPT-SFT trên `2topics-3546` + NEFTune (13/05/2026)

| Use case | HF repo @ branch | VM14K Acc | Bench nhỏ Acc | Deviation |
|---|---|:--:|:--:|:--:|
| 🥇 **Cân bằng (best 2topics)** — CPT-SFT + NEFTune α=5 | [`Venera-Traditional-Model` @ `cpt-sft-…-exp0003-tm-sft-1778605702`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/cpt-sft-unsloth-qwen3-4B-exp0003-tm-sft-1778605702) | **47.52%** | 🥇 **63.81%** | **0%** |

**Nhận xét exp0003** (vs CPT-SFT 2topics no-NEFTune `tm-sft-1776784126`):
- VM14K **+0.36%** (47.16 → 47.52), Bench nhỏ **+0.95%** (62.86 → 63.81 — tie 🥇 với SFT-only).
- **Deviation 1.06% → 0%** — NEFTune chống format-collapse rõ.
- Chạy **A10 24GB ở 4-bit** (vs A100 bf16 cũ) — về lý thuyết 4-bit kém ~0.5-1% nhưng vẫn beat. Effective batch giữ = 4 (batch 2 × grad_accum 2). Config: [`exp0003/config_cpt.yaml`](configs/experiments/unsloth-qwen3-4B/exp0003/config_cpt.yaml).

Baseline: **Qwen3-4B-Instruct-2507 (gốc)** — VM14K 48.05%, Bench nhỏ 53.33%, deviation 0%.

**Nhận xét**:
- Dataset mới (`multiplechoice`, 3379 rows) → **CPT-SFT không còn deviation nào** (0% / 0%) và **ngang Instruct trên VM14K**. Đánh đổi: Bench nhỏ −4.76% vs dataset cũ (vì dataset cũ `2topics-3546` vốn gần distribution Bench nhỏ hơn).
- **SFT-only trên multiplechoice bị regression nặng** (VM14K −12.41%, Bench −33.33%, deviation tăng). Khả năng: multiplechoice format quá narrow, SFT thẳng không có CPT adapter đệm → overwrite general Q&A capability của Instruct. **CPT → SFT pipeline bắt buộc với dataset này**.
- So sánh với kỳ 16/04: CPT-only VM14K **+5.67%**, CPT-SFT deviation giảm ~10× (11.35% → 0%).
- Chi tiết cải thiện: [REPORT_fix.md](REPORT_fix.md); số liệu đầy đủ: [REPORT.md §8](REPORT.md).

### Load model trong code

```python
from unsloth import FastLanguageModel

# CPT-SFT (dataset mới multiplechoice — khuyến nghị cho pipeline inference y học thông thường)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="huyennkai/Venera-Traditional-Model",
    revision="cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776945292",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Kinh Dương Minh đi qua những huyệt nào trên mặt?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
out = model.generate(inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Đổi `model_name` + `revision` theo bảng trên để load 2 model còn lại. `revision=` là bắt buộc — nhánh `main` của HF repo có thể khác.

## 🔍 Nhận xét định tính (inference so sánh, 23/04)

Ngoài MCQ benchmark (VM14K / Bench nhỏ) chỉ đo *câu trả lời đúng A/B/C/D*, tôi chạy thêm **5 câu hỏi mở về Đông y** trên cả 6 model để kiểm tra chất lượng generation. Dataset + response đầy đủ ở HF:

📊 [`Venera-Traditional-Model @ inference-tm-inference-1776953351`](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/inference-tm-inference-1776953351) — chứa [`comparison_20260423_142330.md`](https://huggingface.co/huyennkai/Venera-Traditional-Model/blob/inference-tm-inference-1776953351/eval/comparison/comparison_20260423_142330.md) (table readable), `.csv` (full text cho Excel), `.json` (structured).

### Metric định lượng rút ra từ 5 × 6 = 30 response

| Model | Avg length (chars) | `<tool_call>` leak | `<think>` leak | Empty response |
|---|:--:|:--:|:--:|:--:|
| Qwen3-4B-Instruct (gốc) | 1535 | ✅ 0/5 | ✅ 0/5 | ✅ 0/5 |
| **CPT-only** | **1574** | ✅ **0/5** | ✅ **0/5** | ✅ **0/5** |
| CPT-SFT (multiplechoice) | 611 | 🚨 **5/5** | 1/5 | 0/5 |
| SFT-only (multiplechoice) | 657 | 🚨 4/5 | 🚨 3/5 | 0/5 |
| SFT-only (2topics) | 557 | 🚨 4/5 | 2/5 | 0/5 |
| CPT-SFT (2topics) | 285 | 2/5 | 1/5 | 🚨 **2/5** |

### Kết luận chính

1. **Chỉ Instruct và CPT-only giữ được format sạch**. CPT-only thậm chí có response **dài hơn Instruct** (1574 vs 1535 chars, trung bình), có cấu trúc tốt hơn, biết viết thuật ngữ chuyên môn (vd ghi đúng `Hợp Cốc (L.I.-4)` — huyệt thứ 4 của Kinh Đại Trường — trong khi Instruct chỉ viết "Hợp Cốc").

2. **Tất cả 4 biến thể SFT leak token prompt** (`<tool_call>`, `<think>`) — artifact của dataset chat-template có chứa reasoning tokens. CPT-SFT (multiplechoice) **5/5 câu đều có `<tool_call>` prefix**, dù deviation MCQ = 0%. MCQ benchmark không bắt được lỗi này vì chỉ check ký tự A/B/C/D cuối cùng.

3. **CPT-SFT (2topics)** — dù đứng thứ 2 trên Bench nhỏ MCQ (62.86%) — có **2/5 câu generate rỗng** trong inference mở. Response trung bình chỉ 285 chars, ngắn nhất mọi model. Dataset cũ có thể đã overfit → degenerate generation.

4. **Hallucination domain-specific vẫn tồn tại ở mọi model**:
   - *Vị trí huyệt Hợp Cốc* (Q1): Instruct nói "đầu ngón tay trỏ" (sai), CPT-only nói "giữa ngón út và ngón giữa" (sai khác kiểu), CPT-SFT-2topics gần đúng nhất ("khớp gian đốt giữa ngón tay trỏ và ngón giữa"), SFT-2topics nói "đầu ngón chân cái" (sai nghiêm trọng vì Hợp Cốc nằm ở tay).
   - *Bài Kiện vận tỳ vị* (Q2): CPT-only **hallucinate liệt kê "Bạch Truật, Nhân Sâm, Xa tiền tử, Bạch Truật, Bạch Truật"** — lặp Bạch Truật 3 lần, lỗi rõ.

### Hệ quả tuyển model

| Use case | Chọn |
|---|---|
| Sinh câu trả lời y học tự do (chatbot, explain) | **CPT-only** (`cpt-tm-cpt-1776775114`) — format sạch, domain signal, không leak |
| Pipeline Q&A narrow chỉ cần chọn A/B/C/D | **CPT-SFT (multiplechoice)** (`tm-sft-1776945292`) — 0% deviation, dù có `<tool_call>` nếu không strip sẽ mess cho downstream parser |
| Avoid | **CPT-SFT (2topics)** — 2/5 câu rỗng; **SFT-only (multiplechoice)** — regression; |

**Hành động tiếp theo nên cân nhắc**: strip `<tool_call>`/`<think>` tokens khỏi dataset SFT trước khi train lại, hoặc thêm post-processing ở inference layer. Chi tiết xem `train.log` + `.md` table ở HF revision trên.

## Tổng quan

```
Raw TCM Texts (.parquet)
        │
        ▼
   Continued Pre-Training ──► LoRA adapter @ HF  (cpt-<tag>)
        │                          │
        │                          └── eval/outputs/CPT_New, CPT_VM14K  + train.log
        ▼
   Supervised Fine-Tuning ──► LoRA adapter @ HF  (sft-<cfg>-<tag> hoặc cpt-sft-<cfg>-<tag>)
        │                          │
        │                          └── eval/outputs/{SFT_*|CPT_SFT_*}  + benchmark_evaluations/  + train.log
        ▼
   Inference / Serving
```

**Model hỗ trợ:** Qwen3-0.6B / 1.7B / 4B Instruct (qua Unsloth, 4-bit).

## Cấu trúc dự án

```
├── main.py                         # Entry point SFT training
├── inference.py                    # So sánh output giữa các model variants
├── CPT/
│   └── CPT.py                      # CPT: train → push HF → auto-eval (load từ HF) → push eval
├── EVAL/
│   ├── Data/                       # Bộ đề MCQ (tcm_mcq, eastern_medicine_VM14K, MedQA)
│   ├── Unsloth_QA_*.ipynb          # Notebook đánh giá định tính (tự chạy sau train)
│   └── outputs/                    # (gitignored) kết quả eval mỗi run — được push lên HF branch
├── src/
│   ├── commands/                   # TrainCommand — orchestrator SFT pipeline
│   ├── core/                       # Config, Logger, Tracker (WandB)
│   ├── data/                       # DatasetManager, strategies
│   ├── training/                   # MedTrainer (SFTTrainer), LoRA config, arg builder
│   └── evaluation/                 # Benchmarker (lm-eval MMLU medical)
├── configs/
│   ├── experiments/                # YAML config từng experiment
│   ├── dataset/                    # Config dataset
│   └── evals/                      # Config benchmark
├── scripts/
│   ├── lambda_launch.py            # Local: orchestrate Lambda Cloud runs
│   ├── lambda_bootstrap.sh         # Host (Lambda instance): pull image + docker run
│   ├── lambda_push_to_hf.py        # Container: upload adapter + eval + log lên HF
│   ├── LAMBDA_README.md            # Chi tiết workflow Lambda Cloud
│   ├── multi_experiments*.sh       # Batch local sweeps
│   └── launch_sweep.sh
├── server/                         # FastAPI inference server
├── Dockerfile                      # Pre-baked deps image (CUDA 12.8 + Python 3.10 + requirements)
├── .dockerignore
├── .github/workflows/docker.yml    # CI: auto-build & push ghcr.io/venera-ai/traditional-medical-model
└── requirements.txt
```

## Cài đặt (local)

```bash
# System deps
sudo apt update
sudo apt install python3.10-dev build-essential cmake libffi-dev

# Python deps
pip install -r requirements.txt
```

`.env` (xem [.env.example](.env.example)):
```dotenv
ROOT=/path/to/Traditional-Medical-Model
WANDB_KEY=your_wandb_api_key
HF_TOKEN=hf_...

# chỉ cần khi dùng Lambda Cloud:
LAMBDA_API_KEY=secret_...
LAMBDA_SSH_KEY_NAME=<tên key đã add trong Lambda dashboard>
LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_rsa
GH_TOKEN=ghp_...                    # classic PAT với scope repo, workflow, read:packages
GHCR_LOGIN=1                        # nếu image private (mặc định)
GHCR_USER=<github-username>
HF_CPT_REPO=huyennkai/Venera-CPT-TMModel
HF_SFT_REPO=huyennkai/Venera-Traditional-Model
```

## Chạy trên Lambda Cloud (khuyến nghị)

Flow 1 dòng — launch, train, eval, push, auto-terminate:

```bash
# CPT
python scripts/lambda_launch.py --task cpt --git-branch main

# SFT 1 config
python scripts/lambda_launch.py --task sft --git-branch main \
    --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml

# SFT batch (nhiều config trên cùng instance, tiết kiệm tiền)
python scripts/lambda_launch.py --task sft --git-branch main \
    --config-list scripts/lambda_batch.txt
```

Chi tiết về cơ chế, failure modes, cost guardrails: [scripts/LAMBDA_README.md](scripts/LAMBDA_README.md).

Sau mỗi lần chạy:
- **Adapter** được push lên branch `cpt-<tag>` hoặc `sft-<cfg>-<tag>` / `cpt-sft-<cfg>-<tag>` của HF repo.
- **Eval notebook load adapter đó từ HF** (không phải local disk) — đảm bảo kết quả đúng model trong revision.
- **Eval outputs + `benchmark_evaluations` + `train.log`** được push cùng revision (folder `eval/` + `train.log` ở root).

## Chạy local (không dùng Lambda)

### 1. CPT

```bash
cd CPT
python CPT.py
```

Output: `cpt_qwen3_medical/` (LoRA adapter). Nếu env var `HF_CPT_REPO` và `JOB_TAG` được set, script cũng push lên HF và eval notebook load từ HF; không set thì eval đọc adapter từ local path.

### 2. SFT

```bash
# Dry-run
python main.py --test

# Single
python main.py --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml

# Batch
./scripts/multi_experiments_mini.sh        # nhẹ
./scripts/multi_experiments.sh             # full

# DeepSpeed multi-GPU
deepspeed --num_gpus=2 main.py \
    --config configs/experiments/unsloth-qwen3-4B-full/exp0001/config.yaml \
    --deepspeed

# Override CLI
python main.py --config .../config.yaml --training.batch_size=8
```

### 3. Inference — so sánh Gốc / CPT / CPT+SFT

```bash
python inference.py
```

### 4. Inference server

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /` — danh sách model
- `GET /model/info/{slug}`
- `POST /model/predict` — params `max_new_tokens`, `temperature`, `top_p`, `top_k`

### 5. Evaluation

- **Quantitative:** `Benchmarker` chạy tự động sau SFT → JSON ở `group_outputs/outputs/{SFT|CPT_SFT}/benchmark_evaluations/`.
- **Qualitative:** `EVAL/Unsloth_QA_*.ipynb` chạy tự động sau CPT hoặc SFT — phân tích theo topic, độ khó, deviation rate. Output ở `EVAL/outputs/{CPT_*|SFT_*|CPT_SFT_*}/`.

Trên Lambda, cả 2 loại output được đóng gói cùng adapter trên HF revision.

## Docker image

Image deps pre-built bởi CI:
```
ghcr.io/venera-ai/traditional-medical-model:<branch>   # auto-tagged theo nhánh git
ghcr.io/venera-ai/traditional-medical-model:sha-<7>    # pin theo commit
```
Rebuild tự động khi `Dockerfile` hoặc `requirements.txt` thay đổi (Actions tab trên GitHub). Lambda instance pull image thay vì `pip install`, cắt ~15 phút mỗi launch.

## Công nghệ

| Thành phần | Thư viện |
|---|---|
| Fine-tuning | Unsloth, TRL (SFTTrainer), PEFT (LoRA) |
| Base model | HuggingFace Transformers |
| Distributed | DeepSpeed (ZeRO 2/3) |
| Tracking | Weights & Biases |
| Benchmark | lm-eval (EleutherAI) |
| Serving | FastAPI, Uvicorn, Ngrok |
| Config | Pydantic, PyYAML |
| Infra | Lambda Cloud API, Docker, GitHub Actions, ghcr.io, HF Hub |

## Dataset

- **CPT:** dữ liệu y học cổ truyền dạng raw text (`.parquet`)
- **SFT:** [Maverfrick/TradMedInstructData](https://huggingface.co/datasets/Maverfrick/TradMedInstructData)
- **Eval:** VM14K Eastern Medicine, TCM MCQ, MedQA

## Lưu ý

- `.sh` có CRLF (tạo trên Windows) → chạy `dos2unix` trước.
- Trước khi train với W&B tracker, chạy `wandb login` (hoặc set `WANDB_API_KEY` env var).
- Đừng commit `.env` hay `EVAL/outputs/` (đã có trong `.gitignore`).
- Rotate `HF_TOKEN` / `GH_TOKEN` định kỳ; script chỉ cần quyền `read:packages` (để pull ghcr image) + `Contents: R/W` trên repo.
