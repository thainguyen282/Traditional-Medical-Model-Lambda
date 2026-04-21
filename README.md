# VeneraMedicalM1 — Fine-tuning LLM cho Y Học Cổ Truyền

Dự án nghiên cứu fine-tuning Large Language Model chuyên biệt cho lĩnh vực **Y Học Cổ Truyền** (Đông Y), dùng pipeline hai giai đoạn: **Continued Pre-Training (CPT)** → **Supervised Fine-Tuning (SFT)**.

Hỗ trợ 2 chế độ chạy:
- **Local** — train trên máy có GPU (hoặc kiểm thử dry-run).
- **Lambda Cloud (khuyến nghị)** — launch từ máy local, pre-baked Docker image tự chạy train + auto-eval + push artifact & eval outputs lên Hugging Face, tự terminate khi xong. Xem [scripts/LAMBDA_README.md](scripts/LAMBDA_README.md).

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
