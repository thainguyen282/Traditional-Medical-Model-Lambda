# VeneraMedicalM1 - Fine-tuning LLM cho Y Học Cổ Truyền

Dự án nghiên cứu fine-tuning Large Language Model chuyên biệt cho lĩnh vực **Y Học Cổ Truyền** (Đông Y), sử dụng pipeline hai giai đoạn: **Continued Pre-Training (CPT)** và **Supervised Fine-Tuning (SFT)**.

## Tổng quan

```
Raw TCM Texts (.parquet)
        │
        ▼
   Continued Pre-Training ──► Merged Model
        │
        ▼
   Supervised Fine-Tuning ──► LoRA Adapter + Benchmark
        │
        ▼
   Inference / Evaluation
```

**Model hỗ trợ:** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B (Instruct, qua Unsloth 4-bit)

## Cấu trúc dự án

```
├── main.py                     # Entry point cho SFT training
├── inference.py                # So sánh output giữa các model variants
├── CPT/                        # Continued Pre-Training
│   ├── CPT.py                  # Script train CPT trên dữ liệu y học cổ truyền
│   └── *.parquet               # Dữ liệu thô
├── EVAL/                       # Evaluation
│   ├── Data/                   # Bộ đề MCQ (eastern_medicine_VM14K, tcm_mcq)
│   ├── Unsloth_QA_*.ipynb      # Notebook đánh giá chất lượng
│   └── outputs/                # Kết quả eval các model variants
├── src/
│   ├── commands/               # TrainCommand - orchestrator pipeline training
│   ├── core/                   # Config, Logger, Tracker (WandB)
│   ├── data/                   # DatasetManager, strategies (VeneraMedM1, FineTome)
│   ├── training/               # MedTrainer (SFTTrainer), LoRA config, argument builder
│   └── evaluation/             # Benchmarker (lm-eval trên MMLU medical tasks)
├── configs/
│   ├── experiments/            # YAML config cho từng experiment
│   ├── dataset/                # Config dataset (venera_med_m1, finetome)
│   └── evals/                  # Config benchmark (medical MMLU)
├── scripts/                    # Shell scripts chạy multi-experiments, sweep
├── server/                     # FastAPI inference server + ngrok
├── TMdata/                     # Dữ liệu SFT đã xử lý
└── requirements.txt
```

## Cài đặt

```bash
# System dependencies
sudo apt update
sudo apt install python3.11-dev build-essential cmake libffi-dev

# Python dependencies
pip install -r requirements.txt
```

Cấu hình file `.env` theo mẫu `.env.example`:
```
ROOT=/path/to/Traditional-Medical-Model
WANDB_KEY=your_wandb_api_key
```

## Sử dụng

### 1. Continued Pre-Training (CPT)

Train model trên dữ liệu y học cổ truyền dạng raw text:

```bash
cd CPT
python CPT.py
```

Output: LoRA adapter (`cpt_qwen3_medical/`) và merged model (`cpt_qwen3_medical_merged/`).

### 2. Supervised Fine-Tuning (SFT)

**Chạy thử nghiệm nhanh (dry run):**
```bash
python main.py --test
```

**Chạy single experiment:**
```bash
python main.py --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml
```

**Chạy nhiều experiments:**
```bash
# Mini (nhẹ)
./scripts/multi_experiments_mini.sh

# Full scale
./scripts/multi_experiments.sh
```

**Chạy với DeepSpeed (multi-GPU):**
```bash
deepspeed --num_gpus=2 \
    main.py --config configs/experiments/unsloth-qwen3-4B-full/exp0001/config.yaml \
    --deepspeed
```

**Override config từ CLI:**
```bash
python main.py --config configs/experiments/.../config.yaml --training.batch_size=8
```

### 3. Inference - So sánh model

So sánh output của 3 variants (Gốc / CPT / CPT+SFT) trên cùng bộ câu hỏi:

```bash
python inference.py
```

### 4. Inference Server

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /` - Danh sách model
- `GET /model/info/{model_slug}` - Thông tin model
- `POST /model/predict` - Sinh câu trả lời (hỗ trợ `max_new_tokens`, `temperature`, `top_p`, `top_k`)

### 5. Evaluation

Benchmark tự động chạy sau mỗi lần train (MMLU medical tasks). Kết quả lưu tại `group_outputs/outputs/<run_id>/benchmark_evaluations/`.

Đánh giá qualitative qua các notebook trong `EVAL/`:
- So sánh model trên bộ đề MCQ y học cổ truyền (VM14K, TCM)
- Phân tích theo topic, độ khó, deviation rate

## Công nghệ sử dụng

| Thành phần | Thư viện |
|---|---|
| Fine-tuning | Unsloth, TRL (SFTTrainer), PEFT (LoRA) |
| Base model | HuggingFace Transformers |
| Distributed training | DeepSpeed (ZeRO Stage 2/3) |
| Experiment tracking | Weights & Biases |
| Benchmark | lm-eval (EleutherAI) |
| Serving | FastAPI, Uvicorn, Ngrok |
| Config | Pydantic, PyYAML |

## Dataset

- **CPT:** Dữ liệu y học cổ truyền dạng raw text (`.parquet`)
- **SFT:** [Maverfrick/TradMedInstructData](https://huggingface.co/datasets/Maverfrick/TradMedInstructData) (HuggingFace Hub)
- **Eval:** Bộ đề MCQ Eastern Medicine từ VM14K, TCM multiple-choice questions

## Lưu ý

- File `.sh` nếu có ký tự `\r` (tạo trên Windows), dùng `dos2unix` để convert trước khi chạy.
- Chạy `wandb init` để cấu hình entity và project trước khi train với WandB tracker.
