# Báo cáo các fix — 2026-04-21 → 2026-04-22

Tổng hợp gọn các thay đổi và hiệu quả đo được. Chi tiết: [REPORT.md §7-§8](REPORT.md).

---

## 1. Fix đã sửa (theo thứ tự ảnh hưởng)

| # | Vấn đề | Fix | File ảnh hưởng |
|---|---|---|---|
| 1 | CPT cũ hy sinh đa chuyên khoa: `warmup_steps=200 > max_steps=40` → LR chưa lên đỉnh đã dừng. Không packing → phí compute. Không eval → không biết học được gì. | `warmup_ratio=0.03`, `num_train_epochs=2`, `packing=True`, eval trên `Maverfrick/Continued-Pretraining-Vietnamese-TCM@test`. LR 5e-5 → 3e-5 (giữ alignment Instruct). Không LoRA embedding (dataset quá nhỏ, rủi ro collapse). | [CPT/CPT.py](CPT/CPT.py) |
| 2 | CPT-SFT cũ có deviation 11-12% (format collapse). LR 2e-5 quá thấp → không reset được chat format sau raw-text CPT. | Epochs 2→3, LR 2e-5→5e-5. Giữ `load_best_model_at_end`. | [config_cpt.yaml](configs/experiments/unsloth-qwen3-4B/exp0001/config_cpt.yaml) |
| 3 | Eval notebook load adapter từ local disk → nghi ngờ kết quả là model cũ. | HF-round-trip: train → push adapter lên HF revision → eval notebook load từ HF `revision=` kwarg → push eval outputs cùng revision với `--skip-model`. Mỗi revision là snapshot tự chứa. | [CPT.py](CPT/CPT.py), [lambda_bootstrap.sh](scripts/lambda_bootstrap.sh), [lambda_push_to_hf.py](scripts/lambda_push_to_hf.py), [4 EVAL notebooks](EVAL/) |
| 4 | Fork-after-CUDA: CPT.py train xong spawn subprocess eval → NVML corrupt → `torch.cuda.is_available()=False`. | Mỗi notebook chạy trong **docker run riêng** (fresh process tree, fresh CUDA init). +5-10s overhead × N notebook, bù lại không bao giờ đụng fork bug. | [lambda_bootstrap.sh](scripts/lambda_bootstrap.sh) (`run_cpt`, `run_sft`) |
| 5 | Lambda launch cài `pip install` 15-20 phút/lần, thỉnh thoảng OOM resolver. | Dockerize: CUDA 12.8 + Python 3.10 + `requirements.txt` bake sẵn. CI tự build + push `ghcr.io/venera-ai/traditional-medical-model:<branch>` khi `Dockerfile`/`requirements.txt` đổi. Lambda chỉ `docker pull` ~3-5 phút. | [Dockerfile](Dockerfile), [.github/workflows/docker.yml](.github/workflows/docker.yml), [lambda_bootstrap.sh](scripts/lambda_bootstrap.sh) |
| 6 | `EVAL/outputs/` tracked trong git → Lambda clone kéo theo kết quả run cũ → push đè lên HF. | `git rm --cached -r EVAL/outputs/` (122 file), thêm vào `.gitignore`. CPT.py wipe toàn bộ `EVAL/outputs/*` trước eval. | [.gitignore](.gitignore), [CPT/CPT.py](CPT/CPT.py) |
| 7 | Image mới thiếu `jupyter-nbconvert`; notebook eval fail với exit 1. | Thêm `nbconvert==7.16.4` + `nbformat==5.10.4` vào `requirements.txt`. | [requirements.txt](requirements.txt) |
| 8 | Notebook kernel không thấy GPU dù `docker run --gpus all`. | Convert `.ipynb` → `.py` rồi chạy `python` trực tiếp (bỏ kernel jupyter). | [CPT.py](CPT/CPT.py), patcher scripts |
| 9 | `lambda_launch.py` crash khi `tail -c` cắt giữa ký tự UTF-8 (tiếng Việt) → kéo theo terminate instance giữa chừng eval. | Decode với `errors='replace'`. | [lambda_launch.py](scripts/lambda_launch.py) |
| 10 | SSH flaky sau Lambda boot (cloud-init regen host keys) → scp fail → fail-terminate. | `wait_for_ssh` yêu cầu 3 echo liên tiếp; `scp_to` retry 4 lần. | [lambda_launch.py](scripts/lambda_launch.py) |
| 11 | Config không hỗ trợ HF `revision` → không fix được model tại commit cụ thể. | Thêm `model.revision` → pass xuống `FastLanguageModel.from_pretrained`. | [src/commands/train_command.py](src/commands/train_command.py) |

---

## 2. Hiệu quả đo được (HF eval, 22/04/2026)

### CPT-only — hết trade-off, vượt Instruct cả 2 benchmark

| | VM14K (564 câu) | Bench nhỏ (105 câu) | Deviation |
|---|:--:|:--:|:--:|
| Instruct (baseline) | 48.05% | 53.33% | 0% / 0% |
| CPT cũ (16/04) | 43.44% | 55.24% | 0% / 0.95% |
| **CPT mới (22/04)** | **49.11%** | **58.10%** | **0% / 0%** |
| **Δ vs CPT cũ** | **+5.67%** 📈 | **+2.86%** 📈 | sạch hơn |
| Δ vs Instruct | +1.06% | +4.77% | = |

→ CPT cũ "học TCM thì quên cardiology/neurology". CPT mới giữ được kiến thức cũ **và** gain thêm domain.

### CPT→SFT — sửa format collapse, tất cả metric đều cải thiện

| | VM14K | Bench nhỏ | Deviation |
|---|:--:|:--:|:--:|
| CPT-SFT cũ (16/04) | 39.72% | 51.43% | 🚨 11.35% / 12.38% |
| **CPT-SFT mới (22/04)** | **47.16%** | **62.86%** | **1.06% / 0%** |
| **Δ** | **+7.44%** 📈 | **+11.43%** 📈 | **~10× giảm** 📈 |

### Infra

| | Trước | Sau |
|---|---|---|
| Lambda launch deps | 15-20 phút `pip install`, thỉnh thoảng OOM | 3-5 phút `docker pull`, reproducible theo sha |
| Eval correctness | Load local adapter (nghi ngờ nhầm model cũ) | HF revision, self-contained |
| Notebook eval trên Lambda | Fail (no GPU / no nbconvert / fork-after-CUDA) | Chạy gọn, mỗi notebook trong container riêng |

---

## 3. Model khuyến nghị ship

| Use case | Model | VM14K | Bench | Dev |
|---|---|:--:|:--:|:--:|
| TCM thuần (narrow Q&A Đông y) | SFT-only `tm-sft-1776771092` | 46.45% | 🥇 63.81% | 0% |
| Cân bằng đa chuyên khoa + Đông y | CPT-SFT `tm-sft-1776784126` | 47.16% | 🥈 62.86% | 1.06% |
| Generation tự do (không cần Q&A format) | CPT-only `cpt-tm-cpt-1776775114` | 🥇 49.11% | 58.10% | 0% |

Tất cả đều public trên HF, `revision=<tên branch>`, hash reproducible qua ghcr image tag.
