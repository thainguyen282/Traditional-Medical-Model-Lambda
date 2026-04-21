# Báo cáo đánh giá mô hình Y học cổ truyền (Qwen3-4B)

**Cập nhật cuối: 21/04/2026 — infra & pipeline overhaul (Dockerize, Lambda Cloud, HF-round-trip eval, CPT hyperparam fix)**

---

## 1. Tổng quan

Dự án so sánh 4 cấu hình huấn luyện trên base model `unsloth/Qwen3-4B-Instruct-2507`:

| Cấu hình | Mô tả |
|----------|-------|
| **Instruct** | Model gốc, không fine-tune |
| **SFT-only** | Fine-tune trực tiếp trên dataset Q&A YHCT (HF `TradMedInstruct-2topics-3546`) |
| **CPT-only** | Continual pre-training trên text Đông y thuần (1549 đoạn, 2.48 MB) |
| **CPT → SFT** | CPT trước, sau đó SFT continue-train trên cùng adapter |

Đánh giá trên 2 benchmark:
- **VM14K_Vietnamese** (564 câu MCQ, đa chuyên khoa, focus Đông y)
- **Benchmark nhỏ** (105 câu MCQ, chủ yếu Đông y)

---

## 2. Kết quả tổng hợp

### 2.1. Bảng tổng — 4 model × 2 benchmark (master table)

| **Model** | **VM14K Acc**<br>(564 câu, đa chuyên khoa) | **VM14K Dev** | **Bench nhỏ Acc**<br>(105 câu, chủ yếu Đông y) | **Bench nhỏ Dev** | **Δ Acc vs Instruct**<br>(VM14K / Bench) |
|-----------|:--:|:--:|:--:|:--:|:--:|
| **Instruct** (gốc) | 🥇 **48.05%** (271/564) | ✅ 0.00% (0) | 🥈 53.33% (56/105) | ✅ 0.00% (0) | — / — |
| **SFT-only** | 🥈 43.97% (248/564) | 1.06% (6) | *(chưa eval)* | — | −4.08 / — |
| **CPT-only** | 🥉 43.44% (245/564) | ✅ 0.00% (0) | 🥇 **55.24%** (58/105) | 0.95% (1) | −4.61 / **+1.91** ✅ |
| **CPT → SFT** | ❌ 39.72% (224/564) | ⚠️ 11.35% (64) | 🥉 51.43% (54/105) | ⚠️ 12.38% (13) | −8.33 / −1.90 |

**Đọc nhanh**:
- Trên đa chuyên khoa (VM14K): **Instruct vô địch**, CPT-SFT cuối bảng.
- Trên domain Đông y (bench nhỏ): **CPT-only vô địch** — đã vượt baseline Instruct.
- CPT-only là model duy nhất **vừa giữ format sạch (deviation ~0%) vừa vượt Instruct trên Đông y**.
- CPT-SFT là model duy nhất có deviation > 10% ở cả 2 benchmark → chưa đạt production.

### 2.1b. Bảng so sánh chi tiết theo difficulty (cross-benchmark)

| **Model** | **VM14K Easy**<br>(112) | **VM14K Mod**<br>(339) | **VM14K Hard**<br>(39) | **VM14K Chal**<br>(74) | **Bench Easy**<br>(19) | **Bench Mod**<br>(75) | **Bench Hard**<br>(11) |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Instruct | 39.29% | 🥇 **50.15%** | 35.90% | 🥇 **58.11%** | 🥇 **63.16%** | 52.00% | 🥇 **45.45%** |
| SFT-only | 🥈 45.54% | 43.36% | 35.90% | 48.65% | — | — | — |
| CPT-only | 44.64% | 41.30% | 🥇 **38.46%** | 🥈 54.05% | 🥇 **63.16%** | 🥇 **54.67%** | 🥇 **45.45%** |
| CPT → SFT | 🥇 **46.43%** | 38.64% | 25.64% | 41.89% | 52.63% | 52.00% | 🥇 **45.45%** |

### 2.2. Deviation chi tiết (% câu trả lời sai format)

| Model | VM14K Easy | Mod | Hard | Chal | **VM14K total** | Bench Easy | Mod | Hard | **Bench total** |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Instruct | 0% | 0% | 0% | 0% | **0.00%** | 0% | 0% | 0% | **0.00%** |
| SFT-only | 1.79% | 0.59% | 5.13% | 0% | **1.06%** | — | — | — | — |
| CPT-only | 0% | 0% | 0% | 0% | **0.00%** | 0% | 0% | 9.09% | **0.95%** |
| CPT → SFT | 4.46% | 10.32% | 38.46% | 12.16% | **⚠️ 11.35%** | 5.26% | 10.67% | 36.36% | **⚠️ 12.38%** |

**Pattern quan trọng**: deviation của CPT-SFT tăng theo độ khó — Hard tier có ~36-38% câu sai format. Cho thấy khi gặp câu phức tạp, model "rớt" khỏi format A/B/C/D và trả lời tự do.

### 2.3. Δ so với baseline Instruct (VM14K)

| Model | Δ Accuracy | Δ Deviation | Verdict |
|-------|:--:|:--:|---|
| SFT-only | −4.08 | +1.06 | Tệ hơn baseline |
| CPT-only | −4.61 | 0.00 | Tệ hơn nhẹ trên VM14K, nhưng 🥇 trên Đông y bench |
| CPT → SFT | **−8.33** | **+11.35** | Tệ hơn rõ rệt |

### 2.4. Δ so với Tier 1 trước (Option A → Tier 1 hôm nay)

| Model | Metric | Trước (Option A) | Sau (Tier 1) | Δ |
|-------|--------|:--:|:--:|:--:|
| CPT-only | VM14K Acc | 45.04% | 43.44% | −1.60 |
| CPT-only | Bench Acc | 53.33% | **55.24%** | **+1.91** ✅ |
| CPT-only | Deviation | 0.18% | **0.00%** | **−0.18** ✅ |
| CPT-SFT | VM14K Acc | 38.83% | 39.72% | +0.89 |
| CPT-SFT | Bench Acc | 40.00% | **51.43%** | **+11.43** ✅ |
| CPT-SFT | VM14K Dev | 16.67% | **11.35%** | **−5.32** ✅ |
| CPT-SFT | Bench Dev | 24.76% | **12.38%** | **−12.38** ✅ |

**Tổng kết Tier 1**: cải thiện rõ ở mọi metric ngoại trừ CPT-only VM14K accuracy giảm nhẹ (đánh đổi over-specialize sang Đông y).

---

## 3. Phân tích từng model

### 3.1. Qwen3-4B-Instruct (gốc)

- **Accuracy**: 48.05% (VM14K) / 53.33% (bench nhỏ)
- **Deviation**: 0% — luôn trả lời đúng format A/B/C/D.
- **Điểm mạnh**: ổn định, generalize tốt trên các chuyên khoa Tây y (Otolaryngology 100%, Allergy 100%, Dermatology 78.57%).
- **Điểm yếu**: chuyên khoa Đông y chỉ 48.05%, kém hơn các chuyên khoa Tây y (Gastroenterology 67.50%, Neurology Challenging 83.33%).
- **Kết luận**: baseline tốt, là mục tiêu cần vượt qua.

### 3.2. Qwen3-4B-SFT (chỉ SFT, không CPT)

- **Accuracy**: 43.97% (VM14K) — **giảm 4.08 so với Instruct**.
- **Deviation**: 1.06% — gần như giữ nguyên format.
- **Điểm mạnh**: cải thiện ở Orthopedics (42.59% → 55.56%), Dermatology Easy giữ tốt, đặc biệt Dermatology Moderate 88.89%. Vẫn theo format chuẩn.
- **Điểm yếu**: regression ở nhiều chuyên khoa Tây y (Gastroenterology 67.50% → 42.50%, Hematology 40.62% → 37.50%, Pulmonology 48.94% giữ nhưng thấp). Đặc biệt **Eastern Medicine giảm từ 48.05% → 43.97%** — SFT data không đủ lực kéo lên.
- **Kết luận**: SFT trực tiếp làm model bị "narrow" vào style Q&A của dataset nhưng không thực sự thêm kiến thức Đông y.

### 3.3. Qwen3-4B-CPT (chỉ CPT, sau Tier 1)

- **Accuracy**: 43.44% (VM14K) / **55.24% (bench nhỏ — cao nhất cả 4 model)**.
- **Deviation**: 0% trên VM14K, 0.95% trên bench nhỏ — gần như sạch format. ✅ **Chứng minh fix raw-text completion thành công.**
- **Điểm mạnh trên Đông y**:
  - Bench nhỏ Easy: 63.16% (= Instruct).
  - Bench nhỏ Moderate: **54.67%** (vượt Instruct 52.00%).
  - VM14K Eastern Medicine Challenging: 54.05% (Instruct 58.11%, gần ngang).
- **Điểm yếu**:
  - VM14K tổng giảm 4.61 vs Instruct (43.44% vs 48.05%).
  - Tây y kém: Cardiology 22.92%, Sports Medicine 0%, Geriatrics 40%, Nephrology 29.79%.
  - Có dấu hiệu **over-specialization** sang Đông y, mất kiến thức tổng quát.
- **So với CPT cũ (trước Tier 1)**: VM14K 45.04% → 43.44% (giảm 1.6), nhưng bench nhỏ 53.33% → 55.24% (tăng 1.91). Đổi format raw-text khiến model "Đông y hơn" nhưng "Tây y kém hơn".
- **Kết luận**: CPT làm tốt vai trò của nó (specialize Đông y, giữ format), nhưng phải đánh đổi với năng lực Tây y.

### 3.4. Qwen3-4B-CPT → SFT (sau Tier 1)

- **Accuracy**: 39.72% (VM14K) / 51.43% (bench nhỏ).
- **Deviation**: 11.35% (VM14K) / 12.38% (bench nhỏ) — **vẫn cao**.
- **So với Option A cũ**: VM14K 38.83% → 39.72% (+0.89), deviation 16.67% → 11.35% (−5.32). Có cải thiện nhẹ.
- **Điểm mạnh**:
  - Đông y bench nhỏ Moderate: 52.00% (gần Instruct).
  - VM14K Easy 46.43% (cao nhất 4 model trên cùng tier khó).
  - Một số topic Tây y phục hồi: Dermatology 50%, Orthopedics 46.30%, Otolaryngology 100%.
- **Điểm yếu**:
  - Vẫn thấp hơn cả CPT-only (43.44%) và SFT-only (43.97%) trên VM14K → **pipeline 2-stage chưa cộng giá trị**.
  - Deviation tăng vọt so với CPT-only (0% → 11.35%) → SFT phá format ổn định mà CPT đã có.
  - Geriatrics sụp đổ: 40% (CPT) → 8.89% (CPT-SFT), Hard level chỉ 25.64% (CPT 38.46%).
  - Nhiều topic deviation 100% incorrect: Otolaryngology, Surgery, Cardiology Hard.
- **Kết luận**: SFT vẫn đang **phá** thành quả của CPT. Nguyên nhân nghi vấn: SFT data Q&A free-form xung đột với eval format MCQ, cộng thêm 2 epochs không đủ ổn định.

---

## 4. Nhận định chung

### So sánh xếp hạng

**Trên domain Đông y (bench nhỏ):**
1. 🥇 **CPT-only** (55.24%) — vô địch
2. 🥈 Instruct (53.33%)
3. 🥉 CPT-SFT (51.43%)

**Trên đa chuyên khoa (VM14K):**
1. 🥇 **Instruct** (48.05%) — vẫn là baseline khó vượt
2. 🥈 SFT-only (43.97%)
3. 🥉 CPT-only (43.44%)
4. ❌ CPT-SFT (39.72%)

### Điểm chính rút ra

1. **CPT có giá trị thực sự cho Đông y**: vượt baseline trên bench domain (55.24% vs 53.33%), không deviation.
2. **SFT đang phá hơn là xây**: cả SFT-only và CPT-SFT đều thua Instruct trên VM14K. Vấn đề ở **chất lượng/format SFT data**.
3. **Pipeline CPT → SFT chưa cộng giá trị**: sau Tier 1 cải thiện so với trước, nhưng vẫn dưới CPT-only.
4. **Trade-off domain vs general**: CPT càng specialize Đông y thì càng yếu Tây y. Cần cân bằng.

### Tier 1 thành công ở đâu?

| Metric | Trước Tier 1 (Option A) | Sau Tier 1 | Δ |
|--------|--------------------------|-------------|---|
| CPT-SFT VM14K Acc | 38.83% | 39.72% | +0.89 |
| CPT-SFT VM14K Dev | 16.67% | 11.35% | **−5.32** ✅ |
| CPT-only VM14K Acc | 45.04% | 43.44% | −1.60 |
| CPT-only VM14K Dev | 0.00% | 0.00% | giữ |
| CPT-only bench nhỏ Acc | 53.33% (cũ hơn nữa) | 55.24% | +1.91 ✅ |

**Tóm lại**: Tier 1 đã giảm deviation (mục tiêu chính) và cải thiện CPT trên domain, nhưng chưa giải quyết hoàn toàn vấn đề SFT phá format.

---

## 5. Lịch sử thay đổi

### 31/03/2026 — Đánh giá ban đầu

- Phát hiện CPT-SFT sụp đổ: 42.86% trên 105 câu, 24.76% deviation.

### 13/04/2026 — Pipeline overhaul

- SFT data: chuyển sang `TradMedInstruct-2topics-3546`.
- CPT: tăng max_steps 200 → 2000, warmup 20 → 200.
- Auto-eval sau training.
- Dynamic model path qua env var `SFT_MODEL_PATH`.
- Fix adapter path absolute.

### 14/04/2026 — Option A (continue-train CPT adapter)

- Bỏ merge CPT, lưu nguyên adapter.
- [src/commands/train_command.py:287](src/commands/train_command.py#L287): detect PeftModel → continue-train, không stack LoRA.
- Config SFT: `load_in_4bit: false`, `lr 1e-4 → 2e-5`, `warmup 50 → 100`, `epochs 3 → 2`.
- **Kết quả**: CPT-SFT 13.83% → 38.83% trên VM14K (+25.00), deviation 69.86% → 16.67% (−53.19). Cải thiện rõ rệt nhưng chưa đủ.

### 16/04/2026 — Tier 1 (sửa gốc rễ CPT)

**Vấn đề phát hiện**: [CPT/CPT.py:55-92 (cũ)](CPT/CPT.py) dạy `assistant = input[:300]` (copy 300 ký tự đầu của câu hỏi làm câu trả lời) → model học hành vi sai, gây deviation.

**Sửa**:

1. **[CPT/CPT.py:47-69](CPT/CPT.py#L47-L69)** — đổi sang raw-text completion:
   - Xoá `simple_explanation()` và `apply_chat_template()`.
   - `format_text()` chỉ clean text + append `eos_token`.
   - Đây mới đúng triết lý CPT (next-token prediction trên text thuần).

2. **[CPT/CPT.py:22-24](CPT/CPT.py#L22-L24)** — đồng bộ LoRA scale:
   - `lora_alpha: 256 → 32` (= r), tránh shock khi SFT continue-train.

**Kết quả** (như bảng ở trên):
- ✅ CPT-only deviation 0% trên VM14K.
- ✅ CPT-only bench nhỏ tăng lên 55.24% (cao nhất).
- ⚠️ CPT-SFT deviation giảm còn 11.35% nhưng vẫn cao hơn mục tiêu (<3%).
- ⚠️ CPT-only VM14K giảm nhẹ (45.04% → 43.44%) — đánh đổi specialize.

---

## 6. Hướng đi tiếp

### Tier 2 — Sửa SFT pipeline (ưu tiên cao)

**Vấn đề chính còn lại**: SFT data là Q&A free-form (Gemini-generated), không khớp format MCQ của eval → CPT-SFT bị phá format.

**Đề xuất**:

1. **Trộn MCQ vào SFT data**: generate 500–1000 MCQ Đông y bằng Gemini từ cùng nguồn PDF, format giống VM14K (A/B/C/D + giải thích). Trộn 50/50 với Q&A free-form hiện tại.

2. **Tăng SFT epochs**: 2 → 3-4, kèm `lr 2e-5 → 1e-5` để ổn định hơn.

3. **Generate SFT data local**: chạy [conversation_data_generate.py](TMdata/conversation_data_generate.py) để có dataset kiểm soát được (hiện đang dựa vào HF).

### Tier 3 — Tối ưu CPT (nếu Tier 2 chưa đủ)

1. **Mở rộng CPT data**: thu thập thêm PDF YHCT, target ≥10 MB text thô (hiện 2.48 MB).
2. **Trộn general data**: thêm 10–20% text Tây y/general vào CPT để tránh over-specialization.
3. **Giảm `max_steps`**: nếu data tăng nhiều, không cần 2000 steps nữa.

### KPI mục tiêu cuối

| Metric | Hiện tại | Target |
|--------|----------|--------|
| CPT-SFT VM14K Acc | 39.72% | **≥48%** (vượt Instruct) |
| CPT-SFT VM14K Dev | 11.35% | **<2%** |
| CPT-SFT bench nhỏ Acc | 51.43% | **≥58%** (vượt CPT-only) |

### Quy trình chạy hiện tại

```bash
# 1. Retrain CPT (raw-text completion)
cd CPT && python CPT.py

# 2. SFT continue-train trên CPT adapter
cd .. && python main.py --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml

# 3. Auto-eval chạy sau mỗi bước
```

Kết quả eval lưu tại [EVAL/outputs/](EVAL/outputs/).

---

## 7. Changelog — 2026-04-21: Infra + pipeline overhaul

Ngày này tập trung chuẩn hoá cách train/eval chạy, không đụng tới kết quả eval
ở các phần trên. Các thay đổi dưới đây ảnh hưởng trực tiếp tới **real run tiếp
theo**, nên cần đọc trước khi so sánh kết quả với bảng tổng ở phần 2.

### 7.1. Dockerize toàn bộ deps — bỏ `pip install` trên mỗi lần launch

- Thêm [Dockerfile](Dockerfile) (base `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` +
  Python 3.10 + toàn bộ `requirements.txt` + `nbconvert` + `nbformat`).
- [.github/workflows/docker.yml](.github/workflows/docker.yml): CI tự build + push
  `ghcr.io/venera-ai/traditional-medical-model:<branch>` khi `Dockerfile` hoặc
  `requirements.txt` thay đổi.
- Trước: mỗi Lambda launch mất 15–20 phút cài deps, thỉnh thoảng OOM pip resolver.
- Sau: `docker pull` ~3–5 phút, không OOM, reproducible theo commit sha.

### 7.2. Lambda Cloud orchestrator

- [scripts/lambda_launch.py](scripts/lambda_launch.py): 1 lệnh local → Lambda API
  provision → scp bootstrap → train → auto-terminate. Mặc định
  `gpu_1x_a100_sxm4`, fallback `gpu_1x_a10` khi hết stock.
- [scripts/lambda_bootstrap.sh](scripts/lambda_bootstrap.sh): chạy trên host
  Lambda, `docker pull` + `docker run` job. Dùng `sudo docker` (user `ubuntu`
  không ở group `docker`); scp retry + stable-SSH check (3 echo liên tiếp) để
  chịu được phase flaky ngay sau boot.
- [scripts/lambda_push_to_hf.py](scripts/lambda_push_to_hf.py): upload adapter +
  eval outputs + `train.log` lên HF branch; thêm `--skip-model`, `--print-revision`,
  `--eval-src`, `--log-file` để orchestrator compose pass.
- Chi tiết workflow: [scripts/LAMBDA_README.md](scripts/LAMBDA_README.md).

### 7.3. HF-round-trip eval (quan trọng nhất về tính đúng đắn)

**Vấn đề cũ**: notebook eval load adapter từ local disk → có lúc nhầm adapter
cũ → bảng kết quả trong [EVAL/outputs/](EVAL/outputs/) có thể không phải của
model vừa train. Trước đây kết quả CPT-SFT nghi ngờ bất thường (−8% vs Instruct)
một phần có thể do vấn đề này.

**Flow mới**:
1. Train → save local → **push adapter lên HF revision `<task>-<cfg>-<tag>`**
2. Eval notebook **load model từ HF revision đó** (qua `revision=` kwarg của
   `FastLanguageModel.from_pretrained`) — không dùng path local nữa.
3. Push `EVAL/outputs/` + `benchmark_evaluations/` + `train.log` vào cùng
   revision với `--skip-model`.

Kết quả: mỗi HF revision là **self-contained snapshot** gồm adapter + kết quả
eval tương ứng + log training. Có thể reproduce lại bằng `load_dataset`/`from_pretrained`
với `revision=<tag>`.

### 7.4. Fork-after-CUDA bug fix

**Phát hiện**: khi `CPT.py` (hoặc `main.py`) trained xong rồi spawn subprocess
để chạy notebook eval, subprocess bị `torch.cuda.is_available() = False` và
`Can't initialize NVML`. Lý do: CUDA context của parent không fork-safe, NVML
handle bị corrupted trong child.

**Fix**: bootstrap.sh giờ spawn **một docker run riêng cho mỗi notebook** (fresh
process tree, fresh CUDA init). Áp dụng cho cả CPT (`run_cpt`) và SFT
(`run_sft`). Trade-off: mỗi eval có thêm ~5–10s overhead khởi container, đổi
lại không bao giờ đụng fork issue.

### 7.5. SFT config giờ hỗ trợ `model.revision`

- [src/commands/train_command.py](src/commands/train_command.py):
  `_load_training_model`/`_load_baseline_model` giờ đọc `config.model.revision`
  và pass xuống `FastLanguageModel.from_pretrained`.
- [configs/experiments/unsloth-qwen3-4B/exp0001/config_cpt.yaml](configs/experiments/unsloth-qwen3-4B/exp0001/config_cpt.yaml):
  đổi từ local path `CPT/cpt_qwen3_medical` → HF `huyennkai/Venera-CPT-TMModel`
  @ revision cố định. Fresh Lambda clone không còn phụ thuộc file local.

### 7.6. CPT hyperparam — phát hiện và sửa footguns

Review lại [CPT/CPT.py](CPT/CPT.py):

| Vấn đề | Trước | Sau |
|---|---|---|
| Warmup > train | `warmup_steps=200`, `max_steps=40` → LR chưa lên tới peak | `warmup_ratio=0.03`, `num_train_epochs=2` |
| LR quá cao cho Instruct | 5e-5 | 3e-5 |
| Không packing → phí compute padding | — | `packing=True` |
| Không eval, không biết đang học gì | — | eval trên split `test` của `Maverfrick/Continued-Pretraining-Vietnamese-TCM`, `eval_steps=200` |
| Logging mỗi step | 1 | 10 |
| Save mỗi 40 step | 40 | 500 (cuối epoch) |
| Embedding training | cân nhắc bật | **tắt** (dataset 1.5K rows × 300 tok quá nhỏ để shift 150K×4096 embedding; đang CPT trên Instruct nên rủi ro collapse cao). Giữ `embedding_learning_rate=3e-6` inert để flip 2 dòng là bật lại. |

**Dataset mixing**: KHÔNG mix chat data ở run này (user quyết định). Theo dõi
eval_loss + thử SFT-continue sau CPT; nếu format collapse nặng mới revisit.

### 7.7. Tạp vụ

- `EVAL/outputs/` được untrack khỏi git (122 file từ run cũ commit nhầm gây
  Lambda push eval output cũ đè nhau). Thêm vào `.gitignore`.
- Thêm `nbconvert==7.16.4` + `nbformat==5.10.4` vào `requirements.txt` (trước đó
  image không có → notebook eval fail).
- [scripts/_patch_cpt_notebooks.py](scripts/_patch_cpt_notebooks.py) +
  [scripts/_patch_sft_notebooks.py](scripts/_patch_sft_notebooks.py): idempotent
  patchers cho cell `model_path` và `FastLanguageModel.from_pretrained` — đọc env
  `{CPT,SFT}_HF_REPO/REVISION` rồi mới fallback path cũ.
- README.md viết lại để phản ánh flow Lambda + HF-round-trip.

### 7.8. Việc tiếp theo cần làm trước real run

1. Launch CPT real (num_train_epochs=2, packing, eval trên HF) — xem `eval_loss`
   có giảm không, có dấu hiệu overfit/collapse không.
2. Kick SFT từ CPT revision đó (`config_cpt.yaml` → chọn revision mới push) →
   compare với CPT-SFT cũ (39.72% VM14K, 11.35% deviation). Target: deviation
   <2%, Acc ≥48%.
3. Nếu SFT vẫn collapse → mix 10% chat data (đã có sẵn `mlabonne/FineTome-100k`
   dùng cho SFT regular) và re-CPT.
