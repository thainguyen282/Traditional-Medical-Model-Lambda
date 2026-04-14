# Báo cáo đánh giá mô hình Y học cổ truyền (Qwen3-4B)

**Ngày: 31/03/2026**

### Nhận xét chung 3 model

- **Qwen3-4B-Gốc**
  - Accuracy:
    - 53.33% (benchmark nhỏ)
    - **48.05% (VM14K – cao nhất)**
  - Ưu điểm:
    - Ổn định, **0% deviation**
    - Trả lời có cấu trúc rõ, dễ đọc
    - Generalize tốt trên nhiều chuyên ngành
  - Nhược điểm:
    - Độ chính xác Đông y chưa cao
    - Thiếu chiều sâu, đôi khi từ chối trả lời

---

- **Qwen3-4B-CPT**
  - Accuracy:
    - **61.90% (cao nhất benchmark nhỏ, +8.57 so với gốc)**
    - 46.10% (VM14K, -1.95 so với gốc)
  - Ưu điểm:
    - Cải thiện mạnh domain YHCT
    - Rất tốt ở câu khó (**+36.37 điểm**)
  - Nhược điểm:
    - Over-specialization (giảm trên tập đa ngành)
    - Nội dung nhiễu, sai thuật ngữ
    - Thiếu ổn định giữa các benchmark

---

- **Qwen3-4B-CPT-SFT**
  - Accuracy:
    - **42.86% (thấp nhất, -19.04 so với CPT)**
  - Deviation:
    - **24.76% (26/105 câu sai format)**
  - Ưu điểm:
    - Câu trả lời mượt, ngắn gọn
  - Nhược điểm:
    - Giảm mạnh accuracy
    - Lỗi format nghiêm trọng (tới **66.67% ở câu khó**)
    - Nội dung nông, dễ sai nhưng “trông đúng”

---

### Kết luận nhanh

- **Tốt nhất tổng thể:** Qwen3-4B-Gốc  
- **Tốt nhất domain YHCT:** Qwen3-4B-CPT (nhưng thiếu ổn định)  
- **Kém nhất:** Qwen3-4B-CPT-SFT  

---

## Cập nhật 13/04/2026

### Thay đổi dữ liệu

- **SFT data:** Chuyển sang dataset `TradMedInstruct-2topics-3546` (từ `TMdata/proccessed/dong_y_conversation.jsonl`)
- **CPT data:** Giữ nguyên `train-00000-of-00001.parquet`, tăng training từ 200 → 2000 max_steps, warmup 20 → 200

### Thay đổi pipeline

- **Auto-eval sau SFT:** Sau khi SFT train xong, tự động chạy eval notebooks:
  - Nếu train từ CPT → chạy `Unsloth_QA_CPT_SFT_new` + `Unsloth_QA_CPT_SFT_VM14K`
  - Nếu train từ base model → chạy `Unsloth_QA_SFT_new` + `Unsloth_QA_SFT_VM14K`
- **Auto-eval sau CPT:** Giữ nguyên auto-eval `Unsloth_QA_CPT_new` + `Unsloth_QA_CPT_VM14K`, thêm progress log
- **Dynamic model path:** Eval notebooks nhận model path từ env var `SFT_MODEL_PATH` thay vì hardcode run_id, đảm bảo eval đúng model vừa train
- **Fix adapter path:** Tự động chuyển `base_model_name_or_path` trong adapter_config.json sang absolute path để eval từ bất kỳ directory nào cũng tìm được base model

---

## Cập nhật 14/04/2026 — Chẩn đoán & sửa pipeline CPT → SFT

### Eval mới (VM14K, 564 câu)

| Model | Accuracy | Deviation | Ghi chú |
|-------|----------|-----------|---------|
| **Qwen3-4B-Gốc** | 48.05% (271/564) | 0.00% (0/564) | Baseline, format ổn định |
| **Qwen3-4B-SFT** (từ base) | 43.97% (248/564) | 1.06% (6/564) | SFT-only, format gần như giữ nguyên |
| **Qwen3-4B-CPT** | 45.92% (259/564) | 0.18% (1/564) | Domain YHCT tốt, format tốt |
| **Qwen3-4B-CPT-SFT** | **13.83% (78/564)** | **69.86% (394/564)** | **Sụp đổ format + mất kiến thức** |

### Chẩn đoán nguyên nhân CPT → SFT sụp đổ

1. **Learning rate quá cao cho model đã pre-adapted**
   - CPT: `lr=5e-5`. SFT: `lr=1e-4` (gấp 2×) → ghi đè weights CPT.
2. **LoRA alpha bất đối xứng**
   - CPT: `lora_alpha=256`, `r=32` → effective scale 8×.
   - SFT: `lora_alpha=32`, `r=32` → effective scale 1×.
   - Khi merge CPT rồi stack LoRA mới ở scale khác → bất ổn định, mất kiến thức.
3. **Quantization mismatch**
   - CPT: full precision (`load_in_4bit=false`).
   - SFT: 4bit (`load_in_4bit=true`) → merged CPT weights bị quantize về 4bit → mất thông tin ngay trước khi SFT.
4. **Format collapse là triệu chứng chính**: 69.86% output không phải A/B/C/D → SFT đã xoá cả năng lực tuân theo chat template của model.

### Thay đổi đã áp dụng (Option A: continue-train CPT adapter)

**1. [CPT/CPT.py](CPT/CPT.py)**
- Xoá bước `merge_and_unload()` và thư mục `cpt_qwen3_medical_merged`.
- CPT chỉ lưu LoRA adapter tại `CPT/cpt_qwen3_medical` để SFT continue-train trực tiếp trên adapter đó.

**2. [src/commands/train_command.py](src/commands/train_command.py#L287) — `_load_training_model`**
- Detect khi `model.name` là 1 LoRA adapter path: `hasattr(model, "peft_config")`.
- Nếu đã là PeftModel → **không** gọi `get_peft_model()` lần nữa (tránh stack LoRA-trên-LoRA), gọi `FastLanguageModel.for_training(model)` để bật training mode.
- Nếu là base model + có `lora_config` → tạo adapter mới (nhánh SFT-only giữ nguyên).

**3. [configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml](configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml)**

| Param | Trước | Sau | Lý do |
|-------|-------|-----|-------|
| `model.name` | `CPT/cpt_qwen3_medical_merged` | `CPT/cpt_qwen3_medical` | Trỏ thẳng tới adapter, bỏ merge |
| `unsloth.load_in_4bit` | `true` | `false` | Đồng nhất precision với CPT |
| `training.learning_rate` | `1e-4` | `2e-5` | Giảm 5× — adapter đã pre-adapted |
| `training.warmup_steps` | `50` | `100` | Ổn định hơn khi continue-train |
| `training.num_train_epochs` | `3` | `2` | Tránh overfit trên adapter đã học domain |

### Hạn chế còn lại của Option A

- Adapter tiếp tục dùng `lora_alpha=256` kế thừa từ CPT (không ghi đè được bằng config SFT). Nếu muốn đổi scale phải retrain CPT.
- **CPT data format vẫn nghi vấn**: [CPT.py:72-101](CPT/CPT.py#L72-L101) đang dạy `assistant = input[:300]` (lặp lại câu hỏi), có thể làm model học hành vi sai. Nếu eval CPT→SFT mới vẫn deviation cao, cần chuyển sang **Option B**: đổi CPT sang raw-text completion hoặc instruction data thực sự.

### Cần làm tiếp

1. Retrain CPT (chạy [CPT/CPT.py](CPT/CPT.py) sau khi bỏ merge) để sinh adapter mới.
2. Chạy SFT với config đã sửa.
3. Chạy eval CPT_SFT_VM14K + CPT_SFT_New, so sánh deviation & accuracy với baseline CPT-only.

---

## Cập nhật 14/04/2026 (chiều) — Kết quả sau khi áp dụng Option A

### Eval outputs mới nhất

| Model | VM14K (564 câu) | Deviation | benchmark nhỏ (105 câu) | Deviation |
|-------|-----------------|-----------|-------------------------|-----------|
| **Qwen3-4B-Gốc** | 48.05% | 0.00% | 53.33% | 0.00% |
| **Qwen3-4B-SFT** | 43.97% | 1.06% | — | — |
| **Qwen3-4B-CPT (mới)** | **45.04%** (254/564) | 0.00% (0/564) | 53.33% | 0.00% |
| **Qwen3-4B-CPT-SFT (mới)** | **38.83%** (219/564) | 16.67% (94/564) | 40.00% | 24.76% |

### So sánh trước/sau khi sửa pipeline CPT → SFT

| Metric | Trước Option A | Sau Option A | Thay đổi |
|--------|----------------|---------------|----------|
| CPT-SFT accuracy (VM14K) | 13.83% | **38.83%** | **+25.00** |
| CPT-SFT deviation (VM14K) | 69.86% | **16.67%** | **−53.19** |
| CPT-SFT correct count | 78/564 | 219/564 | +141 |

### Nhận xét

- **Option A có hiệu quả rõ rệt**: accuracy tăng gần 3×, deviation giảm hơn 4×. Chứng minh giả thuyết "merge + 4bit quantize + lr cao + alpha mismatch" là nguyên nhân chính gây sụp đổ trước đây.
- **Nhưng vẫn chưa đạt yêu cầu**:
  - CPT-SFT (38.83%) **vẫn thấp hơn CPT-only (45.04%)** — SFT vẫn làm mất kiến thức domain.
  - CPT-SFT (38.83%) **vẫn thấp hơn SFT-only (43.97%)** — pipeline 2-stage chưa cộng được gì so với SFT đơn lẻ.
  - Deviation 16.67% vẫn cao hơn hẳn CPT-only (0%) và SFT-only (1.06%) → chat template vẫn bị nhiễu một phần.

### Nghi vấn còn lại & bước tiếp

1. **CPT data format vẫn là gốc rễ deviation**: [CPT.py:72-101](CPT/CPT.py#L72-L101) dạy `assistant = input[:300]`. Continue-train adapter vẫn kế thừa hành vi này → SFT data (A/B/C/D) phải "kéo ngược" adapter về đúng format, không đủ lực trong 2 epoch.
2. **Cần Option B**: đổi CPT sang một trong hai hướng:
   - **Raw-text completion**: không dùng chat template, train next-token prediction trên text thuần.
   - **Instruction data thực**: thay vì `assistant = input[:300]`, dùng cặp Q&A thật sự (có thể generate bằng LLM khác).
3. **Thử nghiệm phụ**: tăng `num_train_epochs` của SFT lên 3-4 để adapter có thời gian "học lại" format A/B/C/D.
4. **Kiểm tra**: xem vài câu deviation trong [EVAL/outputs/CPT_SFT_VM14K/](EVAL/outputs/CPT_SFT_VM14K/) để xác định loại lỗi (copy câu hỏi? trả lời tự do? xuống dòng sai?).