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