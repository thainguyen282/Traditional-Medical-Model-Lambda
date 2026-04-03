# Báo cáo đánh giá mô hình Y học cổ truyền (Qwen3-4B)

**Ngày: 31/03/2026**

### 🔎 Nhận xét chung 3 model

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

### ⚖️ Kết luận nhanh

- **Tốt nhất tổng thể:** Qwen3-4B-Gốc  
- **Tốt nhất domain YHCT:** Qwen3-4B-CPT (nhưng thiếu ổn định)  
- **Kém nhất:** Qwen3-4B-CPT-SFT  

---

## Cập nhật: 02/04/2026 – Khắc phục mất kiến thức sau SFT

### Vấn đề

Khi thực hiện SFT (Supervised Fine-Tuning) từ cả **base model** lẫn **CPT model**, mô hình đều bị **catastrophic forgetting** – mất kiến thức đã học trước đó. Biểu hiện rõ nhất là accuracy giảm mạnh và chất lượng câu trả lời nông hơn so với trước SFT.

### Giải pháp đã thử

1. **Freeze nhiều tham số hơn:** Đóng băng thêm các layer của mô hình trong quá trình SFT để giữ lại kiến thức gốc, chỉ cho phép fine-tune một phần nhỏ tham số.
2. **Thêm EWC (Elastic Weight Consolidation):** Áp dụng regularization EWC để phạt việc thay đổi quá lớn các tham số quan trọng đối với kiến thức đã học.

### Kết quả

- Có **cải thiện** so với SFT thông thường, nhưng **không đáng kể**.
- Mô hình vẫn chưa giữ được kiến thức tốt sau quá trình SFT.
- Cần tiếp tục thử nghiệm các phương pháp khác để giải quyết triệt để vấn đề catastrophic forgetting.