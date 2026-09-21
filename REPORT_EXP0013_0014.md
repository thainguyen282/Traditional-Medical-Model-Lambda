 # Báo cáo thí nghiệm exp0013–exp0014

Ngày lập: 15/09/2026. Trạng thái: đã đối chiếu thiết kế và audit dữ liệu local; kết quả inference trên Hugging Face đang chờ quyền truy cập.

## 1. Kết luận hiện có

**exp0013 học trực tiếp YHCT MCQ; exp0014 học QA-20K trước, sau đó tiếp tục học cùng YHCT MCQ.** Mỗi thí nghiệm có hai nhánh: khởi tạo từ base Instruct (SFT-only) và khởi tạo từ adapter CPT (CPT-SFT). Stage YHCT của cả bốn cấu hình có cùng hyperparameter chính, nên đây là phép so sánh tác động của việc thêm stage QA trước MCQ trong từng nhánh.

**Chưa thể kết luận exp0014 tốt hơn exp0013 về accuracy.** Link kết quả được cung cấp và API liệt kê revision đều trả HTTP 401 khi truy cập không xác thực. Báo cáo không điền số liệu suy đoán hoặc xem việc chưa đọc được artifact là bằng chứng run chưa hoàn tất.

**VM14K local có overlap lớn với dữ liệu YHCT:** 542/564 dòng eval có cùng câu hỏi, tập phương án và nội dung đáp án đúng trong dataset YHCT cache, sau khi bỏ khác biệt thứ tự phương án. Vì vậy, nếu artifact dùng đúng các snapshot này, điểm VM14K chủ yếu phản ánh mức độ fit dữ liệu có overlap; không phải benchmark độc lập để kết luận generalization.

## 2. Thiết kế và mục tiêu

| Thí nghiệm | Nhánh | Chuỗi huấn luyện | Câu hỏi cần kiểm chứng |
|---|---|---|---|
| exp0013 | SFT-only | Qwen3-4B-Instruct-2507 → YHCT MCQ | MCQ với prompt và target thống nhất có giữ định dạng trả lời ổn định không? |
| exp0013 | CPT-SFT | CPT adapter → YHCT MCQ | Khởi tạo CPT thay đổi chất lượng MCQ thế nào? |
| exp0014 | SFT-only | Qwen3-4B-Instruct-2507 → QA-20K → YHCT MCQ | Thêm QA trước MCQ có cải thiện kết quả so với exp0013 SFT-only không? |
| exp0014 | CPT-SFT | CPT adapter → QA-20K → YHCT MCQ | Thêm QA trước MCQ có cải thiện kết quả trong nhánh CPT không? |

YHCT sử dụng `Maverfrick/yhct-merged`. Loader ghép câu hỏi và phương án với instruction yêu cầu `Final Answer: X`; assistant target chỉ chứa đáp án theo định dạng này. Đây là huấn luyện trả lời MCQ ngắn, không phải huấn luyện sinh lời giải reasoning, dù dữ liệu gốc có thể xuất phát từ các nguồn khác nhau.

QA-20K sử dụng `venera-ai/New-TradMedInstruct-qa_20k`, giữ hội thoại QA mở qua trường `conversation`. exp0014 là **hai stage tuần tự**, không trộn QA và MCQ trong một stage. Stage hai tiếp tục adapter lưu local từ stage một theo `configs_sft.txt` hoặc `configs.txt`.

## 3. Cấu hình huấn luyện

| Thông số | exp0013: YHCT | exp0014 stage 1: QA-20K | exp0014 stage 2: YHCT |
|---|---:|---:|---:|
| Epoch cấu hình | 3 | 3 | 3 |
| Learning rate | 5e-5 | 5e-5 | 5e-5 |
| Batch/GPU | 8 | 2 | 8 |
| Gradient accumulation | 8 | 2 | 8 |
| Effective batch nếu 1 GPU | 64 | 4 | 64 |
| Validation từ split train | 5% | 20% | 5% |
| Eval/save interval (steps) | 100 | 200 | 100 |
| Max sequence length | 2048 | 2048 | 2048 |
| Training seed | 3407 | 3407 | 3407 |

Các cấu hình đều dùng 4-bit, BF16, LoRA `r=32`, `alpha=32`, dropout 0, rank-stabilized LoRA, NEFTune alpha 5, optimizer `adamw_8bit`, weight decay 0.01, cosine scheduler và 100 warmup steps. LoRA áp dụng lên các projection attention và MLP. `load_best_model_at_end=true`, chọn checkpoint theo `eval_loss`; số epoch cấu hình không chứng minh adapter được export là checkpoint ở bước cuối.

Seed chia dữ liệu trong loader là **42**, khác training seed 3407. Loader lấy split `train` rồi chia lại train/validation; không dùng official test như một phần train trong nhánh xử lý này.

### Khác biệt checkpoint CPT cần lưu ý

- exp0013 CPT-SFT: `huyennkai/Venera-CPT-TMModel@cpt-tm-cpt-1776775114`.
- exp0014 stage 1 CPT-SFT: `huyennkai/Venera-CPT-TMModel@main`.
- exp0014 stage 2 CPT-SFT: `group_outputs/exp0014/qa20k/CPT_SFT/model`.
- exp0014 stage 2 SFT-only: `group_outputs/exp0014/qa20k/SFT/model`.

Chưa xác minh `main` tại thời điểm train exp0014 trỏ tới cùng commit với revision CPT của exp0013. Do đó, chênh lệch giữa hai nhánh CPT-SFT có thể đồng thời chịu tác động của stage QA và checkpoint khởi tạo. exp0014 cũng có thêm dữ liệu và compute; không thể quy mọi cải thiện cho thứ tự curriculum nếu chưa có đối chứng cùng budget.

## 4. Audit dữ liệu local và ý nghĩa của VM14K

Snapshot YHCT cache: `b5b4739dfced8c622c2faec2ac80bd5bf44a4d68`. Metadata ghi **10.710 dòng**, chỉ có split `train`; trường `source` gồm 245 dòng `vm14k` và 10.465 dòng `venera`. Không thể chỉ dựa vào nhãn nguồn để xác định overlap.

Với tỷ lệ validation 5% và seed 42 như loader, phép tái dựng trên snapshot local cho **10.174 train / 536 validation**. Đây là tái dựng từ config/code, chưa phải xác nhận số dòng của run remote.

File eval local `EVAL/Data/eastern_medicine_VM14K.jsonl` có **564 dòng nhưng chỉ 232 câu hỏi phân biệt** sau chuẩn hóa Unicode NFKC, chữ thường và khoảng trắng. Cơ cấu difficulty: Easy 112, Moderate 339, Hard 39, Challenging 74. Notebook đặt `subset_size=10000`, nhưng giới hạn đó không đồng nghĩa đã eval 10.000 câu.

| Phép đối chiếu với 564 dòng eval local | Toàn bộ YHCT cache | Train tái dựng | Validation tái dựng |
|---|---:|---:|---:|
| Trùng nội dung câu hỏi sau chuẩn hóa | 542 | 512 | 43 |
| Trùng câu hỏi + phương án đúng thứ tự + ký tự đáp án | 214 | 196 | 18 |
| Trùng câu hỏi + tập phương án + nội dung đáp án, không xét thứ tự phương án | 542 | 499 | 43 |

Phép đối chiếu không xét thứ tự phương án dùng danh sách nội dung phương án đã chuẩn hóa rồi sắp xếp, và nội dung đáp án lấy theo `answer_index`. Kiểm tra toàn bộ eval local không phát hiện bất nhất giữa `answer` và `answer_index`.

Theo phép đối chiếu cuối, **96,10%** dòng eval trùng dataset nguồn và **88,48%** trùng train tái dựng. Chỉ 22/564 dòng không trùng dataset nguồn theo phép kiểm này; chưa đủ để gọi là tập test sạch vì chưa audit near-duplicate hoặc dữ liệu CPT/QA-20K. Các cột train và validation ở phép so câu hỏi không cộng trực tiếp: câu hỏi trùng lặp có thể xuất hiện ở cả hai. Có **50 câu hỏi phân biệt** chung giữa train và validation tái dựng.

Hệ quả: cần báo điểm có overlap riêng; đánh giá unseen phải dùng split theo nhóm câu hỏi trước khi shuffle phương án. Các bản hoán đổi phương án của một câu không phải các quan sát độc lập. Audit trên chỉ áp dụng cho snapshot local; cần đối chiếu dữ liệu và manifest của artifact trước khi áp dụng chính xác các tỷ lệ này cho run remote.

SHA-256 file eval local: `314006d439047ac9b98a890fe14ee1954025b28040c3e162ef71dc771af56574`.

## 5. Kết quả thực nghiệm và trạng thái xác minh

| Run | VM14K accuracy (đúng/tổng) | Deviation (số/tổng) |
|---|---|---|
| exp007 QA data | 57.23 | 0 |
| exp0013 SFT-only | 60.64 | 0 | 
| exp0013 CPT-SFT | 64.32 | 0 |
| exp0014 2 stage SFT-only | 62.54 | 0 | 
| exp0014 2 stage CPT-SFT |70.21 | 0 | 
| GPT‑6 Astra     | Chưa đánh giá |       Chưa đánh giá |
| Claude Opus 5   | Chưa đánh giá |       Chưa đánh giá |
| Gemini 3.1 Pro  | Chưa đánh giá |       Chưa đánh giá |
| Qwen3.8‑27B     | Chưa đánh giá |       Chưa đánh giá |
| Qwen3.5‑4B      | Chưa đánh giá |       Chưa đánh giá |
| Gemma‑4‑E4B     | Chưa đánh giá |       Chưa đánh giá |
| MedGemma‑1.5‑4B | 38.48 |       0 |
| HuatuoGPT‑o1‑8B | 41.13 |       0 |
| Qwen3‑4B base   | 48.05 |       0 |


Artifact được cung cấp: [exp0014 CPT-SFT — infer_result.txt](https://huggingface.co/Maverfrick/Venera-Traditional-Model/blob/cpt-sft-unsloth-qwen3-4B-exp0014-tm-sft-1789493283/eval/outputs/CPT_SFT_VM14K/infer_result.txt).

Revision trong URL: `cpt-sft-unsloth-qwen3-4B-exp0014-tm-sft-1789493283`. Tên branch cho biết run được nhắm tới, nhưng chưa đủ xác nhận nội dung config, checkpoint, denominator hoặc kết quả khi file chưa đọc được.

Log local `logs/exp0013-lambda.log` ghi một lần launch instance bị terminate trước khi active. Log này chỉ mô tả lần launch đó; không chứng minh exp0013 chưa được chạy thành công ở nơi khác.

## 6. Cách đọc metric và so sánh

Notebook VM14K local tính accuracy bằng số đáp án đúng chia tổng câu được chấm; deviation vẫn nằm trong mẫu số và bị tính sai. Deviation là trường hợp không trích được đáp án hợp lệ hoặc đáp án nằm ngoài tập phương án. Parser chấp nhận nhiều dạng ngoài `Final Answer: X`, kể cả heading Markdown và các fallback tìm chữ cái. Vì vậy, **deviation 0% không đồng nghĩa tuân thủ tuyệt đối template**, và nhãn “hallucination” trong biểu đồ không phải phép đo hallucination y khoa.

Generation config local: `max_new_tokens=250`, `do_sample=false`; temperature 0.7, top-p 0.8 và top-k 20 được khai báo nhưng không tạo sampling khi `do_sample=false`. Đây là cấu hình notebook hiện tại, chưa xác minh cấu hình thực thi remote.

Khi đọc được artifacts, đối chiếu `infer_result.txt` với `responses.jsonl` và manifest từng run, xác nhận cùng câu hỏi/phương án/nhãn trước khi tính chênh lệch. Báo cáo delta bằng điểm phần trăm và số câu đúng tăng/giảm cho:

1. exp0014 SFT-only so với exp0013 SFT-only: tác động quan sát được của việc thêm stage QA.
2. exp0014 CPT-SFT so với exp0013 CPT-SFT: cùng phép so, kèm kiểm tra commit CPT ban đầu.
3. CPT-SFT so với SFT-only trong từng experiment: tác động quan sát được của initialization, khi xác nhận các thành phần còn lại khớp.

Nếu có response từng câu, báo cả số câu đổi từ sai → đúng và đúng → sai; phân tích theo nhóm câu hỏi gốc để tránh coi các hoán đổi phương án là mẫu độc lập. Không suy luận khả năng QA mở được giữ lại chỉ từ điểm MCQ; cần eval QA riêng sau stage hai.

## 7. Nguồn và phần còn thiếu

Nguồn local đã đọc:

- [exp0013 SFT config](configs/experiments/unsloth-qwen3-4B/exp0013/config.yaml) và [CPT-SFT config](configs/experiments/unsloth-qwen3-4B/exp0013/config_cpt.yaml).
- [exp0014 stage 1 SFT](configs/experiments/unsloth-qwen3-4B/exp0014/stage1_qa20k/config.yaml), [stage 1 CPT-SFT](configs/experiments/unsloth-qwen3-4B/exp0014/stage1_qa20k/config_cpt.yaml), [stage 2 SFT](configs/experiments/unsloth-qwen3-4B/exp0014/config.yaml), [stage 2 CPT-SFT](configs/experiments/unsloth-qwen3-4B/exp0014/config_cpt.yaml).
- [YHCT dataset config](configs/dataset/venera_yhct_mcq.yaml), [QA-20K dataset config](configs/dataset/venera_qa_20k.yaml), [loader và chia dữ liệu](src/data/strategies/venera_med_m1.py).
- [Notebook CPT-SFT VM14K](EVAL/Unsloth_QA_CPT_SFT_VM14K.ipynb), [eval data local](EVAL/Data/eastern_medicine_VM14K.jsonl) và metadata/Arrow YHCT cache nêu ở mục 4.

Phần cần bổ sung để hoàn tất báo cáo kết quả: các revision exp0013/0014, nội dung inference và response logs, config/manifest của run, commit CPT thực tế và hash dữ liệu eval. Không cần chạy lại training để hoàn thành phần này nếu các artifacts đã có sẵn.
