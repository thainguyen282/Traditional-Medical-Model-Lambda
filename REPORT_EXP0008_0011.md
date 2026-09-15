# Báo cáo: Thiết kế và trạng thái thí nghiệm exp0008–exp0011

## 1. Mục tiêu và thiết kế

Mục tiêu của nhóm thí nghiệm exp0008–exp0011 là kiểm tra ba câu hỏi:

1. QA mở quy mô khoảng 20K có giúp giảm hiện tượng sụp định dạng từng thấy khi SFT trên MCQ sinh hay không.
2. Fine-tune một giai đoạn hay hai giai đoạn trên VM14K cho khả năng fit benchmark và giữ kiến thức tốt hơn.
3. Bộ YHCT-MCQ-Reasoning-10K mới có hiệu quả hơn khi train riêng MCQ hay khi trộn thêm QA mở; đồng thời CPT-SFT có tiếp tục ổn định hơn SFT-only hay không.

Các thí nghiệm đều dùng Qwen3-4B-Instruct-2507, QLoRA 4-bit, LoRA `r=32`, `alpha=32`, rank-stabilized LoRA và NEFTune `alpha=5`. Tuy nhiên, pipeline và dữ liệu khác nhau:

- **exp0008:** CPT-SFT một giai đoạn trên hỗn hợp VM14K MCQ và QA-20K.
- **exp0009:** SFT giai đoạn hai, khởi tạo từ adapter exp0007 đã học QA-20K rồi chuyên biệt hóa trên VM14K.
- **exp0010:** YHCT reasoning MCQ; so sánh SFT-only từ base với CPT-SFT.
- **exp0011:** YHCT reasoning MCQ trộn QA-20K; so sánh SFT-only từ base với CPT-SFT.

Lưu ý quan trọng: VM14K trong exp0008/0009 không còn là benchmark generalization sạch vì dữ liệu train chứa VM14K. Điểm VM14K của hai thí nghiệm này phải được đọc như **fit/learnability**. Split test chính thức 29 dòng không được dùng để train; exp0009 yêu cầu chạy riêng evaluation trên split này sau khi push adapter.

## 2. Ma trận thí nghiệm

| exp | Data train | Pipeline | Epoch | LR | Validation | NEFTune |
|---|---|---|:--:|:--:|---|:--:|
| exp0008 | VM14K MCQ + QA-20K | CPT → mixed SFT | 3 | 5e-5 | 10% VM14K train split | 5 |
| exp0009 | VM14K MCQ | exp0007 QA-20K adapter → stage-2 SFT | 3 | 5e-5 | 10% VM14K train split | 5 |
| exp0010 | YHCT-MCQ-Reasoning-10K | SFT-only | 1 | 2e-4 | 5% YHCT train split | 5 |
| exp0010 | YHCT-MCQ-Reasoning-10K | CPT-SFT | 3 | 5e-5 | 5% YHCT train split | 5 |
| exp0011 | YHCT-MCQ-Reasoning-10K + QA-20K | SFT-only | 1 | 2e-4 | 5% YHCT train split | 5 |
| exp0011 | YHCT-MCQ-Reasoning-10K + QA-20K | CPT-SFT | 3 | 5e-5 | 5% YHCT train split | 5 |

## 3. Chi tiết từng thí nghiệm

### exp0008: VM14K MCQ + QA-20K trong một stage

exp0008 tiếp tục từ CPT adapter `huyennkai/Venera-CPT-TMModel@cpt-tm-cpt-1776775114`, sau đó train đồng thời VM14K-MCQ-Instruct và New-TradMedInstruct-qa_20k. Validation chỉ lấy từ VM14K với tỷ lệ 10%; QA-20K được dùng cho train.

Mục đích là kiểm tra liệu QA mở có đóng vai trò regularizer chống collapse khi trộn với MCQ, đồng thời vẫn giúp model fit VM14K. Vì VM14K đã xuất hiện trong train, điểm VM14K full-set không được dùng như bằng chứng generalization.

### exp0009: QA-20K trước, VM14K sau

exp0009 bắt đầu từ adapter exp0007:

`huyennkai/Venera-Traditional-Model@cpt-sft-unsloth-qwen3-4B-exp0007-tm-sft-1782355488`

Adapter này đã học QA-20K, sau đó exp0009 chỉ train thêm VM14K trong stage thứ hai. Thiết kế này tách thứ tự học thành hai giai đoạn, khác exp0008 ở chỗ exp0008 trộn QA và VM14K trong cùng một run.

So sánh exp0008 với exp0009 sẽ trả lời liệu **joint training** hay **sequential specialization** tốt hơn. Cần đặc biệt theo dõi catastrophic forgetting: exp0009 có thể fit VM14K cao hơn nhưng giảm điểm trên bộ New sạch do stage cuối quá hẹp.

### exp0010: YHCT reasoning MCQ thuần

exp0010 dùng `Maverfrick/YHCT-MCQ-Reasoning-10K` (10.174 train, 535 test theo dataset manifest). Hai pipeline cần được so sánh:

- SFT-only: train từ base Instruct, 1 epoch, LR 2e-4.
- CPT-SFT: train từ CPT adapter cố định, 3 epoch, LR 5e-5.

Đây là phép thử trực tiếp chất lượng của bộ reasoning MCQ mới. Nếu deviation thấp và điểm held-out tăng, dữ liệu mới đã khắc phục được lỗi định dạng của New-TradMedInstruct-multiplechoice. Nếu SFT-only vẫn collapse nhưng CPT-SFT ổn định, kết luận cũ về vai trò bảo vệ của CPT tiếp tục được củng cố.

### exp0011: YHCT reasoning MCQ + QA-20K

exp0011 giữ nguyên YHCT reasoning MCQ của exp0010 và thêm New-TradMedInstruct-qa_20k. Validation chỉ lấy từ YHCT; QA-20K train-only.

So sánh exp0011 với exp0010 đo tác động của QA mở:

- Nếu deviation giảm: QA đa dạng hóa output format và chống collapse.
- Nếu điểm MCQ held-out giảm nhưng QA/generalization tăng: có trade-off do task interference.
- Nếu cả accuracy lẫn deviation tốt hơn: mixed SFT là hướng scale phù hợp.
- Nếu SFT-only exp0011 kém mạnh hơn exp0010: tỷ trọng hoặc chất lượng QA cần được audit.

## 4. Bảng kết quả cần điền sau khi chạy

Hiện chưa có artifact hoặc evaluation output công khai cho exp0008–exp0011, vì vậy không được suy diễn accuracy từ config.

| Model | VM14K | New | YHCT test | dev VM14K | dev New | dev YHCT | Trạng thái |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| Baseline Qwen3-4B-Instruct | 48.05% | 53.33% | chưa đo | 0% | 0% | chưa đo | mốc cũ |
| exp0008 CPT-SFT mixed | chờ eval | chờ eval | N/A | chờ eval | chờ eval | N/A | chưa có artifact |
| exp0009 stage-2 VM14K | chờ eval | chờ eval | N/A | chờ eval | chờ eval | N/A | chưa có artifact |
| exp0010 SFT-only | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chưa có artifact |
| exp0010 CPT-SFT | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chưa có artifact |
| exp0011 SFT-only | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chưa có artifact |
| exp0011 CPT-SFT | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chờ eval | chưa có artifact |

Không nên viết kết luận “model nào tốt nhất” trước khi có đủ accuracy, denominator và deviation. Đặc biệt, accuracy thấp đi cùng deviation cao là lỗi tuân thủ format, không đồng nghĩa hoàn toàn với thiếu kiến thức.

## 5. So sánh cần thực hiện

### 5.1. exp0008 so với exp0009

- So sánh cùng checkpoint nền có nguồn gốc QA-20K/CPT nhưng khác lịch train: mixed một stage so với QA trước, VM14K sau.
- Báo cáo riêng điểm trên VM14K train/validation và official test 29 dòng.
- Dùng New 105 câu làm chỉ báo generalization sạch và forgetting.
- Kiểm tra response-level deviation để tránh kết luận nhầm khi model sai format.

### 5.2. exp0010 SFT-only so với CPT-SFT

- Đo tác động của CPT trên cùng YHCT reasoning MCQ.
- Tách lợi ích do số epoch/LR khỏi lợi ích do CPT: hai recipe hiện không chỉ khác initialization mà còn khác 1 so với 3 epoch và 2e-4 so với 5e-5.
- Nếu cần causal comparison chặt, chạy thêm ablation giữ cùng epoch và LR.

### 5.3. exp0011 SFT-only so với CPT-SFT

- Đo CPT có chống collapse trên mixture MCQ + QA hay không.
- So sánh deviation, không chỉ accuracy.
- Theo dõi loss theo từng nguồn nếu pipeline hỗ trợ; validation hiện chỉ đại diện cho YHCT.

### 5.4. exp0010 so với exp0011

- Đây là ablation về dữ liệu: MCQ thuần so với MCQ + QA.
- So sánh trong cùng pipeline trước: SFT-only với SFT-only, CPT-SFT với CPT-SFT.
- Không gộp chéo pipeline vì initialization, epoch và LR khác nhau.

## 6. Trạng thái artifact và đường dẫn

Hugging Face repo: [huyennkai/Venera-Traditional-Model](https://huggingface.co/huyennkai/Venera-Traditional-Model)

Link được cung cấp là artifact của **exp0004 SFT-only**, không phải exp0008–exp0011:

- [exp0004 SFT-only (`sft-unsloth-qwen3-4B-exp0004-tm-sft-1779619463`)](https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/sft-unsloth-qwen3-4B-exp0004-tm-sft-1779619463)

Tại thời điểm lập báo cáo, danh sách branch công khai của repo chỉ có experiment tới exp0007; chưa có branch mang tên exp0008, exp0009, exp0010 hoặc exp0011. Sau khi các run hoàn tất, thêm link theo mẫu:

- `https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/<revision-exp0008>`
- `https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/<revision-exp0009>`
- `https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/<revision-exp0010>`
- `https://huggingface.co/huyennkai/Venera-Traditional-Model/tree/<revision-exp0011>`

Mỗi dòng kết quả phải ghi rõ revision chính xác để bảo đảm config, adapter, response và metric cùng thuộc một snapshot.

## 7. Tiêu chí chốt kết luận

Chỉ chốt hướng scale sau khi có đủ kết quả theo các tiêu chí sau:

1. **Format stability:** deviation gần 0% trên tất cả eval.
2. **Generalization:** New và YHCT official test không chứa mẫu train tương ứng.
3. **Fit:** VM14K train/validation được báo riêng cho exp0008/0009, không gọi là generalization.
4. **Reproducibility:** mỗi kết quả gắn với Hugging Face revision, config và seed.
5. **Fair comparison:** ưu tiên so sánh trong cùng pipeline; ghi rõ confounder do epoch và LR khi so SFT-only với CPT-SFT.

Kỳ vọng cần kiểm chứng, không phải kết luận: exp0009 có thể fit VM14K mạnh nhất nhưng dễ quên QA; exp0011 có thể ổn định format hơn exp0010 nhờ QA mở; CPT-SFT nhiều khả năng ổn định hơn SFT-only dựa trên exp0004–exp0006. Các giả thuyết này chỉ được xác nhận sau khi artifact và evaluation output của exp0008–exp0011 được công bố.
