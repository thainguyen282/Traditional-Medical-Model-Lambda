# Báo cáo: Probe learnability trên VM14K và tổng kết thí nghiệm

## 1. Mục tiêu và thiết kế

Mục tiêu: train thẳng model lên chính bộ eval VM14K để xác định model có thực sự học (nuốt) được kiến thức Y học cổ truyền hay không.

Cơ sở: `Maverfrick/VM14K-MCQ-Instruct` (564 dòng = 535 train + 29 test) chính là 564 câu của bộ eval `eastern_medicine_VM14K.jsonl`, chỉ convert sang chat format. Do đó với các thí nghiệm này:

- VM14K accuracy = mức độ model FIT được dữ liệu đã train (đo capacity/learnability), không phải generalization.
- Tín hiệu generalization sạch là New eval (105 câu held-out, model không train trên đó).
- Khoảng 428/564 câu VM14K thực sự đi vào gradient update (535 train, split 0.2 seed=42 thành ~428 train / 107 val); 29 câu test của dataset không được nạp.

Bốn thí nghiệm: exp0005 và exp0006, mỗi cái 2 pipeline (SFT-only và CPT-SFT).

## 2. Chi tiết từng thí nghiệm

| exp | Data train | Pipeline | VM14K (fit) | New (sạch) | dev VM14K / New |
|---|---|---|:--:|:--:|:--:|
| exp0005 | VM14K only (535) | SFT-only, 1 epoch, lr 2e-4 | 62.41% | 50.48% | 0% / 0% |
| exp0005 | VM14K only (535) | CPT to SFT, 3 epoch, lr 5e-5 | 75.35% | 53.33% | 0% / 0% |
| exp0006 | VM14K + New mix (~3.7k) | SFT-only, 1 epoch, lr 2e-4 | 13.30% | 1.90% | 81% / 94% |
| exp0006 | VM14K + New mix (~3.7k) | CPT to SFT, 3 epoch, lr 5e-5 | 75.18% | 41.90% | 2% / 16% |

Mốc so sánh (baseline Qwen3-4B-Instruct): VM14K 48.05%, New 53.33%, dev 0%.

Đọc từng exp:

- exp0005 SFT-only: fit VM14K 62.41% (chỉ hơn baseline 14 điểm dù được xem đáp án), format sạch (dev 0%). Train ít (1 epoch, ~107 step) nên fit yếu.
- exp0005 CPT-SFT: fit VM14K 75.35% (cao nhất), format sạch. CPT base cộng 3 epoch giúp nuốt data tốt hơn rõ rệt. New = 53.33% đúng bằng baseline, nghĩa là kiến thức học được không transfer sang câu chưa thấy, chủ yếu là nhớ câu cụ thể.
- exp0006 SFT-only: sụp đổ format hoàn toàn, dev 81% (VM14K) và 94% (New), accuracy rơi xuống 13% và 2%. Model gần như không xuất nổi đáp án hợp lệ.
- exp0006 CPT-SFT: CPT cứu lại, dev về 2% và 16%, fit VM14K 75.18% (xấp xỉ exp0005, tức thêm data New không giúp fit VM14K hơn), New 41.90%.

## 3. Kết luận learnability

1. Chỉ SFT-only thì bộ data MCQ gen ra (New-TradMedInstruct) khiến model sụp đổ format hoàn toàn: exp0006 mix dev 81 đến 94%, và trước đó exp0004 (New-mc thuần) dev 34 đến 53%. Model mất khả năng tuân thủ định dạng đáp án. Ngoại lệ: VM14K-only SFT (exp0005) không sụp, dev 0%, nhưng chỉ fit 62%.

2. Có thêm CPT thì data chỉ ảnh hưởng rất nhỏ:
   - Format gần như nguyên vẹn: deviation từ 94% về 16% (New) và từ 81% về 2% (VM14K).
   - Năng lực tổng quát (New, sạch) chỉ giảm 0 đến 11.4 điểm phần trăm so baseline: exp0005 CPT-SFT 53.33% bằng baseline (giảm 0), exp0006 CPT-SFT 41.90% (giảm 11.4).
   - Đồng thời vẫn fit VM14K tới khoảng 75%.

3. Model có thực sự học được kiến thức không: kể cả train thẳng lên đáp án, CPT-SFT chỉ fit khoảng 75% tổng (ước tính khoảng 84% trên câu đã-train, khoảng baseline trên câu chưa-train), tức không học thuộc trọn vẹn. SFT-only fit kém hơn (khoảng 62 đến 67%) hoặc collapse. Kết luận: model đang under-fit (LoRA r=32, số epoch và lr chưa đủ) và kiến thức không transfer; CPT là điều kiện bắt buộc để model còn dùng được.

## 4. Tổng kết tất cả thí nghiệm đã làm (leaderboard)

Cột VM14K của exp0005/0006 là điểm FIT (đã train trên data này), không so sánh như generalization. New 105 câu là tín hiệu sạch nhất.

| Model | VM14K | New | dev | Ghi chú |
|---|:--:|:--:|:--:|---|
| Baseline Qwen3-4B-Instruct | 48.05 | 53.33 | 0% | mốc so sánh |
| CPT-only | 48.76 | 57.14 | 0% | format sạch, generation tốt |
| exp0001 SFT 2topics | 46.45 | 63.81 | 0% | New cao nhất |
| exp0001 CPT-SFT 2topics | 47.16 | 62.86 | ~1% | |
| exp0001 SFT mc(old) | 34.04 | 30.48 | 3 đến 8% | regression |
| exp0001 CPT-SFT mc(old) | 48.05 | 58.10 | 0% | |
| exp0002 SFT mix(old+mc) | 44.50 | 48.57 | 7 đến 19% | dev cao |
| exp0003 CPT-SFT 2topics+NEFTune | 47.52 | 63.81 | 0% | tốt nhất trên dữ liệu sạch |
| exp0004 SFT New-mc | 25.53 | 19.05 | 34 đến 53% | collapse |
| exp0004 CPT-SFT New-mc | khoảng 48 | 62.86 | 0% | |
| exp0005 SFT VM14K (train) | 62.41 | 50.48 | 0% | fit yếu |
| exp0005 CPT-SFT VM14K (train) | 75.35 | 53.33 | 0% | fit tốt nhất, New bằng baseline |
| exp0006 SFT mix (train) | 13.30 | 1.90 | 81 đến 94% | collapse |
| exp0006 CPT-SFT mix (train) | 75.18 | 41.90 | 2 đến 16% | |

Model tốt nhất trên dữ liệu sạch vẫn là exp0003 (CPT-SFT 2topics cộng NEFTune): New 63.81%, VM14K sạch 47.52%, dev 0%.

## 5. Chốt hướng scale (cách train)

1. Cố định recipe lõi CPT to SFT cộng NEFTune (alpha=5). Đã chứng minh: CPT chống collapse (dev 94% về 2 đến 16%), NEFTune đưa dev về 0. Không bao giờ SFT-only trên MCQ gen.
2. Bật chat_mix_ratio khoảng 0.1 (mix FineTome, infra có sẵn, exp5/6 chưa bật) để gia cố chống collapse khi SFT.
3. Tách eval khỏi train vĩnh viễn: đóng băng VM14K 564 cộng New test làm held-out; mọi train phải disjoint thì mới đo được generalization thật.
4. Đòn bẩy generalization là scale data, không phải train lên benchmark: mở rộng corpus CPT cộng SFT instruct disjoint eval, theo hướng 2topics (recipe exp0003 đang dẫn đầu sạch). Train lên VM14K chỉ cho điểm fit, không nâng năng lực thật.
5. Nếu mục tiêu là probe capacity sâu hơn: tăng LoRA r (32 lên 64 hoặc 128), tăng epoch và lr trên VM14K-train, vẽ đường cong fit; và phân rã seen vs unseen (reproduce split seed=42, đối chiếu responses.jsonl theo id) để tách rõ nhớ với transfer.

## 6. Chốt hướng làm data

Phát hiện cốt lõi từ góc data: bộ MCQ gen ra (New-TradMedInstruct-multiplechoice) chính là thủ phạm gây sụp đổ format, không phải pipeline. Bằng chứng: SFT-only trên 2topics (dev 0%) và trên VM14K-only (dev 0%) đều không sụp; chỉ khi có New-mc mới sụp (exp0004 dev 34 đến 53%), và trộn thêm New-mc vào VM14K còn làm tệ hơn (exp0006 dev 81 đến 94% so với VM14K-only 0%). Nghĩa là New-mc là thành phần độc, càng nhiều càng hỏng.

Hướng làm data theo thứ tự ưu tiên:

1. Sửa pipeline gen của New-TradMedInstruct trước khi dùng lại. Đây là gốc rễ collapse. Cần audit: tính nhất quán định dạng đáp án (### X), rò rỉ token scaffolding (tool_call, think đã thấy ở inference report), chất lượng distractor, phân phối đáp án (kiểm tra model có học mẹo luôn chọn B không), và trùng lặp gần làm data nghèo. Chưa fix thì không scale tập này.
2. Tách nguồn train khỏi eval ngay từ khâu gen. Đóng băng VM14K 564 cộng New test làm held-out vĩnh viễn. Khi sinh MCQ, loại mọi source_id hoặc page_id trùng với câu trong eval để số generalization về sau là thật.
3. Phá thế độc canh định dạng để chống collapse từ data. Đừng để 100% là MCQ hẹp dạng câu hỏi cộng lý giải ngắn cộng ### letter. Trộn nhiều dạng: MCQ, QA mở, giải thích tự do, chat tổng quát (chat_mix_ratio khoảng 0.1). 2topics chạy tốt chính vì rộng hơn, New-mc sụp vì quá đơn điệu.
4. CPT corpus là đòn bẩy kiến thức chính, không phải MCQ. Mở rộng raw TCM text cho CPT để bơm kiến thức nền (CPT giữ format sạch, thêm domain signal mà không collapse). MCQ và SFT chỉ để dạy cách trả lời, không phải kênh nhồi kiến thức chính; train thẳng MCQ chỉ fit khoảng 75% và không transfer.
5. Nhân bản theo công thức 2topics, không theo VM14K. Tập sinh ra nên giống phong cách 2topics-3546 (đa dạng, có lý giải, disjoint eval), đây là data đang cho generalization tốt nhất (New 63.81%). Vài nghìn item chất lượng cao và cân bằng topic, difficulty quan trọng hơn số lượng lớn nhưng hẹp.
6. Giữ chain-of-thought nhưng làm sạch token. Đáp án nên có lý giải (đã có), nhưng strip token Qwen3 thừa khỏi data trước khi train để inference không leak.

Tóm tắt một câu: scale kiến thức bằng CPT corpus cộng SFT data đa dạng kiểu 2topics, disjoint hoàn toàn với eval; còn New-TradMedInstruct phải audit và fix pipeline gen trước, không trộn thô như exp0006.
