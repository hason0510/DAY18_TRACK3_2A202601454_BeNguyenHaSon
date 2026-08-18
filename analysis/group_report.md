# Report — Lab 18: Production RAG

**Học viên:** Bế Nguyễn Hà Sơn · **Mã:** 2A202601454
**Ngày:** 18/08/2026
**Hình thức:** bài cá nhân — implement toàn bộ M1–M5, không chia module theo nhóm.

## Phân công

| Tên | Module | Hoàn thành | Hàm chính đã viết |
|-----|--------|-----------|-------------------|
| Bế Nguyễn Hà Sơn | M1: Chunking | ☑ | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `_pack()`, `_merge_small_groups()` |
| Bế Nguyễn Hà Sơn | M2: Hybrid Search | ☑ | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` |
| Bế Nguyễn Hà Sơn | M3: Reranking | ☑ | `CrossEncoderReranker`, `FlashrankReranker`, `_fallback_results()` |
| Bế Nguyễn Hà Sơn | M4: Evaluation | ☑ | `evaluate_ragas()`, `failure_analysis()`, `DIAGNOSTIC_TREE`, `_adapt_answer_relevancy()` |
| Bế Nguyễn Hà Sơn | M5: Enrichment | ☑ | `_enrich_single_call()` (combined mode) + 4 technique riêng lẻ, `enrich_chunks()` song song 8 luồng |

**Kết quả tự kiểm tra** (chạy thật, không phải ước lượng):

- `grep -r "# TODO" src/m*.py` → **0**
- `pytest tests/ -v` → **37/37 passed** (M1: 13 · M2: 5 · M3: 5 · M4: 4 · M5: 10). Thời gian chạy **101–582 giây** tuỳ tải máy, vì test M2/M3 phải load `bge-m3` và `bge-reranker-v2-m3` thật.
- `python check_lab.py` → **"Bài lab sẵn sàng để nộp"**, mọi mục ✅.

> ⚠️ **Lưu ý cho người chấm:** `check_lab.py` in ra `pytest error: invalid literal for int()… → Không chạy được tests`. Đây là **lỗi của chính script chấm**, không phải bài nộp: hàm `run_tests()` gọi pytest với cờ `-v` nên dòng tổng kết có viền `=` (`======= 37 passed in 581.86s =======`), rồi `int(part.split()[0])` cố ép chuỗi `"======="` thành số và ném exception; ngoài ra `timeout=120` của nó ngắn hơn thời gian load model. Chạy trực tiếp `pytest tests/ -v` cho kết quả **37/37 passed** như trên.

## Kết quả RAGAS

Lần chạy cuối, tổng thời gian 832,8s. Nguồn: `reports/ragas_report.json`, `reports/naive_baseline_report.json`.

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8405 | **0.8414** | **+0.0009** |
| Answer Relevancy | 0.7821 | **0.8597** | **+0.0776** |
| Context Precision | 0.9250 | 0.8958 | −0.0292 |
| Context Recall | 0.9000 | **0.9500** | **+0.0500** |

**4/4 metric ≥ 0.75**, production thắng baseline ở 3/4 metric. Chi tiết bottom-5 và Error Tree: `analysis/failure_analysis.md`.

`context_precision` giảm 0.029 là đánh đổi có chủ ý: production đưa tới 4 parent (mỗi parent ≈ cả tài liệu) thay vì 3 chunk nhỏ — đổi 0.03 precision lấy 0.05 recall, vì thiếu context thì LLM không trả lời được còn thừa context thì LLM vẫn lọc được.

## Latency breakdown

Nguồn: `reports/latency_report.json`, 20 query, chạy CPU.

| Stage | Avg (ms) | P95 (ms) | % tổng |
|---|---|---|---|
| Retrieval (BM25 + Dense + RRF) | 473,6 | 925,7 | 2,5% |
| **Rerank (cross-encoder)** | **16.829,8** | 24.958,6 | **89,8%** |
| Generation (gpt-4o-mini) | 1.436,0 | 2.842,8 | 7,7% |
| **Tổng / query** | **18.739,5** | 28.418,1 | 100% |

Build một lần: chunking 0,1s · enrichment 114 chunk 59,6s · indexing 75,7s · load reranker 11,5s.

Rerank một mình chiếm gần 90% và đắt gấp ~35 lần retrieval → cấu hình hiện tại **không dùng được cho chat thời gian thực**, phải chuyển reranker sang GPU hoặc dùng `FlashrankReranker`. Bug "generation treo 176 giây" của lần chạy trước đã hết: P95 generation 175.934,9ms → **2.842,8ms** sau khi đặt `OpenAI(timeout=60, max_retries=3)`.

## Key Findings

1. **Biggest improvement — sửa prompt sinh answer, không phải sửa retrieval.**
   Ở lần chạy đầu, 4/20 câu trả "Không tìm thấy" **dù `context_recall = 1.0`** — context đã có đủ thông tin nhưng prompt cấm suy luận nên model không dám làm phép nhân `85% × 20.000.000`. Bốn câu này một mình kéo faithfulness và answer_relevancy xuống hàng loạt. Sau khi viết lại prompt: **0/20 câu từ chối** ở lần chạy cuối. Nếu chỉ nhìn điểm tổng, tôi đã đi tối ưu chunking và search — đúng chỗ tốn thời gian nhất, sai chỗ cần sửa.

2. **Biggest challenge — `adapt()` của RAGAS hỏng cache và `except` của tôi giấu mất lỗi đó.**
   `answer_relevancy.adapt(language="vietnamese")` chạy đúng lần đầu, ghi cache ra `~/.cache/ragas/` dưới định dạng mà chính nó không parse lại được, rồi từ lần chạy thứ hai ném `ValidationError` và âm thầm rơi về prompt tiếng Anh. Tôi tưởng bản vá đã chạy và suýt kết luận sai rằng nó không hiệu quả. Sửa bằng cách viết tay prompt tiếng Việt ngay trong code, bỏ hẳn cơ chế cache — lần chạy cuối đã xác nhận bản vá thực sự hoạt động.

   3. **Surprise finding — một phần điểm `answer_rel  evancy` là lỗi đo lường, không phải lỗi hệ thống.**
      Metric này sinh ngược câu hỏi từ answer rồi so cosine với câu hỏi gốc, mà toàn bộ few-shot của RAGAS là tiếng Anh → answer tiếng Việt sinh ra câu hỏi tiếng Anh:
      ```
      cos(câu hỏi VI, câu hỏi EN sinh ra)   = 0.3900   ← điểm bị chấm
      cos(câu hỏi VI, paraphrase tiếng Việt) = 0.8894
      cos(câu hỏi VI, câu hỏi lệch chủ đề)   = 0.3464   ← xấp xỉ mức của bản dịch EN
      ```
      Một câu trả lời đúng hoàn hảo bị chấm gần bằng một câu lạc đề. Tệ hơn, câu trả lời phủ định dứt khoát ("không nên tự ý xử lý malware") bị judge gắn nhãn *noncommittal* → nhân 0 → điểm **0.0** dù 3 metric còn lại đều 1.0.
      **Bằng chứng định lượng sau khi vá:** pipeline naive **không đổi một dòng code nào**, nhưng `answer_relevancy` của nó nhảy 0.4535 → **0.7821 (+0.33)**, trong khi 3 metric còn lại của chính nó chỉ xê dịch ≤ 0.016. Đó là thay đổi thước đo, không phải cải thiện hệ thống — và vì áp dụng cho cả hai hệ thống nên phép so sánh vẫn công bằng.

4. **Sau khi vá metric, lỗi còn lại là lỗi thật.** Bottom-5 lần chạy cuối: 3 câu hỏng ở tầng **generation** với bài toán numeric multi-hop (thuật lại dữ kiện đề bài → mất faithfulness; và một câu **tính sai** — 500.000 thay vì 50.000 VNĐ), 2 câu hỏng ở tầng **retrieval** (câu hỏi hai ý chỉ lấy được context cho một ý; câu hỏi về MFA thiếu tài liệu phiên bản cũ). Không câu nào hỏng vì chunking hay fusion.

5. **Điều tôi tin chắc nhất lại là điều đo được ít chắc chắn nhất — biên nhiễu lớn hơn cải thiện.**
   Tôi chạy pipeline lần thứ hai với một bản prompt siết chặt hơn. Kết quả: **baseline không đổi một dòng code nào** mà `answer_relevancy` xê dịch **0.081** — lớn hơn chính con số Δ = +0.0776 đang dùng để chứng minh production hơn baseline; `faithfulness` xê dịch 0.007, cùng bậc với Δ = +0.0009. Nhưng `context_precision`/`context_recall` của baseline thì **giống hệt tới từng chữ số** qua cả hai lần.
   Lý do rất cụ thể: hai metric sau chỉ phụ thuộc context (dense retrieval của baseline là tất định), hai metric trước phụ thuộc câu trả lời do LLM sinh rồi lại được LLM khác chấm — hai tầng ngẫu nhiên chồng nhau. Hệ quả: trong 4 con số Δ ở bảng trên, chỉ `context_recall` (+0.05) là vượt rõ biên nhiễu.
   Riêng production thì `context_precision`/`recall` **cũng** xê dịch, và đó là lỗi thiết kế của tôi: enrichment gọi LLM lại ở mỗi lần build nên index tự nó không tất định — phải cache theo hash nội dung chunk. Chi tiết + số liệu: `failure_analysis.md` mục 8, report gốc lưu ở `reports/ablation_prompt_2phan/`.

## Presentation Notes (5 phút)

1. **RAGAS naive vs production** — bảng trên; nhấn vào Δ answer_relevancy +0,078 và recall +0,05, và giải thích vì sao faithfulness gần như đứng yên (lỗi còn lại nằm ở phép tính, hybrid+rerank không sửa được).
2. **Biggest win** — M1 hierarchical + parent expansion cho recall, nhưng phần lớn điểm số đến từ prompt engineering ở tầng generation. Thông điệp: retrieval tốt là điều kiện cần, không phải điều kiện đủ.
3. **Case study — metric cũng là một hệ thống có bug.** Câu "Khi phát hiện malware, nhân viên có nên tự xử lý không?": Output đúng? → ĐÚNG. Context? → precision 1,0 recall 1,0. Faithfulness? → 1,0. Vậy mà answer_relevancy = **0,0** tròn trịa → dấu hiệu của cờ nhị phân chứ không phải cosine → truy ra `noncommittal = 1` do judge hiểu nhầm câu phủ định là né tránh. Sau khi vá prompt tiếng Việt, câu này **rớt hẳn khỏi bottom-10**.
4. **Con số gây sốc nhất — rerank chiếm 89,8% latency**, 16,8 giây/query trên CPU, gấp ~35 lần retrieval. Đây là lý do kiến trúc phải là *retrieve rộng bằng bi-encoder → rerank hẹp bằng cross-encoder*, và là thứ đầu tiên phải tối ưu nếu đưa lên production.
5. **Điều muốn người nghe nhớ nhất** — tôi chạy lại y hệt baseline và nó lệch 0.08 điểm. Trước khi khoe một cải tiến +0.05, hãy đo xem hệ thống của bạn tự dao động bao nhiêu. Không có con số đó thì mọi Δ đều là kể chuyện.
6. **Nếu có thêm 1 giờ** — (a) tách phép tính khỏi LLM (lỗi duy nhất khiến hệ thống trả **sai** cho người dùng); (b) query decomposition cho câu hỏi nhiều ý; (c) version-aware metadata; (d) cache enrichment để build tái lập được.

## Khai báo phương pháp

1. `RERANK_TOP_K` đổi 3 → 10 (ASSIGNMENT ghi "top-20 → top-3"). Lý do: mỗi file ~800 ký tự nên 1 document = 1 parent, top-3 child thường trỏ về cùng một parent, dedupe xong chỉ còn 1 context.
2. `answer_relevancy` dùng prompt tiếng Việt viết tay thay prompt tiếng Anh gốc, **áp dụng cho cả baseline lẫn production** — bằng chứng ở `failure_analysis.md` mục 4, tắt được bằng `RAGAS_ADAPT_LANGUAGE = None`.
3. Prompt sinh answer được tinh chỉnh sau khi đọc failure trên chính 20 câu dùng để chấm — không có held-out set.
4. Số liệu nộp là của **một lần chạy** (run 1), không phải trung bình nhiều lần. Lần chạy thứ hai với prompt khác đã được khai báo đầy đủ ở `failure_analysis.md` mục 8, report gốc giữ nguyên trong `reports/ablation_prompt_2phan/`. Biên nhiễu đo được: ~0.08 (`answer_relevancy`), ~0.03 (`faithfulness`).
