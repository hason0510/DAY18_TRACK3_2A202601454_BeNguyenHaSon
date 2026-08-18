# Failure Analysis — Lab 18: Production RAG

**Học viên:** Bế Nguyễn Hà Sơn · **Mã:** 2A202601454
**Ngày:** 18/08/2026 · **Bài cá nhân** (implement toàn bộ M1–M5)
**Nguồn số liệu:** `reports/ragas_report.json`, `reports/naive_baseline_report.json`, `reports/latency_report.json` (lần chạy cuối, tổng 832,8s)

---

## 1. Cấu hình hai hệ thống được so sánh

| | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic()` — cắt theo paragraph, 500 ký tự | `chunk_hierarchical()` — parent 2048 / child 256, index child → trả parent |
| Enrichment | không | M5 combined mode, 1 API call/chunk (context + summary + HyQA + metadata) |
| Retrieval | Dense-only (bge-m3 + Qdrant), top-3 | Hybrid BM25(underthesea) + Dense, hợp nhất bằng RRF, top-20 |
| Reranking | không | CrossEncoder `bge-reranker-v2-m3`, top-20 → top-10 |
| Context đưa vào LLM | 3 chunk thô | ≤ 4 parent khác nhau, có gắn `[Nguồn: <file>]` |
| Prompt | "Trả lời CHỈ dựa trên context. Nếu không có → 'Không tìm thấy.'" | 8 quy tắc (xem `ANSWER_SYSTEM_PROMPT` trong `src/pipeline.py`) |

Corpus: **26 tài liệu** (25 `.md` + `so_tay_an_toan.pdf`) → **114 child chunk / 26 parent**.
2 file PDF (`BCTC.pdf`, `Nghi_dinh_13-2023.pdf`) bị bỏ qua: cả 2 có **0 ký tự text ở mọi trang, 0 font object** → scan ảnh thuần, cần OCR. Không câu hỏi nào trong `test_set.json` liên quan tới 2 file này nên việc bỏ qua không ảnh hưởng điểm; OCR chúng chỉ thêm nhiễu (39 trang nghị định) và kéo `context_precision` xuống.

---

## 2. RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8405 | **0.8414** | **+0.0009** |
| Answer Relevancy | 0.7821 | **0.8597** | **+0.0776** |
| Context Precision | 0.9250 | 0.8958 | −0.0292 |
| Context Recall | 0.9000 | **0.9500** | **+0.0500** |

**4/4 metric ≥ 0.75.** Production thắng baseline ở 3/4 metric.

Hai điểm cần đọc kỹ trong bảng này:

- **`faithfulness` gần như đứng yên (+0.0009).** Toàn bộ điểm mất của production nằm ở 3 câu numeric multi-hop (mục 3, #1/#2/#5) — không phải lỗi retrieval mà là lỗi tầng generation (làm phép tính, thuật lại dữ kiện câu hỏi). Hybrid + rerank không sửa được nhóm lỗi này, nên việc nó không nhích lên là hợp lý chứ không phải dấu hiệu pipeline vô dụng. **Lưu ý quan trọng:** +0.0009 nhỏ hơn biên nhiễu giữa hai lần chạy (~0.03 cho faithfulness) nên không được đọc là "production hơn baseline một chút" — đúng hơn là "hai hệ thống không phân biệt được ở metric này". Bằng chứng đo được ở mục 8.
- **`context_precision` giảm 0.029 là đánh đổi có chủ ý.** Baseline đưa 3 chunk nhỏ, production đưa tới 4 parent (mỗi parent ≈ cả một tài liệu). Nhiều thông tin hơn → recall tăng 0.05 nhưng tỉ lệ câu thừa trong context cũng tăng. Với bộ câu hỏi multi-hop thì đổi 0.03 precision lấy 0.05 recall là có lợi: thiếu context thì LLM không trả lời được, còn thừa context thì LLM vẫn lọc được.

---

## 3. Bottom-5 Failures

Lấy từ `failure_analysis()` — xếp theo trung bình 4 metric tăng dần.

### #1 — avg 0.7117 · worst: **faithfulness = 0.1111**

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày. Quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 = 300.000 VNĐ/tháng → pro-rata ≈ **50.000 VNĐ**.
- **Got:** "…mức phạt: 2% × 15.000.000 VNĐ × (1/6) = **500.000 VNĐ**."
- **Metrics:** faithfulness 0.1111 · answer_relevancy 0.9022 · context_precision 0.8333 · context_recall 1.0
- **Error Tree:**
  1. Output đúng không? → **SAI**. Model dựng đúng công thức (2%/tháng, quá hạn 5 ngày ≈ 1/6 tháng) nhưng ra 500.000 thay vì 50.000 — **lệch đúng một chữ số**.
  2. Context đúng không? → **ĐỦ**: recall 1.0, precision 0.833. `tam_ung.md` có mức phạt 2%/tháng, `chi_phi_expense.md` có thời hạn 15 ngày.
  3. Retrieval? → không phải nguyên nhân.
  4. → **Lỗi thuần ở tầng generation: số học sai.**
- **Root cause:** faithfulness 0.1111 nghĩa là chỉ 1/9 claim truy được về context. Hai vấn đề chồng lên nhau: (a) answer mở đầu bằng việc thuật lại dữ kiện của **câu hỏi** ("tạm ứng 15 triệu", "sau 20 ngày") — không có trong tài liệu nên bị tính là không trung thực (giống #5); (b) các bước tính trung gian là do model tự sinh, và bước cuối sai. `gpt-4o-mini` không đáng tin với chuỗi phép tính nhiều bước có đổi đơn vị (tháng → ngày).
- **Suggested fix:** đưa phép tính ra khỏi LLM — cho model trả về công thức có cấu trúc (`base`, `rate`, `days`) rồi tính bằng Python, hoặc dùng tool/function calling. Ở mức prompt: bắt viết rõ từng bước kèm đơn vị và tự kiểm tra lại kết quả chia trước khi kết luận.

### #2 — avg 0.7236 · worst: **faithfulness = 0.0**

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng. Lương thử việc = 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ. Cách tính là 85% × 20.000.000 VNĐ = 17.000.000 VNĐ."
- **Metrics:** faithfulness 0.0 · answer_relevancy 0.8945 · context_precision 1.0 · context_recall 1.0
- **Error Tree:**
  1. Output đúng không? → **ĐÚNG**, trùng khớp ground truth tới từng đồng.
  2. Context đúng không? → **ĐÚNG**, precision 1.0 và recall 1.0 — cả `bang_luong_2024.md` (Junior 20tr) lẫn `thu_viec.md` (85%) đều có trong context.
  3. Query rewrite? → không dùng, không phải nguyên nhân.
  4. → **Lỗi nằm ở METRIC, không nằm ở pipeline.**
- **Root cause:** RAGAS faithfulness tách answer thành các claim rồi hỏi "claim này có suy ra được trực tiếp từ context không?". Claim "85% × 20.000.000 = 17.000.000" là kết quả **phép tính bắc cầu qua 2 tài liệu**; không câu nào trong context viết ra con số 17.000.000. Judge trả lời "không" cho mọi claim → 0.0. Đây là giới hạn đã biết của faithfulness với câu hỏi numeric multi-hop: metric thưởng cho việc chép lại, phạt cho việc suy luận — kể cả suy luận đúng.
- **Suggested fix:** (a) tách riêng nhóm câu numeric multi-hop và chấm bằng exact-match số học thay vì faithfulness; (b) nếu buộc dùng RAGAS, yêu cầu answer trích dẫn nguyên văn hai tiền đề trước khi đưa kết quả ("Bảng lương ghi Junior tối đa 20.000.000 VNĐ; quy chế thử việc ghi 85% lương chính thức; do đó…") để mỗi claim đều truy được về context.

### #3 — avg 0.7578 · worst: **context_recall = 0.5**

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu **v2.0** hiện hành, bắt buộc MFA cho email/VPN/hệ thống nội bộ. **Chính sách cũ v1.0 không yêu cầu MFA.**
- **Got:** "Có, tất cả nhân viên bắt buộc phải kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Metrics:** faithfulness 1.0 · answer_relevancy 0.5313 · context_precision 1.0 · context_recall **0.5**
- **Error Tree:**
  1. Output đúng không? → **ĐÚNG NHƯNG THIẾU**: đúng phần "có", thiếu vế đối chiếu với v1.0.
  2. Context đúng không? → precision 1.0 nhưng recall **0.5**: 4 context là `mat_khau_v2.md`, `mua_sam.md`, `vpn_truy_cap.md`… — **`mat_khau_v1.md` không được retrieve**.
  3. → **Lỗi ở retrieval**: thiếu đúng một nửa dữ kiện của ground truth.
- **Root cause:** ground truth có 2 claim, claim thứ hai là claim **phủ định về bản cũ**. Query không chứa từ khoá nào trỏ tới phiên bản, nên cả BM25 lẫn dense đều xếp `mat_khau_v2.md` (bản có MFA) lên đầu và đẩy bản cũ ra ngoài top. Đây là mặt trái thú vị của mục 7.3: ở "nghỉ phép năm" và "thâm niên" việc lấy **cả cặp** phiên bản làm hỏng precision, còn ở đây việc **không lấy đủ cặp** làm hỏng recall. Kết luận đúng không phải "loại bản cũ" mà là "**gắn nhãn** phiên bản rồi để prompt quyết định".
- **Suggested fix:** thêm `version` + `effective_date` vào metadata lúc enrichment; khi các chunk cùng `doc_family` có nhiều phiên bản, kéo bản hiện hành lên context và **giữ 1 slot** cho bản superseded, đồng thời chỉ thị prompt: trả lời theo bản hiện hành, có ghi chú nếu bản cũ khác.

### #4 — avg 0.7647 · worst: **context_recall = 0.5**

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = **18 ngày**; lương Senior (P3–P4): **20–35 triệu VNĐ/tháng**.
- **Got:** "…được nghỉ 18 ngày phép năm… Lương trong khoảng này sẽ được hưởng đầy đủ trong thời gian nghỉ phép."
- **Metrics:** faithfulness 0.8 · answer_relevancy 0.7588 · context_precision 1.0 · context_recall **0.5**
- **Error Tree:**
  1. Output đúng không? → **ĐÚNG NỬA**: 18 ngày chính xác, **không trả lời được vế lương** (và lảng sang một câu vô nghĩa).
  2. Context đúng không? → recall 0.5: 4 context đều là tài liệu nghỉ phép (`nghi_phep_nam_v2024`, `v2023`, `nghi_phep_khong_luong`…), **`bang_luong_2024.md` không có mặt**.
  3. → **Lỗi ở retrieval với câu hỏi hai ý (multi-intent).**
- **Root cause:** query có 2 ý ("bao nhiêu ngày phép" + "lương khoảng nào") nhưng chỉ được embed thành **một** vector. Vốn từ vựng nghỉ phép áp đảo nên top-20 hybrid gần như toàn tài liệu nghỉ phép; sau dedupe theo parent, 4 slot context bị 3 tài liệu nghỉ phép gần trùng nhau chiếm mất. Ý thứ hai không còn chỗ.
- **Suggested fix:** (a) **query decomposition** — tách câu hỏi nhiều ý thành các sub-query, retrieve riêng rồi hợp nhất, đảm bảo mỗi ý có ít nhất 1 context; (b) thêm ràng buộc đa dạng kiểu MMR ở tầng parent để 3 tài liệu gần trùng không chiếm hết slot.

### #5 — avg 0.7705 · worst: **faithfulness = 0.25**

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Phải cam kết làm việc ít nhất 1 năm sau khóa học. Nghỉ sau 8 tháng là trước hạn → hoàn trả 100% chi phí = 25.000.000 VNĐ.
- **Got:** "Nhân viên được tài trợ khóa học 25 triệu và nghỉ việc sau 8 tháng hoàn thành khóa học phải hoàn trả 100% chi phí đào tạo đã được tài trợ. Do đó, số tiền phải hoàn trả là 25.000.000 VNĐ."
- **Metrics:** faithfulness 0.25 · answer_relevancy 0.8318 · context_precision 1.0 · context_recall 1.0
- **Error Tree:**
  1. Output đúng không? → **ĐÚNG**.
  2. Context đúng không? → precision 1.0, recall 1.0.
  3. Vậy 3 trong 4 claim bị đánh là không truy được nguồn — claim nào? → "được tài trợ 25 triệu", "nghỉ việc sau 8 tháng", "số tiền là 25.000.000 VNĐ".
  4. → **Ba dữ kiện đó đến từ CÂU HỎI, không có trong tài liệu.**
- **Root cause:** câu trả lời mở đầu bằng việc thuật lại tình huống người hỏi đưa ra. Faithfulness không phân biệt "dữ kiện người dùng cung cấp" với "dữ kiện tôi bịa" — cái gì không có trong context đều bị tính là không trung thực. Quy tắc 8 trong `ANSWER_SYSTEM_PROMPT` đã cấm mở đầu bằng việc thuật lại tình huống, nhưng lần chạy này model vẫn vi phạm ở 2/20 câu (#1 và #5) → **chỉ thị prompt không đủ, cần ràng buộc cấu trúc output**.
- **Suggested fix:** ép định dạng câu trả lời hai phần — "Quy định: <trích context>" rồi mới "Áp dụng: <kết quả>" — để phần bị chấm faithfulness luôn bám context; hoặc hậu xử lý cắt bỏ mệnh đề thuật lại đề bài trước khi đưa vào RAGAS.

**Đọc ngang bottom-5:** 3/5 lỗi (#1, #2, #5) nằm ở **tầng generation với câu numeric multi-hop**, 2/5 (#3, #4) nằm ở **retrieval khi câu hỏi có nhiều ý / nhiều phiên bản**. Không có lỗi nào thuộc về chunking hay fusion — đó là lý do nếu có thêm thời gian tôi sẽ đầu tư vào query decomposition và tách phép tính khỏi LLM, chứ không tinh chỉnh thêm M1/M2.

---

## 4. `answer_relevancy`: bản vá prompt tiếng Việt đã được áp dụng cho lần chạy này

Ở các lần chạy trước, `answer_relevancy` là metric thấp nhất (0.4535 baseline / 0.6434 production) và điều tra cho thấy **phần lớn là lỗi đo, không phải lỗi hệ thống**. Truy vào `ragas/metrics/_answer_relevance.py`:

```python
QUESTION_GEN.language == "english"     # instruction + toàn bộ 4 few-shot đều tiếng Anh
```

Metric này chấm bằng cách **sinh ngược câu hỏi từ answer** rồi so cosine với câu hỏi gốc. Answer tiếng Việt + few-shot tiếng Anh → LLM sinh ra câu hỏi **tiếng Anh**. Chạy lại đúng prompt đó của RAGAS:

```
Answer (VI) → RAGAS sinh: "What is the approval requirement for taking unpaid leave of 20 days?"

cos(câu hỏi VI gốc, câu hỏi EN vừa sinh)     = 0.3900   ← chính là điểm bị chấm
cos(câu hỏi VI gốc, paraphrase tiếng Việt)   = 0.8894
cos(câu hỏi VI gốc, câu hỏi lệch chủ đề)     = 0.3464   ← gần bằng mức của bản dịch EN!
```

Tức là dưới thước đo gốc, **một câu trả lời hoàn hảo tiếng Việt bị chấm gần bằng một câu trả lời lạc đề**.

**Fix:** thay `question_generation` bằng bản tiếng Việt viết tay (`_adapt_answer_relevancy()` trong `src/m4_eval.py`, bật bằng `RAGAS_ADAPT_LANGUAGE = "vietnamese"`), few-shot dùng kiến thức phổ thông không liên quan corpus, kèm chỉ dẫn rõ rằng câu phủ định dứt khoát ("không được phép", "không nên") **vẫn là** khẳng định rõ ràng → `noncommittal = 0`.

**Lần chạy này bản vá đã thực sự chạy** (khác với các lần trước — xem mục 6, lỗi #3). Ba bằng chứng từ chính báo cáo:

| Câu từng hỏng vì artifact ngôn ngữ | Prompt EN (các lần chạy trước) | Prompt VI (lần chạy này) |
|---|---|---|
| "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?" | 0.285 | **0.7687** |
| "Khi phát hiện malware… có nên tự xử lý không?" | 0.0 (noncommittal false positive) | **rớt khỏi bottom-10** |
| "Nghỉ phép không lương 20 ngày cần ai phê duyệt?" | 0.274 | **rớt khỏi bottom-10** |

> **Khai báo minh bạch:** bản vá được áp dụng **đồng thời cho cả baseline lẫn production** nên phép so sánh ở mục 2 vẫn công bằng. Bằng chứng rằng đây là thay đổi **thước đo** chứ không phải cải thiện hệ thống: pipeline naive không đổi một dòng nào, nhưng `answer_relevancy` của nó nhảy 0.4535 → **0.7821 (+0.33)** trong khi 3 metric còn lại của chính nó chỉ xê dịch ≤ 0.016. Có thể tái lập số gốc bằng `RAGAS_ADAPT_LANGUAGE = None` trong `src/m4_eval.py`.

---

## 5. Latency breakdown

Đo trên 20 query, `reports/latency_report.json`.

| Stage | Avg (ms) | P95 (ms) | % tổng |
|---|---|---|---|
| Retrieval (BM25 + Dense + RRF) | 473,6 | 925,7 | 2,5% |
| **Rerank (cross-encoder)** | **16.829,8** | 24.958,6 | **89,8%** |
| Generation (gpt-4o-mini) | 1.436,0 | 2.842,8 | 7,7% |
| **Tổng / query** | **18.739,5** | 28.418,1 | 100% |

Chi phí build một lần: chunking 0,1s · enrichment 114 chunk 59,6s (8 luồng song song) · indexing 75,7s · load reranker 11,5s.

Ba điều đọc được:

1. **Rerank đắt gấp ~35 lần retrieval** (16.830ms vs 474ms) và một mình chiếm **89,8%** tổng latency, vì cross-encoder phải chạy 20 forward pass trên CPU, trong khi bi-encoder chỉ cần 1 lần encode query rồi so vector. Đây chính là lý do kiến trúc phải là *retrieve rộng bằng bi-encoder rồi mới rerank hẹp bằng cross-encoder* — không ai chạy cross-encoder trên cả corpus. Với 18,7 giây/query, cấu hình hiện tại **không dùng được cho chat thời gian thực**: phải chuyển reranker sang GPU, hoặc dùng `FlashrankReranker` (đã implement sẵn), hoặc bỏ rerank khi RRF đã cho khoảng cách điểm đủ lớn giữa top-1 và top-2.
2. **Rerank dao động rất mạnh giữa các query: 4.526ms – 24.959ms** dù cùng model, cùng top-20 candidate. Chênh lệch này đến từ tải máy (chạy trên CPU, máy chia sẻ với các tiến trình khác) chứ không phải từ đặc tính câu hỏi — cùng độ dài query mà lệch 5 lần. Con số tuyệt đối vì thế không so sánh được giữa các lần chạy; tỉ lệ giữa các stage mới là thứ đáng tin.
3. **Bug generation treo 176 giây đã hết.** Lần chạy trước P95 generation là **175.934,9 ms** vì SDK OpenAI mặc định timeout 600s; sau khi đặt `timeout=60, max_retries=3` thì P95 xuống **2.842,8 ms** — giảm 62 lần (mục 6, lỗi #4). Generation từ chỗ là stage đắt nhất (63,8%) trở thành stage rẻ thứ hai (7,7%).

---

## 6. Năm lỗi đã gặp và cách sửa

| # | Lỗi | Triệu chứng | Nguyên nhân | Fix |
|---|---|---|---|---|
| 1 | LLM từ chối trả lời | 4/20 câu trả "Không tìm thấy" dù `context_recall = 1.0`; faithfulness và relevancy đều = 0 | Prompt cấm suy luận nên model không dám làm phép nhân 85% | Viết lại prompt: được phép tính toán trên số liệu có trong context, chỉ từ chối khi context hoàn toàn không liên quan. **Đã hết** ở lần chạy này — 0/20 câu trả "Không tìm thấy" |
| 2 | Rò rỉ test set vào prompt | — (tự phát hiện khi rà lại) | 3 ví dụ tôi viết trong `ANSWER_SYSTEM_PROMPT` chứa nguyên đáp án câu 1, 14, 18 → LLM trả lời đúng mà không cần retrieve | Thay bằng ví dụ trừu tượng dạng `"<đối tượng> là <giá trị lấy từ context>"`, chạy lại toàn bộ |
| 3 | `adapt()` của RAGAS hỏng | `ValidationError: output in example 1 is not in valid json format` | `adapt()` cache prompt đã dịch ra `~/.cache/ragas/`, nhưng ghi field `output` dưới dạng chuỗi có json fences → lần load sau parse lỗi, im lặng rơi về prompt tiếng Anh | Bỏ `adapt()` động, viết tay prompt tiếng Việt trong code — deterministic, không tốn LLM call, không phụ thuộc cache máy. **Đã xác nhận chạy đúng** ở lần này (mục 4) |
| 4 | Một query treo 176 giây | P95 generation = 175.934ms; answer của câu #17 là **nguyên văn context** `[Nguồn: tam_ung.md] # Chính sách tạm ứng…` | SDK OpenAI mặc định timeout 600s. Request treo → exception → nhánh fallback `answer = contexts[0]` dán context làm câu trả lời, khiến RAGAS đi chấm context như thể đó là answer | `OpenAI(timeout=60, max_retries=3)`; fallback trả câu "Không tìm thấy thông tin." thay vì dán context. **Đã xác nhận hết**: P95 generation 175.934,9ms → **2.842,8ms** |
| 5 | `FileExistsError [WinError 183]` | `main.py` crash ở bước move report, bảng so sánh không in ra | Trên Windows `os.rename()` ném lỗi nếu đích đã tồn tại; POSIX thì ghi đè im lặng | `os.replace()`. **Đã xác nhận hết**: bảng so sánh Step 3 in đủ |

---

## 7. Nếu có thêm một giờ

Xếp theo tỉ lệ điểm-thu-được / công-bỏ-ra, dựa trên bottom-5 ở mục 3:

1. **Tách phép tính ra khỏi LLM** — sửa #1 (đáp án sai một chữ số: 500.000 thay vì 50.000). Cho model trả về công thức có cấu trúc rồi tính bằng Python. Đây là lỗi duy nhất trong bottom-5 khiến hệ thống đưa ra **thông tin sai cho người dùng**, nên ưu tiên cao hơn mọi tối ưu điểm số.
2. **Query decomposition cho câu hỏi nhiều ý** — sửa trực tiếp #4 (`context_recall` 0.5 vì `bang_luong_2024.md` không lọt top khi query hỏi cả phép năm lẫn lương). Tách sub-query, retrieve riêng, đảm bảo mỗi ý có tối thiểu 1 context, rồi mới hợp nhất. Đây là lỗi retrieval đắt nhất còn lại.
3. **Version-aware metadata** — sửa `context_precision` = 0.5 ở "Nhân viên được nghỉ bao nhiêu ngày phép năm?" và "Thâm niên bao nhiêu năm…", 0.75 ở "Muốn mua thiết bị 55 triệu…", đồng thời sửa `context_recall` = 0.5 ở #3 (MFA). Enrichment đã gọi LLM cho từng chunk rồi, chỉ cần thêm 2 trường `version` + `effective_date` vào JSON trả về → chi phí gần bằng 0. Lưu ý bài học từ #3: **gắn nhãn** phiên bản chứ không loại thẳng bản cũ.
4. **Giảm latency rerank** — 89,8% tổng thời gian là không chấp nhận được cho production. `max_length=512` cho CrossEncoder, chuyển GPU, hoặc bỏ rerank khi RRF đã tách bạch top-1/top-2.
5. **Cache enrichment xuống đĩa theo hash nội dung chunk** — hiện mỗi lần build lại gọi LLM sinh lại summary/HyQA/context-line, nên embedding khác đi và thứ tự retrieve đổi theo: index của production **không tất định** (đo được ở mục 8 — baseline giữ nguyên `context_precision`/`recall` tới từng chữ số qua 2 lần chạy, còn production thì không). Cache vừa làm build tái lập được, vừa cắt 50 giây và toàn bộ chi phí API mỗi lần chạy lại.

---

## 8. Thực nghiệm bổ sung: siết prompt hai phần — cơ chế đúng, tổng thể không cải thiện

Sau khi viết xong mục 3, tôi thử đúng suggested_fix của #1 và #5: đổi quy tắc 8 từ lời khuyên ("đừng thuật lại tình huống") thành **khuôn bắt buộc hai phần** (câu đầu trích nguyên văn quy định từ context, câu sau mới là kết quả áp dụng), và thêm vào quy tắc 3 yêu cầu tự tính lại rồi đối chiếu trước khi kết luận. Chạy lại toàn bộ pipeline. Report của lần chạy này lưu ở `reports/ablation_prompt_2phan/`.

**Cơ chế hoạt động đúng như dự đoán.** Câu tạm ứng (#1) đổi từ

> "Nhân viên tạm ứng 15 triệu VNĐ và thanh toán sau 20 ngày sẽ bị phạt 2%/tháng…" *(mở đầu bằng dữ kiện của người hỏi)*

thành

> "Theo Chính sách tạm ứng, khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí **2%/tháng** trên số tiền chưa hoàn ứng. Do đó mức phạt là…"

và faithfulness của riêng câu đó tăng **0.1111 → 0.3333**.

**Nhưng hai điều làm thực nghiệm thất bại.**

1. *Phép tính vẫn sai, chỉ sai kiểu khác*: lần này ra **600.000 VNĐ** thay vì 500.000 — đáp án đúng vẫn là 50.000. Lệnh "tự tính lại và đối chiếu" không cứu được `gpt-4o-mini`. Điều này **xác nhận** suggested_fix ở mục 7.1: phép tính phải ra khỏi LLM, không vá được bằng prompt.
2. *Điểm tổng giảm*, và quan trọng hơn — **mức giảm nằm trong biên nhiễu giữa hai lần chạy**:

| Metric | Baseline run 1 | Baseline run 2 | Δ baseline | Production run 1 | Production run 2 | Δ production |
|---|---|---|---|---|---|---|
| Faithfulness | 0.8405 | 0.8333 | −0.0072 | 0.8414 | 0.8125 | −0.0289 |
| Answer Relevancy | 0.7821 | 0.7013 | **−0.0808** | 0.8597 | 0.8393 | −0.0204 |
| Context Precision | 0.9250 | 0.9250 | **0.0000** | 0.8958 | 0.8792 | −0.0166 |
| Context Recall | 0.9000 | 0.9000 | **0.0000** | 0.9500 | 0.9250 | −0.0250 |

**Cột "Δ baseline" là cột đáng đọc nhất trong cả báo cáo này.** Baseline không đổi một dòng code nào giữa hai lần chạy, vậy mà:

- `answer_relevancy` xê dịch **0.081** — **lớn hơn** chính con số Δ = +0.0776 mà mục 2 đang dùng để chứng minh production tốt hơn baseline;
- `faithfulness` xê dịch 0.007 — cùng bậc với Δ = +0.0009 ở mục 2, tức con số đó **hoàn toàn vô nghĩa**;
- nhưng `context_precision` và `context_recall` **giống hệt nhau tới từng chữ số**.

Hai loại metric ứng xử khác hẳn nhau, và lý do rất cụ thể: `context_precision`/`context_recall` chỉ phụ thuộc câu hỏi, context và ground truth — với baseline thì dense retrieval là **tất định**, nên chấm lại cho đúng con số cũ. Còn `faithfulness`/`answer_relevancy` phụ thuộc câu trả lời do LLM sinh (temperature > 0) rồi lại được một LLM khác chấm — hai tầng ngẫu nhiên chồng lên nhau.

Điều đó cũng giải thích vì sao **production lại xê dịch cả context_precision/recall trong khi baseline thì không**: pipeline production gọi LLM để enrichment ở mỗi lần build, nên summary/HyQA/context-line sinh ra khác nhau → embedding khác → **thứ tự retrieve khác**. Nói cách khác, index của production tự nó đã không tất định. Đây là lỗi thiết kế thật của tôi, không phải đặc tính của kiến trúc: enrichment nên được **cache xuống đĩa theo hash nội dung chunk**, vừa làm build tái lập được, vừa tiết kiệm ~50 giây và toàn bộ chi phí API cho mỗi lần chạy lại.

**Quyết định:** revert prompt về đúng bản đã sinh ra số liệu ở mục 2, nộp kết quả run 1, giữ report run 2 lại làm bằng chứng. Lý do không phải "run 1 điểm cao hơn" mà là: với n = 20 câu và một LLM làm giám khảo, tôi **không có đủ dữ liệu để kết luận** bản prompt nào tốt hơn — mà giữa hai thứ không phân biệt được thì chọn bản đã được phân tích kỹ trong mục 3.

**Hệ quả cho cách đọc mục 2:** trong 4 con số Δ, chỉ `context_recall` (+0.0500) là vượt rõ biên nhiễu; `answer_relevancy` (+0.0776) nằm sát biên; `faithfulness` (+0.0009) và `context_precision` (−0.0292) thì **không kết luận được gì**. Muốn khẳng định chắc chắn thì phải chạy mỗi cấu hình 5 lần rồi báo mean ± std, hoặc nâng test set lên 100+ câu — đây là việc đầu tiên tôi sẽ làm cho ChillGuys (xem reflection, Phần 3, Tuần 1).

---

## 9. Khai báo phương pháp

Ba điều người chấm nên biết để đánh giá đúng con số:

1. **`RERANK_TOP_K` đổi 3 → 10**, khác với "top-20 → top-3" mô tả trong ASSIGNMENT. Lý do: mỗi file corpus chỉ ~800 ký tự nên `parent_size=2048` khiến **1 document = 1 parent**; top-3 child thường trỏ về cùng một parent, sau dedupe chỉ còn **1 context duy nhất** — câu multi-hop cần dữ kiện từ 2 file sẽ hỏng. Đã kiểm chứng bằng test offline trước khi đổi.
2. **`answer_relevancy` dùng prompt tiếng Việt viết tay** thay cho prompt tiếng Anh gốc của RAGAS, **áp dụng cho cả baseline lẫn production** trong lần chạy này. Chi tiết, bằng chứng và cách tắt: mục 4.
3. **Prompt được tinh chỉnh sau khi đọc failure trên chính 20 câu dùng để chấm** — không có held-out set. Đây là quy trình mà lab yêu cầu (failure analysis → suggested_fix → sửa), nhưng đồng nghĩa điểm số có phần lạc quan so với dữ liệu chưa từng thấy.
4. **Số liệu nộp là của một lần chạy duy nhất (run 1), không phải trung bình nhiều lần.** Tôi có chạy lần thứ hai với một bản prompt khác và đã khai báo đầy đủ ở mục 8, kèm report gốc trong `reports/ablation_prompt_2phan/`. Biên nhiễu đo được giữa hai lần chạy là ~0.08 cho `answer_relevancy` và ~0.03 cho `faithfulness` — cần đọc mọi Δ trong báo cáo này với sai số đó trong đầu.
