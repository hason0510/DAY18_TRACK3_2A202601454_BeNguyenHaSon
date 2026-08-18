# Individual Reflection — Lab 18: Production RAG

**Tên:** Bế Nguyễn Hà Sơn · **Mã:** 2A202601454
**Module phụ trách:** M1 → M5 (bài cá nhân, implement toàn bộ)
**Ngày:** 18/08/2026

---

## Phần 1 — Mapping bài giảng vào code

Toàn bộ số liệu dưới đây lấy từ lần chạy thật, không phải ước lượng.
Bảng A/B chunking: chạy `python src/m1_chunking.py` trên corpus 26 tài liệu (21.014 ký tự).

| Chiến lược | Chunks | Avg | Min | Max |
|---|---|---|---|---|
| basic (baseline) | 51 | 410 | 273 | 565 |
| semantic | 87 | 239 | 99 | 647 |
| hierarchical (child) | 109 | 191 | 3 | 256 |
| structure-aware | 106 | 197 | 87 | 789 |

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên cosine giữa 2 câu liền kề tạo **87 chunk** so với 51 của basic — tức semantic **chia nhỏ hơn** chứ không gộp lại như tôi tưởng ban đầu. Lý do: header markdown ("## Nghỉ phép năm") có embedding rất xa câu nội dung ngay sau nó, nên mỗi header thành một điểm cắt. Tôi phải thêm `_merge_small_groups()` gộp các mảnh < 100 ký tự, nếu không thì min length rơi xuống ~15 ký tự và chunk mất hết ngữ cảnh. |
| Hierarchical / small-to-big | M1 + pipeline | `chunk_hierarchical()`, `_expand_to_parents()` | Corpus 26 tài liệu → **26 parent / 114 child**. Mỗi file chỉ ~800 ký tự nên `parent_size=2048` khiến **1 document = 1 parent**. Hệ quả bất ngờ: top-3 child sau rerank hầu như luôn trỏ về *cùng một* parent, dedupe xong chỉ còn **1 context duy nhất** → câu multi-hop cần 2 file bị hỏng. Phải nâng `RERANK_TOP_K` lên 10 và bù thêm từ kết quả hybrid mới đủ 4 nguồn khác nhau. Đây là bài học: tham số chunking chỉ có nghĩa khi đặt cạnh kích thước tài liệu thật. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Split theo `^#{1,3}` giữ nguyên header trong text nên chunk tự mang ngữ cảnh. Max length 789 vs 256 của hierarchical — section dài không bị cắt, tốt cho bảng biểu nhưng làm chunk không đều. |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép"). Nếu giữ nguyên thì corpus có token `nghỉ_phép` còn query "nghỉ phép" tách thành 2 token → BM25 **không khớp gì cả**. Phải `replace("_", " ")`. Giá trị thật của bước segment không phải giữ từ ghép mà là chuẩn hoá ranh giới từ và bỏ dấu câu. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng **thứ hạng**, không dùng score gốc: `score(d) = Σ 1/(60 + rank + 1)`. Nhờ vậy không phải normalize giữa BM25 (0..∞) và cosine (0..1) — vốn là bài toán không có lời giải đẹp. Doc xuất hiện ở cả hai list được cộng dồn nên tự động nổi lên đầu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Latency đo thật: **16.829,8 ms/query**, chiếm **89,8%** tổng thời gian, gấp **~35 lần** retrieval (473,6 ms). Bi-encoder encode query và doc riêng rồi so vector; cross-encoder đẩy cả cặp qua một forward pass nên bắt được quan hệ token-level, đổi lại là O(n) forward pass. Con số này giải thích chính xác vì sao kiến trúc phải là *retrieve rộng bằng bi-encoder → rerank hẹp bằng cross-encoder*, không ai chạy cross-encoder trên cả corpus. Nó cũng dao động 4.526–24.959 ms giữa các query dù cùng model và cùng 20 candidate — đó là nhiễu do chạy CPU trên máy chia sẻ, nên tỉ lệ giữa các stage mới là con số đáng tin, không phải giá trị tuyệt đối. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness 0.8414 · Answer Relevancy 0.8597 · Context Precision 0.8958 · Context Recall 0.9500 — **4/4 metric ≥ 0.75**. Ở các lần chạy trước answer_relevancy là metric thấp nhất (0.6434), và điều tra cho thấy **phần lớn không phải lỗi hệ thống** mà là artifact ngôn ngữ của bộ chấm — sau khi vá prompt tiếng Việt, nó lên 0.8597 và metric thấp nhất giờ là faithfulness (0.8414), mất điểm ở đúng nhóm câu numeric multi-hop. Xem Phần 2 mục 4. Điều học được lớn nhất: metric cũng là một hệ thống có bug, phải audit nó trước khi tin. |
| Failure analysis / Diagnostic Tree | M4 | `failure_analysis()` + `DIAGNOSTIC_TREE` | Map metric thấp nhất → chẩn đoán → hành động sửa. Hữu ích thật: chính bảng này chỉ ra 4/20 câu trả "Không tìm thấy" dù `context_recall = 1.0`, tức lỗi ở generation chứ không ở retrieval — nếu chỉ nhìn điểm tổng thì tôi đã đi sửa nhầm chỗ (chunking/search) và tốn cả buổi vô ích. |
| Contextual embeddings (Anthropic) | M5 | `contextual_prepend()`, `_enrich_single_call()` | Prepend một câu mô tả chunk nằm ở đâu trong tài liệu trước khi embed. Với corpus có nhiều file gần trùng nhau (v2023/v2024, mật khẩu v1/v2), câu context này là thứ giúp chunk con phân biệt được mình thuộc phiên bản nào. |
| Cost optimization cho enrichment | M5 | `_enrich_single_call()` | Combined mode: 1 API call/chunk trả về đồng thời summary + 3 câu hỏi + context line + metadata, thay vì 4 call riêng → **giảm 75% số call**. Thêm `ThreadPoolExecutor(8)`: 114 chunk enrich xong trong **59,6 giây** thay vì ~2 phút tuần tự. |

---

## Phần 2 — Khó khăn và cách giải quyết

### 1. `ModuleNotFoundError: No module named pytest`

```
D:\...\.venv\Scripts\python.exe: No module named pytest
```

`requirements.txt` không liệt kê `pytest` dù RUBRIC chấm bằng `pytest tests/ -v`. Sửa: `pip install pytest` và bổ sung `pytest>=8.0` vào `requirements.txt` để người chấm không vấp lại.
**Thời gian:** 2 phút.

### 2. `FileExistsError: [WinError 183]` — pipeline chạy xong vẫn báo lỗi

```
File "main.py", line 39, in main
    os.rename(f, f"reports/{f}")
FileExistsError: [WinError 183] Cannot create a file when that file already exists:
'ragas_report.json' -> 'reports/ragas_report.json'
```

Lỗi chỉ xuất hiện **từ lần chạy thứ hai**. Trên Windows `os.rename()` từ chối ghi đè file đã tồn tại, trong khi trên POSIX nó ghi đè im lặng — code gốc viết theo giả định POSIX. Toàn bộ pipeline đã chạy xong và report đã lưu, chỉ bước dọn file là chết, nên bảng so sánh Step 3 không bao giờ được in.
**Sửa:** `os.replace()` — ghi đè nhất quán trên mọi nền tảng.
**Thời gian:** 5 phút. **Bài học:** đọc kỹ traceback trước khi hoảng — dòng cuối chỉ đúng vào `os.rename`, không liên quan gì tới RAG.

### 3. `ValidationError` — RAGAS `adapt()` tự phá cache của chính nó

```
pydantic.v1.error_wrappers.ValidationError: 1 validation error for Prompt
__root__
  output in example 1 is not in valid json format: Expecting value: line 1 column 1 (char 0)
```

Đây là lỗi khó nhất và cũng thú vị nhất. `answer_relevancy.adapt(language="vietnamese")` **chạy đúng ở lần đầu**, nhưng nó cache prompt đã dịch xuống `~/.cache/ragas/vietnamese/question_generation.json`, và khi ghi thì bọc field `output` trong ```json fences — tức ghi ra thứ mà chính nó không đọc lại được. Từ lần chạy thứ hai trở đi, `_load()` ném ValidationError, `try/except` của tôi nuốt lỗi và **âm thầm rơi về prompt tiếng Anh**.

Hệ quả: tôi tưởng đã sửa xong answer_relevancy, chạy `main.py`, thấy chỉ nhích từ 0.6180 lên 0.6434 và suýt kết luận "adapt không hiệu quả". Thực tế là **bản vá chưa từng chạy**. Khi nó thực sự chạy ở lần cuối, answer_relevancy lên **0.8597** — đúng mức mà phép đo tay đã dự báo.

**Cách debug:** viết script độc lập gọi lại đúng `adapt()` ngoài `try/except` để thấy traceback thật, thay vì tin vào cảnh báo mà mình tự in ra.
**Sửa:** bỏ hẳn `adapt()` động, viết tay prompt tiếng Việt ngay trong `src/m4_eval.py` — deterministic, không tốn LLM call, không phụ thuộc trạng thái cache của máy.
**Bài học lớn nhất của cả lab:** một `except Exception` nuốt lỗi rồi in cảnh báo hiền lành nguy hiểm hơn là để chương trình chết, vì nó biến bug thành "kết quả thí nghiệm" và mình sẽ đi rút ra kết luận sai từ đó.
**Thời gian:** ~40 phút.

### 4. Kiến thức thiếu: tôi không biết RAGAS chấm `answer_relevancy` như thế nào

Ban đầu tôi thấy answer_relevancy 0.4654 và mặc định là "câu trả lời chưa tốt", rồi đi sửa prompt sinh answer. Sửa xong vẫn thấp. Chỉ khi mở `ragas/metrics/_answer_relevance.py` đọc mới hiểu metric này **sinh ngược câu hỏi từ answer rồi so cosine với câu hỏi gốc**, và toàn bộ few-shot của nó là tiếng Anh (`QUESTION_GEN.language == "english"`).

Đo thực nghiệm trên câu "Nghỉ phép không lương 20 ngày cần ai phê duyệt?":

```
RAGAS sinh ra: "What is the approval requirement for taking unpaid leave of 20 days?"

cos(câu hỏi VI, câu hỏi EN vừa sinh)  = 0.3900   ← chính là điểm bị chấm
cos(câu hỏi VI, paraphrase tiếng Việt) = 0.8894
cos(câu hỏi VI, câu hỏi lệch chủ đề)   = 0.3464   ← xấp xỉ mức của bản dịch EN
```

Một câu trả lời **đúng hoàn hảo** bị chấm gần bằng một câu **lạc đề**, chỉ vì khác ngôn ngữ. Tệ hơn, câu trả lời phủ định "không nên tự ý xử lý malware" bị judge gắn `noncommittal = 1` (né tránh) → nhân 0 → điểm **0.0** tuyệt đối dù 3 metric còn lại đều 1.0.

**Cách bổ sung kiến thức:** đọc thẳng source của thư viện thay vì chỉ đọc docs. Sau khi thay bằng prompt tiếng Việt viết tay có chỉ dẫn rõ về câu phủ định, đo lại đúng 3 câu đó: **0.285 → 0.7661**, **0.0 → 0.9333**, **0.274 → 0.9492**.

Lần chạy cuối xác nhận trên toàn bộ 20 câu: câu "thâm niên" đạt **0.7687** (khớp phép đo tay), hai câu còn lại **rớt hẳn khỏi bottom-10**. Bằng chứng mạnh nhất là ở phía baseline: pipeline naive không đổi một dòng nào mà answer_relevancy nhảy **0.4535 → 0.7821 (+0.33)**, trong khi 3 metric còn lại của chính nó xê dịch ≤ 0.016 — tức toàn bộ mức tăng đó là **thay đổi thước đo**, không phải cải thiện hệ thống.
**Thời gian:** ~50 phút.

### 5. Tự phát hiện mình làm rò rỉ test set vào prompt

Khi sửa prompt sinh answer, tôi lấy ví dụ minh hoạ từ chính các câu đang debug. Kết quả là `ANSWER_SYSTEM_PROMPT` chứa nguyên đáp án của 3 câu trong `test_set.json`: "nghỉ **3 ngày** khi kết hôn" (câu 1), "**85% × 20.000.000 = 17.000.000**" (câu 18), "hoàn trả **100% chi phí, tức 25.000.000 VNĐ**" (câu 14). Nghĩa là với 3 câu đó, LLM trả lời đúng **mà không cần retrieve gì**.

Không có thông báo lỗi nào cả — chỉ phát hiện khi rà lại toàn bộ thay đổi để trả lời câu hỏi "sửa thế có phạm quy không". **Sửa:** thay bằng ví dụ trừu tượng dạng `"<đối tượng> là <giá trị lấy từ context>"`, grep lại toàn bộ `src/` để chắc không còn con số nào của corpus, và chạy lại từ đầu vì kết quả cũ không dùng được.
**Bài học:** khi tối ưu theo số, ranh giới giữa "sửa hệ thống" và "vá vào đáp án" rất dễ vượt qua mà không nhận ra.

### 6. Một query treo 176 giây và câu trả lời là nguyên văn context

Bảng latency cho P95 generation = **175.934,9 ms**. Truy vào report thì câu #17 có answer là `[Nguồn: tam_ung.md] # Chính sách tạm ứng > Phiên bản: 1.0...` — tức nguyên văn context.

Nguyên nhân: SDK OpenAI mặc định `timeout=600s`. Một request treo, cuối cùng ném exception, rơi vào nhánh `except` của scaffold là `answer = contexts[0]` → **RAGAS đi chấm context như thể đó là câu trả lời**. Faithfulness được 1.0 (answer trùng context) nhưng answer_relevancy 0.334 — bug này vừa giấu mình vừa làm nhiễu cả hai metric.
**Sửa:** `OpenAI(timeout=60, max_retries=3)`, và fallback trả câu "Không tìm thấy thông tin." thay vì dán context — thà nhận điểm thấp còn hơn che mất lỗi. **Kết quả đo lại:** P95 generation **175.934,9 ms → 2.842,8 ms** (giảm 62 lần); generation từ stage đắt nhất (63,8% tổng latency) xuống còn 7,7%, và rerank lộ ra là nút cổ chai thật sự với 89,8%.

### 7. Những cái bẫy version tránh được nhờ đọc trước

- `qdrant-client 1.19`: `recreate_collection()` đã deprecated → dùng `collection_exists()` + `delete_collection()` + `create_collection()`; `search()` → `query_points()`.
- `ragas 0.1.22`: default LLM vẫn trỏ `gpt-3.5-turbo-16k` (OpenAI đã ngừng phục vụ) → phải truyền tường minh `llm=ChatOpenAI("gpt-4o-mini")` và `embeddings=OpenAIEmbeddings(...)`.
- PDF: `pypdf` trả **0 ký tự** cho `Nghi_dinh_13-2023.pdf`, `PyMuPDF` trả **38 ký tự** — nhưng 38 ký tự đó chỉ là dòng metadata "Cơ quan phát hành: Văn phòng Chính phủ", không phải 39 trang nội dung. Nếu chỉ kiểm tra `if text:` thì file rác này lọt vào index thành một parent vô nghĩa. Phải đặt ngưỡng `MIN_PDF_TEXT_CHARS = 200`.



---

## Phần 3 — Action Plan cho project cá nhân

### Project: ChillGuys — trợ lý hỏi đáp tài liệu nội bộ tiếng Việt

**Hiện tại**
- RAG pipeline: chunking cố định theo số ký tự + dense-only search + top-k thô, không rerank, không đánh giá tự động.
- Known issues: (a) trả lời sai khi tài liệu có nhiều phiên bản; (b) câu hỏi cần ghép dữ kiện từ 2 tài liệu thì trả lời thiếu; (c) không có cách nào biết một thay đổi làm hệ thống tốt lên hay tệ đi ngoài việc thử tay vài câu.

### Plan áp dụng

1. **Chunking — hierarchical parent/child, nhưng đo kích thước tài liệu trước khi chọn tham số.**
   Bài học đắt nhất từ lab: `parent_size=2048` trên corpus mà mỗi file chỉ ~800 ký tự khiến 1 document = 1 parent và làm sập tính đa dạng của context. Việc đầu tiên phải làm là vẽ histogram độ dài tài liệu, rồi mới đặt `parent_size ≈ trung vị độ dài tài liệu` và `child_size ≈ 1/8 parent`. Không copy tham số từ lab.

2. **Search — hybrid BM25 + Dense hợp nhất bằng RRF.**
   Tài liệu nội bộ đầy mã số, tên viết tắt, số hiệu văn bản — dense search rất kém với những token này còn BM25 thì bắt chính xác. RRF cho phép ghép mà không cần normalize score. Bắt buộc có `segment_vietnamese()` + `replace("_", " ")` cho nhánh BM25.

3. **Reranking — có, `bge-reranker-v2-m3`, nhưng phải tính chi phí trước.**
   Đo được: **16,8 giây/query** trên CPU, chiếm **89,8%** tổng latency (18,7 giây/query cho cả pipeline). Với ứng dụng chat thời gian thực thì con số này không chấp nhận được. Kế hoạch: chạy reranker trên GPU, hoặc dùng flashrank (đã implement sẵn `FlashrankReranker` làm phương án nhẹ), hoặc bỏ qua rerank khi RRF đã cho khoảng cách điểm đủ lớn giữa top-1 và top-2.

4. **Evaluation — RAGAS làm nền, nhưng phải audit metric trước khi tin.**
   Cụ thể với tiếng Việt: kiểm tra `answer_relevancy` có sinh câu hỏi ngược đúng ngôn ngữ không (lab này mất ~0.33 điểm oan vì chuyện đó, đo được bằng cách chạy lại baseline không đổi code), và tách riêng nhóm câu hỏi numeric multi-hop để chấm bằng exact-match thay vì faithfulness — faithfulness phạt suy luận số học kể cả khi suy luận đúng (câu #2 và #5 trong failure analysis). Đồng thời phải kiểm tra **kết quả số học** bằng code chứ không tin LLM: câu #1 của bottom-5 model tính ra 500.000 VNĐ trong khi đáp án là 50.000 VNĐ — sai một chữ số mà không metric nào bắt được. Bổ sung một bộ **held-out** không dùng để tinh chỉnh prompt.

5. **Enrichment — contextual prepend + auto metadata, gọi gộp 1 call/chunk.**
   Ưu tiên `version` và `effective_date` trong metadata: đây là fix trực tiếp cho vấn đề nhức nhối nhất của tài liệu nội bộ — chính sách cũ và mới cùng tồn tại trong index. Lab này `context_precision` bị kéo xuống 0.5 ở 2 câu và 0.75 ở 1 câu chỉ vì retrieve cả cặp v2023/v2024. Nhưng chú ý mặt trái: câu hỏi về MFA lại mất `context_recall` (0.5) vì **không** lấy được bản mật khẩu v1.0 mà ground truth có nhắc tới — nên giải pháp đúng là **gắn nhãn** phiên bản để prompt tự quyết, không phải loại thẳng bản cũ khỏi index.

### Timeline

| Tuần | Việc | Tiêu chí xong |
|---|---|---|
| Tuần 1 | Dựng bộ đánh giá trước, không code pipeline: 50–100 cặp Q&A từ ticket thật, chia train/held-out, dựng `evaluate_ragas()` + audit metric trên tiếng Việt, **chạy baseline 3 lần để đo biên nhiễu** | Có baseline số của hệ thống hiện tại; biết chắc metric không bị artifact ngôn ngữ; **có ngưỡng "Δ bao nhiêu thì mới là thật"** |
| Tuần 2 | Đo phân bố độ dài tài liệu → chọn tham số → hierarchical chunking + parent expansion; thêm hybrid BM25+RRF | `context_recall` tăng so với baseline tuần 1 |
| Tuần 3 | Enrichment (contextual + version metadata) + lọc bản superseded; thêm reranking kèm đo latency từng bước | `context_precision` tăng; có bảng latency để quyết định giữ hay bỏ rerank |
| Tuần 4 | Tối ưu prompt sinh answer trên tập train, **đánh giá cuối trên held-out**; viết failure analysis; chốt cấu hình | Chênh lệch điểm train vs held-out < 0.05 (nếu lớn hơn nghĩa là đã overfit prompt) |

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|---|---|---|
| Hiểu bài giảng | 4 | Nắm được vì sao RRF không cần normalize và vì sao cross-encoder chỉ dùng ở tầng rerank — có số liệu tự đo để chứng minh |
| Code quality | 4 | 0 TODO còn lại, mọi nhánh lỗi đều có fallback; trừ điểm vì `except Exception` quá rộng ở M4 đã che mất bug `adapt()` gần một tiếng |
| Problem solving | 4 | Truy được ba lỗi mà không có thông báo rõ ràng: `adapt()` hỏng cache, artifact ngôn ngữ của metric, và query treo 176s làm answer thành context |
| Trung thực học thuật | 4 | Tự phát hiện và sửa rò rỉ test set vào prompt; khai báo đầy đủ 4 thay đổi phương pháp ở `failure_analysis.md` mục 9; và giữ lại nguyên vẹn report của lần chạy thất bại (mục 8) thay vì lặng lẽ chọn con số đẹp hơn |
