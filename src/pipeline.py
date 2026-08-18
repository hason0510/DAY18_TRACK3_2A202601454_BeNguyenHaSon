from __future__ import annotations

"""Production RAG Pipeline — ghép M1+M2+M3+M4+M5."""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, MAX_CONTEXTS, OPENAI_API_KEY

# Child chunk là đơn vị retrieve (precision), parent là đơn vị trả về (context).
# parent_id chỉ unique trong 1 document nên key phải kèm source.
_PARENT_MAP: dict[str, str] = {}
_BUILD_TIMINGS: dict[str, float] = {}
_QUERY_TIMINGS: list[dict] = []

ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu quy định nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context được cung cấp.

Quy tắc:
1. Trả lời trực tiếp bằng câu hoàn chỉnh, nhắc lại chủ thể của câu hỏi.
   Hỏi "<đối tượng> là bao nhiêu?" → "<đối tượng> là <giá trị lấy từ context>."
   KHÔNG trả lời cụt lủn chỉ một con số trần.
2. Nêu đủ con số, mốc thời gian, cấp phê duyệt, điều kiện kèm theo nếu context có.
3. ĐƯỢC PHÉP cộng/trừ/nhân/chia trên các số liệu CÓ SẴN trong context để suy ra đáp án.
   Ghi cách tính GỌN trong một mệnh đề dạng "<tỉ lệ> × <mức gốc> = <kết quả>".
   Không trình bày nhiều bước suy luận dài dòng.
4. Nếu context chứa nhiều phiên bản của cùng một chính sách (v1/v2, v2023/v2024):
   dùng phiên bản MỚI NHẤT (số hiệu/năm lớn hơn) làm câu trả lời chính,
   và ghi chú ngắn rằng phiên bản cũ đã bị thay thế. Tên file nguồn ghi ở đầu mỗi context.
5. Chỉ trả lời "Không tìm thấy thông tin trong tài liệu." khi context HOÀN TOÀN không liên quan.
   Nếu context trả lời được một phần, hãy trả lời phần đó thay vì từ chối.
6. Không bịa thêm thông tin ngoài context.
7. Độ dài: 1-4 câu. Mỗi khẳng định phải truy nguyên được về context.
8. Các dữ kiện do NGƯỜI HỎI đưa ra (số tiền, số ngày, số năm thâm niên...) chỉ dùng để
   tra bảng và tính toán — đừng chép lại chúng thành một câu khẳng định riêng, vì chúng
   KHÔNG nằm trong tài liệu. Hãy phát biểu quy định trước rồi mới tới kết quả áp dụng,
   đừng mở đầu bằng việc thuật lại tình huống của người hỏi."""


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    _PARENT_MAP.clear()
    _BUILD_TIMINGS.clear()
    _QUERY_TIMINGS.clear()

    # Step 1: Load & Chunk (M1) — hierarchical: index child, trả parent
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        source = doc["metadata"].get("source", "")
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for p in parents:
            _PARENT_MAP[f"{source}::{p.metadata['parent_id']}"] = p.text
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata,
                             "parent_id": child.parent_id,
                             "parent_key": f"{source}::{child.parent_id}"},
            })
    _BUILD_TIMINGS["chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} child chunks / {len(_PARENT_MAP)} parents "
          f"from {len(docs)} documents ({_BUILD_TIMINGS['chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        _BUILD_TIMINGS["enrichment"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({_BUILD_TIMINGS['enrichment']:.1f}s)", flush=True)
    else:
        _BUILD_TIMINGS["enrichment"] = time.time() - t0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    _BUILD_TIMINGS["indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({_BUILD_TIMINGS['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()                      # load sẵn để không tính vào latency query đầu
    _BUILD_TIMINGS["reranker_load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({_BUILD_TIMINGS['reranker_load']:.1f}s)", flush=True)

    return search, reranker


def _expand_to_parents(reranked, fallback_results) -> list[str]:
    """Child → parent expansion: retrieve bằng child (chính xác), trả parent (đủ ngữ cảnh).

    Đây là điểm mấu chốt của hierarchical chunking: child 256 ký tự đủ để match
    query nhưng thiếu ngữ cảnh cho LLM trả lời câu multi-hop/numeric.

    Corpus lab này mỗi file chỉ ~800 ký tự nên 1 document = 1 parent → nhiều child
    top-rank hay trỏ về CÙNG một parent. Vì vậy phải duyệt sâu (RERANK_TOP_K=10)
    rồi mới lấy đủ MAX_CONTEXTS parent KHÁC NHAU; nếu vẫn thiếu thì bù bằng
    kết quả hybrid còn lại — câu multi-hop cần dữ kiện từ 2 file khác nhau.
    """
    contexts, seen = [], set()

    for r in list(reranked or []) + list(fallback_results or []):
        meta = getattr(r, "metadata", {}) or {}
        parent_text = _PARENT_MAP.get(meta.get("parent_key", ""), "") or r.text
        if parent_text in seen:                 # nhiều child cùng 1 parent → chỉ lấy 1 lần
            continue
        seen.add(parent_text)
        source = meta.get("source", "")
        # Gắn tên file để LLM phân biệt được v2023 vs v2024, v1 vs v2
        contexts.append(f"[Nguồn: {source}]\n{parent_text}" if source else parent_text)
        if len(contexts) >= MAX_CONTEXTS:
            break

    return contexts


_LLM_CLIENT = None


def _get_llm_client():
    """1 client dùng chung, có timeout + retry.

    Mặc định SDK để timeout 600s: một request treo sẽ ngốn 10 phút rồi mới lỗi.
    Lần chạy trước đúng 1 query mất 175.9s rồi ném exception, khiến answer rơi về
    contexts[0] — tức RAGAS đi chấm nguyên văn context như thể đó là câu trả lời.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from openai import OpenAI
        _LLM_CLIENT = OpenAI(timeout=60.0, max_retries=3)
    return _LLM_CLIENT


def _generate_answer(query: str, context_str: str) -> str:
    try:
        resp = _get_llm_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # KHÔNG trả contexts[0]: dán nguyên context làm answer sẽ thổi phồng
        # faithfulness (answer trùng context) và bóp answer_relevancy — che mất lỗi thật.
        print(f"  ⚠️  LLM generation failed sau 3 lần thử: {e}", flush=True)
        return "Không tìm thấy thông tin."


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline: hybrid search → rerank → parent expand → LLM."""
    t0 = time.time()
    results = search.search(query)
    t_retrieval = (time.time() - t0) * 1000

    t0 = time.time()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t_rerank = (time.time() - t0) * 1000

    contexts = _expand_to_parents(reranked, results)

    t0 = time.time()
    if OPENAI_API_KEY and contexts:
        context_str = "\n\n---\n\n".join(contexts)
        answer = _generate_answer(query, context_str)
    else:
        answer = "Không tìm thấy thông tin."
    t_generation = (time.time() - t0) * 1000

    _QUERY_TIMINGS.append({
        "query": query,
        "retrieval_ms": round(t_retrieval, 1),
        "rerank_ms": round(t_rerank, 1),
        "generation_ms": round(t_generation, 1),
        "total_ms": round(t_retrieval + t_rerank + t_generation, 1),
        "n_contexts": len(contexts),
    })
    return answer, contexts


def print_latency_report(path: str = "reports/latency_report.json") -> dict:
    """Bảng latency breakdown từng bước (bonus: latency report)."""
    if not _QUERY_TIMINGS:
        return {}

    stages = ["retrieval_ms", "rerank_ms", "generation_ms", "total_ms"]
    n = len(_QUERY_TIMINGS)
    avg = {s: sum(q[s] for q in _QUERY_TIMINGS) / n for s in stages}
    p95 = {s: sorted(q[s] for q in _QUERY_TIMINGS)[min(int(n * 0.95), n - 1)] for s in stages}

    print("\n" + "=" * 60)
    print(f"LATENCY BREAKDOWN (n={n} queries)")
    print("=" * 60)
    print(f"{'Stage':<22} {'Avg (ms)':>10} {'P95 (ms)':>10} {'% total':>9}")
    print("-" * 60)
    for s in stages[:-1]:
        share = avg[s] / avg["total_ms"] * 100 if avg["total_ms"] else 0
        print(f"{s.replace('_ms',''):<22} {avg[s]:>10.1f} {p95[s]:>10.1f} {share:>8.1f}%")
    print("-" * 60)
    print(f"{'TOTAL / query':<22} {avg['total_ms']:>10.1f} {p95['total_ms']:>10.1f} {100.0:>8.1f}%")

    print("\nOne-time build cost:")
    for stage, secs in _BUILD_TIMINGS.items():
        print(f"  {stage:<20} {secs:>8.1f}s")

    report = {
        "per_query_avg_ms": {s: round(avg[s], 1) for s in stages},
        "per_query_p95_ms": {s: round(p95[s], 1) for s in stages},
        "build_seconds": {k: round(v, 1) for k, v in _BUILD_TIMINGS.items()},
        "queries": _QUERY_TIMINGS,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nLatency report saved to {path}")
    return report


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    print_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
