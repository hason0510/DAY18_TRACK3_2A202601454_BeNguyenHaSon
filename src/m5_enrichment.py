from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json as _json, threading
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICH_MODEL = "gpt-4o-mini"
MAX_WORKERS = 8          # enrichment là I/O bound → chạy song song cho nhanh


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── OpenAI client (lazy + reuse) ────────────────────────

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _get_client():
    """Tạo 1 client duy nhất (thread-safe) — mỗi OpenAI() mở 1 connection pool riêng."""
    global _CLIENT
    if _CLIENT is None and OPENAI_API_KEY:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                try:
                    from openai import OpenAI
                    _CLIENT = OpenAI()
                except Exception as e:
                    print(f"  ⚠️  OpenAI client init failed: {e}")
    return _CLIENT


def _chat(system: str, user: str, max_tokens: int = 200, json_mode: bool = False) -> str:
    """Gọi chat completion, trả "" nếu không có API key hoặc lỗi."""
    client = _get_client()
    if client is None:
        return ""
    try:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=ENRICH_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return ""


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text or not text.strip():
        return text

    summary = _chat(
        "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt. "
        "Summary PHẢI ngắn hơn đoạn gốc, giữ nguyên các con số và mốc thời gian.",
        text,
        max_tokens=150,
    )
    # Summary dài hơn bản gốc là vô nghĩa (LLM hay diễn giải thêm) → dùng extractive
    if summary and len(summary) <= len(text):
        return summary

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text or not text.strip():
        return []

    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Viết bằng tiếng Việt, mỗi câu hỏi trên 1 dòng, không đánh số.",
        text,
        max_tokens=200,
    )
    if raw:
        questions = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
        if questions:
            return questions[:n_questions]

    # Fallback extractive: biến câu khẳng định thành câu hỏi thô
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text or not text.strip():
        return text

    context = _chat(
        "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu "
        "và nói về chủ đề gì. Chỉ trả về đúng 1 câu, không giải thích thêm.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if context:
        return f"{context}\n\n{text}"

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    default = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if not text or not text.strip():
        return default

    raw = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text,
        max_tokens=150,
        json_mode=True,
    )
    if raw:
        try:
            data = _json.loads(raw)
            if isinstance(data, dict):
                return {**default, **data}
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  Metadata JSON parse failed: {e}")
    return default


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_SYSTEM = """Bạn phân tích một đoạn văn trong tài liệu nội bộ và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu bằng tiếng Việt",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Giữ nguyên mọi con số, mốc thời gian và số hiệu văn bản. Chỉ trả về JSON."""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if not text or not text.strip():
        return {}

    raw = _chat(_COMBINED_SYSTEM, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                max_tokens=400, json_mode=True)
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        return data if isinstance(data, dict) else {}
    except _json.JSONDecodeError as e:
        print(f"  ⚠️  Enrichment JSON parse failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def _enrich_one(chunk: dict, methods: list[str], use_combined: bool) -> EnrichedChunk:
    """Enrich 1 chunk (dùng lại cho cả chế độ tuần tự lẫn song song)."""
    text = chunk["text"]
    source = chunk.get("metadata", {}).get("source", "")

    if use_combined:
        result = _enrich_single_call(text, source)
        summary = result.get("summary", "")
        questions = result.get("questions", []) or []
        context_line = result.get("context", "")
        # Contextual prepend: chunk tự mang ngữ cảnh document khi được embed
        enriched_text = f"{context_line}\n\n{text}" if context_line else text
        auto_meta = result.get("metadata", {}) or {}
    else:
        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}

    if not isinstance(auto_meta, dict):
        auto_meta = {}

    return EnrichedChunk(
        original_text=text,
        enriched_text=enriched_text,
        summary=summary,
        hypothesis_questions=list(questions),
        auto_metadata={**chunk.get("metadata", {}), **auto_meta},
        method="+".join(methods),
    )


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if not chunks:
        return []

    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    total = len(chunks)
    enriched: list[EnrichedChunk | None] = [None] * total

    def _report(done: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  Enriched {done}/{total} chunks...", flush=True)

    # Enrichment là I/O bound (network) → ThreadPool rút thời gian từ N×latency
    # xuống ~N/MAX_WORKERS×latency. Không dùng khi không có API key (fallback thuần CPU).
    if _get_client() is not None and total > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as pool:
            futures = {pool.submit(_enrich_one, c, methods, use_combined): i
                       for i, c in enumerate(chunks)}
            for done, fut in enumerate(as_completed(futures), start=1):
                i = futures[fut]
                try:
                    enriched[i] = fut.result()
                except Exception as e:
                    print(f"  ⚠️  Enrich chunk {i} failed: {e}")
                    enriched[i] = EnrichedChunk(
                        original_text=chunks[i]["text"],
                        enriched_text=chunks[i]["text"],
                        summary="", hypothesis_questions=[],
                        auto_metadata=chunks[i].get("metadata", {}),
                        method="+".join(methods),
                    )
                _report(done)
    else:
        for i, chunk in enumerate(chunks):
            enriched[i] = _enrich_one(chunk, methods, use_combined)
            _report(i + 1)

    return [e for e in enriched if e is not None]


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

    combined = _enrich_single_call(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"\nCombined (1 call): {combined}")
