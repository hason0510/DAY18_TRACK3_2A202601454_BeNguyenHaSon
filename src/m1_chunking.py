from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

MIN_PDF_TEXT_CHARS = 200      # duoi nguong nay coi nhu PDF khong co text layer


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text).

    Thứ tự thử:
      1. PyMuPDF (fitz) — giữ layout tốt hơn, đọc được nhiều PDF mà pypdf bó tay
      2. pypdf — fallback khi chưa cài PyMuPDF
      3. OCR (tuỳ chọn, bật bằng env ENABLE_PDF_OCR=1) — cho PDF scan ảnh thuần.
         Cần thêm: pip install pytesseract + cài Tesseract-OCR kèm gói ngôn ngữ 'vie'.
    """
    text = ""

    try:                                        # 1. PyMuPDF
        try:
            import pymupdf                      # tên module mới; `import fitz` đã deprecated
        except ImportError:
            import fitz as pymupdf
        with pymupdf.open(path) as doc:
            text = "\n\n".join(page.get_text() or "" for page in doc).strip()
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️  PyMuPDF failed on {os.path.basename(path)}: {e}")

    if not text:                                # 2. pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            print(f"  ⚠️  pypdf failed on {os.path.basename(path)}: {e}")

    if not text and os.getenv("ENABLE_PDF_OCR") == "1":   # 3. OCR (opt-in)
        text = _ocr_pdf(path)

    return text


def _ocr_pdf(path: str, lang: str = "vie", dpi: int = 200) -> str:
    """OCR PDF scan ảnh: render từng trang bằng PyMuPDF rồi đẩy qua Tesseract.

    Tắt mặc định vì (a) chậm, (b) cần binary Tesseract ngoài pip,
    (c) với corpus lab này 2 PDF scan không liên quan tới câu hỏi nào trong
    test_set — OCR chúng chỉ thêm nhiễu và kéo context_precision xuống.
    """
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        import pytesseract
        from PIL import Image
    except ImportError as e:
        print(f"  ⚠️  OCR bật nhưng thiếu dependency ({e}). "
              f"Cần: pip install pymupdf pytesseract pillow + cài Tesseract-OCR (gói 'vie').")
        return ""

    pages = []
    try:
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(pytesseract.image_to_string(img, lang=lang))
                print(f"    OCR {os.path.basename(path)} trang {i + 1}/{doc.page_count}", flush=True)
    except Exception as e:
        print(f"  ⚠️  OCR failed on {os.path.basename(path)}: {e}")
        return ""

    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        # Ngưỡng độ dài, không chỉ kiểm tra rỗng: PDF scan vẫn có thể trả về vài chục ký tự
        # từ lớp metadata (VD Nghị định 13 ra đúng 38 ký tự "Cơ quan phát hành: ...").
        # Một document 38 ký tự chỉ tạo thêm 1 parent rác cho retrieval.
        if len(text) >= MIN_PDF_TEXT_CHARS:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: chỉ trích được {len(text)} ký tự "
                  f"(< {MIN_PDF_TEXT_CHARS}) → PDF scan ảnh, cần OCR.")

    return docs


# ─── Helpers dùng chung cho 3 strategies ────────────────

_SEMANTIC_MODEL = None
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n\n')
_HEADER_RE = re.compile(r'^#{1,3}\s+.+$', flags=re.MULTILINE)


def _get_semantic_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (và cache) embedding model dùng cho semantic chunking.

    Cache ở module level: compare_strategies() gọi chunk_semantic() nhiều lần,
    load model mỗi lần sẽ mất ~10s/lần.
    """
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer(model_name)
    return _SEMANTIC_MODEL


def _split_sentences(text: str) -> list[str]:
    """Tách text thành câu: theo dấu câu kết thúc hoặc blank line."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Cắt cứng theo word boundary khi 1 unit (câu/đoạn) dài hơn size."""
    words = text.split()
    blocks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            blocks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        blocks.append(cur)
    return blocks or [text[:size]]


def _pack(units: list[str], size: int, sep: str) -> list[str]:
    """Gộp tuần tự các unit thành block ≤ size (không cắt giữa unit nếu tránh được)."""
    blocks, cur = [], ""
    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(u) > size:                      # unit dài hơn cả block → buộc phải cắt
            if cur:
                blocks.append(cur)
                cur = ""
            blocks.extend(_hard_split(u, size))
            continue
        if cur and len(cur) + len(sep) + len(u) > size:
            blocks.append(cur)
            cur = u
        else:
            cur = f"{cur}{sep}{u}" if cur else u
    if cur:
        blocks.append(cur)
    return blocks


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def _merge_small_groups(groups: list[list[str]], min_chars: int, max_chars: int) -> list[list[str]]:
    """Gộp các group quá ngắn vào group liền trước.

    Semantic split thuần tuý hay tạo ra mảnh vụn (header, câu 1 dòng) vì
    cosine similarity giữa header và body luôn thấp. Mảnh vụn làm embedding
    mất ngữ cảnh → merge lại cho tới khi đạt min_chars (nhưng không vượt max_chars).
    """
    merged: list[list[str]] = []
    for g in groups:
        if merged:
            prev_len = sum(len(s) + 1 for s in merged[-1])
            g_len = sum(len(s) + 1 for s in g)
            if (prev_len < min_chars or g_len < min_chars) and prev_len + g_len <= max_chars:
                merged[-1].extend(g)
                continue
        merged.append(list(g))
    return merged


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None,
                   min_chunk_chars: int = 100,
                   max_chunk_chars: int = 1000) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Thuật toán:
        1. Tách câu → embed từng câu (all-MiniLM-L6-v2)
        2. cosine(sent[i-1], sent[i]) < threshold  → biên giới chủ đề → chunk mới
        3. Merge các chunk vụn (< min_chunk_chars) để tránh phân mảnh
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        groups = [sentences]
    else:
        model = _get_semantic_model()
        embeddings = model.encode(sentences)

        def _cosine(a, b) -> float:
            return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

        groups = [[sentences[0]]]
        cur_len = len(sentences[0])
        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            too_long = cur_len + len(sentences[i]) > max_chunk_chars
            if sim < threshold or too_long:          # đổi chủ đề → cắt
                groups.append([sentences[i]])
                cur_len = len(sentences[i])
            else:                                     # cùng chủ đề → gộp
                groups[-1].append(sentences[i])
                cur_len += len(sentences[i]) + 1

    groups = _merge_small_groups(groups, min_chunk_chars, max_chunk_chars)

    return [
        Chunk(
            text=" ".join(g),
            metadata={**metadata, "strategy": "semantic",
                      "chunk_index": i, "n_sentences": len(g)},
        )
        for i, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_text in _pack(paragraphs, parent_size, "\n\n"):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "strategy": "hierarchical", "chunk_type": "parent",
                      "parent_id": pid, "chunk_index": len(parents)},
        ))

        # Mỗi parent → nhiều child nhỏ (đơn vị retrieve, ưu tiên precision)
        for j, child_text in enumerate(_pack(_split_sentences(parent_text), child_size, " ")):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "strategy": "hierarchical", "chunk_type": "child",
                          "parent_id": pid, "chunk_index": j},
                parent_id=pid,
            ))

    return (parents, children)


def get_parent_text(parents: list[Chunk], parent_id: str | None) -> str:
    """Child → parent lookup: retrieve bằng child, trả context bằng parent."""
    if not parent_id:
        return ""
    for p in parents:
        if p.metadata.get("parent_id") == parent_id:
            return p.text
    return ""


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None,
                          max_section_chars: int = 4000) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Mỗi chunk = header + toàn bộ nội dung của section đó. Header được giữ lại
    trong text để chunk tự mang ngữ cảnh ("## Nghỉ phép năm" + nội dung).
    """
    metadata = metadata or {}
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []

    def _emit(header: str, body: str) -> None:
        body = body.strip()
        header = header.strip()
        if not header and not body:
            return
        section = header.lstrip("#").strip()
        level = len(header) - len(header.lstrip("#")) if header else 0
        base_meta = {**metadata, "strategy": "structure",
                     "section": section, "header_level": level}

        full = f"{header}\n\n{body}".strip() if header else body
        if len(full) <= max_section_chars:
            pieces = [full]
        else:                                   # section quá dài → cắt theo paragraph
            paragraphs = [p for p in body.split("\n\n") if p.strip()]
            pieces = [f"{header}\n\n{p}".strip() if header else p
                      for p in _pack(paragraphs, max_section_chars, "\n\n")]

        for piece in pieces:
            if piece.strip():
                chunks.append(Chunk(text=piece,
                                    metadata={**base_meta, "chunk_index": len(chunks)}))

    cur_header, cur_body = "", []
    for part in parts:
        if not part:
            continue
        if re.match(r'^#{1,3}\s+.+$', part.strip()) and part.strip().startswith("#"):
            _emit(cur_header, "\n".join(cur_body))   # đóng section trước
            cur_header, cur_body = part, []
        else:
            cur_body.append(part)
    _emit(cur_header, "\n".join(cur_body))           # section cuối

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
