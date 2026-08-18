from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _fallback_results(documents: list[dict], top_k: int) -> list[RerankResult]:
    """Nếu reranker không load được: giữ nguyên thứ tự retrieval (no-op rerank).

    Pipeline vẫn chạy end-to-end thay vì crash — chỉ mất phần precision gain.
    """
    ordered = sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)
    return [
        RerankResult(
            text=doc.get("text", ""),
            original_score=float(doc.get("score", 0.0)),
            rerank_score=float(doc.get("score", 0.0)),
            metadata=doc.get("metadata", {}),
            rank=i,
        )
        for i, doc in enumerate(ordered[:top_k])
    ]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"  ⚠️  CrossEncoder load failed: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        Bi-encoder (dense search) encode query và doc RIÊNG rồi so cosine.
        Cross-encoder đưa cả cặp (query, doc) qua cùng 1 forward pass → bắt được
        quan hệ token-level → chính xác hơn nhiều, nhưng O(n) forward pass nên
        chỉ dùng cho top-20 đã lọc, không dùng để quét cả corpus.
        """
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            return _fallback_results(documents, top_k)

        try:
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = model.predict(pairs)
        except Exception as e:
            print(f"  ⚠️  Rerank failed: {e}")
            return _fallback_results(documents, top_k)

        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)

        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from flashrank import Ranker
                self._model = Ranker()
            except Exception as e:
                print(f"  ⚠️  Flashrank load failed: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            return _fallback_results(documents, top_k)

        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": d.get("text", ""), "meta": d.get("metadata", {})}
                        for i, d in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank rerank failed: {e}")
            return _fallback_results(documents, top_k)

        out = []
        for i, r in enumerate(results[:top_k]):
            doc = documents[r.get("id", i)] if isinstance(r.get("id"), int) else {}
            out.append(RerankResult(
                text=r.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(r.get("score", 0.0)),
                metadata=r.get("meta", {}),
                rank=i,
            ))
        return out


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"Latency: avg={stats['avg_ms']:.1f}ms  min={stats['min_ms']:.1f}ms  max={stats['max_ms']:.1f}ms")
