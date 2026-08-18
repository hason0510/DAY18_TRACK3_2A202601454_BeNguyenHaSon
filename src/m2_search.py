from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" ("nghỉ_phép"). BM25 tokenize bằng split()
    → "nghỉ_phép" là 1 token, còn query "nghỉ phép" là 2 token → không khớp.
    Vì vậy phải replace("_", " ") sau khi segment: giá trị của bước này là
    chuẩn hoá ranh giới từ (bỏ dấu câu, tách dính), không phải giữ từ ghép.
    """
    if not text or not text.strip():
        return text
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as e:                      # underthesea lỗi → fallback thô
        print(f"  ⚠️  Vietnamese segmentation failed ({e}), fallback to raw text")
        return text


def _tokenize(text: str) -> list[str]:
    """Chuẩn hoá tokens cho BM25 (dùng chung cho corpus và query)."""
    return segment_vietnamese(text).lower().split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks or []
        self.corpus_tokens = [_tokenize(c.get("text", "")) for c in self.documents]
        # BM25Okapi crash nếu corpus rỗng → giữ bm25 = None, search() trả []
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:                  # 0 = không share token nào với query
                continue
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            print("  ⚠️  No chunks to index")
            return

        # recreate_collection() đã deprecated ở qdrant-client >= 1.12
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c.get("text", "") for c in chunks]
        vectors = self._get_encoder().encode(texts, batch_size=16, show_progress_bar=True)

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": texts[i]},
            )
            for i in range(len(chunks))
        ]

        for start in range(0, len(points), 128):        # upsert theo batch cho nhẹ request
            self.client.upsert(collection_name=collection, points=points[start:start + 128])

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            # qdrant-client >= 1.10 dùng query_points(), search() đã deprecated
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"  ⚠️  Dense search failed: {e}")
            return []

        return [
            SearchResult(
                text=(pt.payload or {}).get("text", ""),
                score=float(pt.score),
                metadata=pt.payload or {},
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    RRF chỉ dùng THỨ HẠNG, không dùng score gốc — nên không cần normalize
    giữa BM25 (0..∞) và cosine (0..1). Doc xuất hiện ở cả 2 list được cộng dồn
    → đó là lý do hybrid thắng từng method riêng lẻ.
    """
    rrf_scores: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)

    return [
        SearchResult(
            text=e["result"].text,
            score=float(e["score"]),
            metadata=e["result"].metadata,
            method="hybrid",
        )
        for e in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
