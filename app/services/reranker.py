"""Cross-encoder reranker for retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import ScoredPoint
from sentence_transformers import CrossEncoder


@dataclass
class RankedResult:
    """Reranked vector point with a cross-encoder score."""

    point: ScoredPoint
    score: float


class Reranker:
    """Cross-encoder reranking service."""

    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredPoint], top_k: int) -> list[RankedResult]:
        """Rerank candidates and return top results."""

        if not candidates:
            return []
        pairs = [(query, str(candidate.payload.get("content", ""))) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [RankedResult(point=point, score=float(score)) for point, score in ranked[:top_k]]
