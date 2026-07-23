from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


@dataclass(frozen=True)
class RerankResult:
    document: Document
    score: float
    original_rank: int
    rerank_rank: int


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[RerankResult]:
        ...