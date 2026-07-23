from langchain_core.documents import Document

from rag.rerank.base import RerankResult


class NoopReranker:
    """保持粗召回顺序，只截取前 top_k 条。"""

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not documents:
            return []

        return [
            RerankResult(
                document=document,
                score=0.0,
                original_rank=index,
                rerank_rank=index,
            )
            for index, document in enumerate(
                documents[:top_k],
                start=1,
            )
        ]