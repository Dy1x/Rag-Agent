import time
from threading import Lock

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.rerank.base import RerankResult
from rag.rerank.noop_reranker import NoopReranker
from utils.logger_handler import logger


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        score_threshold: float | None = None,
        fallback_enabled: bool = True,
    ):
        if not model_name:
            raise ValueError("rerank_model_name 不能为空")

        self.model_name = model_name
        self.device = None if device in (None, "", "auto") else device
        self.batch_size = batch_size
        self.max_length = max_length
        self.score_threshold = score_threshold
        self.fallback_enabled = fallback_enabled

        self._model: CrossEncoder | None = None
        self._model_lock = Lock()
        self._fallback = NoopReranker()

    def _get_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                logger.info(
                    "[CrossEncoderReranker] 开始加载本地模型，"
                    f"model={self.model_name}, device={self.device or 'auto'}"
                )

                started_at = time.perf_counter()

                self._model = CrossEncoder(
                    model_name_or_path=self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                )

                elapsed_ms = (
                    time.perf_counter() - started_at
                ) * 1000

                logger.info(
                    "[CrossEncoderReranker] 模型加载完成，"
                    f"elapsed_ms={elapsed_ms:.1f}"
                )

        return self._model

    @staticmethod
    def _prepare_documents(
        documents: list[Document],
    ) -> list[tuple[int, Document]]:
        valid_documents = []

        for original_rank, document in enumerate(
            documents,
            start=1,
        ):
            if not document:
                continue

            content = document.page_content

            if not content or not content.strip():
                continue

            valid_documents.append(
                (original_rank, document)
            )

        return valid_documents

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query or not query.strip():
            raise ValueError("query 不能为空")

        if not documents:
            return []

        valid_documents = self._prepare_documents(documents)

        if not valid_documents:
            return []

        try:
            model = self._get_model()

            sentence_pairs = [
                (
                    query,
                    document.page_content,
                )
                for _, document in valid_documents
            ]

            started_at = time.perf_counter()

            raw_scores = model.predict(
                sentence_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            elapsed_ms = (
                time.perf_counter() - started_at
            ) * 1000

            scores = np.asarray(
                raw_scores,
                dtype=float,
            ).reshape(-1)

            if len(scores) != len(valid_documents):
                raise ValueError(
                    "Rerank模型返回分数数量与候选文档数量不一致："
                    f"scores={len(scores)}, "
                    f"documents={len(valid_documents)}"
                )

            scored_documents = []

            for (
                original_rank,
                document,
            ), score in zip(valid_documents, scores):
                score_value = float(score)

                if (
                    self.score_threshold is not None
                    and score_value < self.score_threshold
                ):
                    continue

                scored_documents.append(
                    {
                        "document": document,
                        "score": score_value,
                        "original_rank": original_rank,
                    }
                )

            scored_documents.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            selected_documents = scored_documents[:top_k]

            results = [
                RerankResult(
                    document=item["document"],
                    score=item["score"],
                    original_rank=item["original_rank"],
                    rerank_rank=rerank_rank,
                )
                for rerank_rank, item in enumerate(
                    selected_documents,
                    start=1,
                )
            ]

            logger.info(
                "[CrossEncoderReranker] 重排序完成，"
                f"model={self.model_name}, "
                f"input_count={len(valid_documents)}, "
                f"output_count={len(results)}, "
                f"elapsed_ms={elapsed_ms:.1f}"
            )

            for result in results:
                logger.debug(
                    "[CrossEncoderReranker] 排名结果，"
                    f"original_rank={result.original_rank}, "
                    f"rerank_rank={result.rerank_rank}, "
                    f"score={result.score:.6f}, "
                    f"source={result.document.metadata.get('source')}"
                )

            return results

        except Exception as exc:
            logger.error(
                "[CrossEncoderReranker] 重排序失败，"
                f"error={exc}",
                exc_info=True,
            )

            if not self.fallback_enabled:
                raise

            logger.warning(
                "[CrossEncoderReranker] 降级使用原始检索顺序"
            )

            return self._fallback.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )