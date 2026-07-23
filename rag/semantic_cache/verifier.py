import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from utils.config_handler import semantic_cache_conf
from utils.logger_handler import logger
from utils.semantic_cache_handler import (
    SemanticCacheCandidate,
)


@dataclass(frozen=True)
class SemanticCacheVerification:
    """单条缓存候选的Cross-Encoder验证结果。"""

    candidate: SemanticCacheCandidate
    score: float
    accepted: bool
    original_rank: int
    verification_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cached_query": self.candidate.query,
            "answer": self.candidate.answer,
            "embedding_score": self.candidate.score,
            "cross_encoder_score": self.score,
            "accepted": self.accepted,
            "original_rank": self.original_rank,
            "verification_rank": self.verification_rank,
            "created_at": self.candidate.created_at,
            "metadata": dict(self.candidate.metadata),
        }


class SemanticCacheVerifier:
    """
    使用Cross-Encoder判断当前问题和缓存问题能否共享答案。

    Embedding负责宽松召回，Cross-Encoder负责精确验证。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        threshold: float | None = None,
        enabled: bool | None = None,
        fail_closed: bool | None = None,
    ):
        self.enabled = bool(
            enabled
            if enabled is not None
            else semantic_cache_conf.get(
                "enable_candidate_verification",
                True,
            )
        )

        self.model_name = str(
            model_name
            or semantic_cache_conf.get(
                "verifier_model_name",
                "BAAI/bge-reranker-v2-m3",
            )
        )

        configured_device = (
            device
            if device is not None
            else semantic_cache_conf.get(
                "verifier_device",
                "auto",
            )
        )

        self.device = (
            None
            if configured_device in (None, "", "auto")
            else str(configured_device)
        )

        self.batch_size = int(
            batch_size
            if batch_size is not None
            else semantic_cache_conf.get(
                "verifier_batch_size",
                8,
            )
        )

        self.max_length = int(
            max_length
            if max_length is not None
            else semantic_cache_conf.get(
                "verifier_max_length",
                512,
            )
        )

        self.threshold = float(
            threshold
            if threshold is not None
            else semantic_cache_conf.get(
                "verifier_threshold",
                0.85,
            )
        )

        self.fail_closed = bool(
            fail_closed
            if fail_closed is not None
            else semantic_cache_conf.get(
                "verifier_fail_closed",
                True,
            )
        )

        if not self.model_name:
            raise ValueError(
                "verifier_model_name不能为空"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "verifier_batch_size必须大于0"
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "verifier_threshold必须位于0到1之间"
            )

        self._model: CrossEncoder | None = None
        self._model_lock = Lock()

    def _get_model(self) -> CrossEncoder:
        """延迟加载模型，第一次验证时才占用内存。"""

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            logger.info(
                "[语义缓存验证]开始加载Cross-Encoder，"
                f"model={self.model_name}, "
                f"device={self.device or 'auto'}"
            )

            started_at = time.perf_counter()

            self._model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
                max_length=self.max_length,
                activation_fn=torch.nn.Sigmoid(),
            )

            elapsed_ms = (
                time.perf_counter() - started_at
            ) * 1000

            logger.info(
                "[语义缓存验证]模型加载完成，"
                f"elapsed_ms={elapsed_ms:.1f}"
            )

        return self._model

    def score_candidates(
        self,
        query: str,
        candidates: list[SemanticCacheCandidate],
    ) -> list[SemanticCacheVerification]:
        """
        对全部候选进行评分。

        本方法返回所有候选，不根据阈值删除，
        方便离线评估多个阈值。
        """

        if not self.enabled:
            logger.info(
                "[语义缓存验证]验证功能已关闭"
            )
            return []

        if not query or not query.strip():
            return []

        if not candidates:
            return []

        try:
            model = self._get_model()

            sentence_pairs = [
                (
                    query,
                    candidate.query,
                )
                for candidate in candidates
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

            if len(scores) != len(candidates):
                raise ValueError(
                    "Cross-Encoder返回分数数量错误："
                    f"scores={len(scores)}, "
                    f"candidates={len(candidates)}"
                )

            scored_items = [
                {
                    "candidate": candidate,
                    "score": float(score),
                    "original_rank": rank,
                }
                for rank, (
                    candidate,
                    score,
                ) in enumerate(
                    zip(candidates, scores),
                    start=1,
                )
            ]

            scored_items.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            results = [
                SemanticCacheVerification(
                    candidate=item["candidate"],
                    score=item["score"],
                    accepted=(
                        item["score"] >= self.threshold
                    ),
                    original_rank=item["original_rank"],
                    verification_rank=rank,
                )
                for rank, item in enumerate(
                    scored_items,
                    start=1,
                )
            ]

            logger.info(
                "[语义缓存验证]候选验证完成，"
                f"query={query}, "
                f"candidate_count={len(candidates)}, "
                f"elapsed_ms={elapsed_ms:.1f}"
            )

            return results

        except Exception as exc:
            logger.error(
                "[语义缓存验证]Cross-Encoder验证失败，"
                f"query={query}, error={exc}",
                exc_info=True,
            )

            if self.fail_closed:
                return []

            raise

    def select_best(
        self,
        query: str,
        candidates: list[SemanticCacheCandidate],
        threshold: float | None = None,
    ) -> SemanticCacheVerification | None:
        """返回分数最高且达到阈值的候选。"""

        selected_threshold = (
            float(threshold)
            if threshold is not None
            else self.threshold
        )

        if not 0 <= selected_threshold <= 1:
            raise ValueError(
                "threshold必须位于0到1之间"
            )

        results = self.score_candidates(
            query=query,
            candidates=candidates,
        )

        for result in results:
            if result.score >= selected_threshold:
                return SemanticCacheVerification(
                    candidate=result.candidate,
                    score=result.score,
                    accepted=True,
                    original_rank=result.original_rank,
                    verification_rank=(
                        result.verification_rank
                    ),
                )

        return None