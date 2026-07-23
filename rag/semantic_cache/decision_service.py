import time
from dataclasses import dataclass
from typing import Any

from rag.semantic_cache.entity_guard import (
    SemanticCacheEntityGuard,
)
from rag.semantic_cache.verifier import (
    SemanticCacheVerifier,
)
from utils.config_handler import semantic_cache_conf
from utils.logger_handler import logger
from utils.semantic_cache_handler import SemanticCache


@dataclass(frozen=True)
class SemanticCacheDecision:
    answer: str
    cached_query: str
    embedding_score: float
    verification_score: float
    candidate_count: int
    entity_blocked_count: int
    latency_ms: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "cached_query": self.cached_query,
            "embedding_score": self.embedding_score,
            "verification_score": self.verification_score,
            "candidate_count": self.candidate_count,
            "entity_blocked_count": self.entity_blocked_count,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


class SemanticCacheDecisionService:
    """
    语义缓存最终决策链：

    Embedding Top3
    → 实体冲突过滤
    → Cross-Encoder验证
    → 返回缓存答案
    """

    def __init__(
        self,
        cache: SemanticCache | None = None,
        entity_guard: SemanticCacheEntityGuard | None = None,
        verifier: SemanticCacheVerifier | None = None,
    ):
        self.cache = cache or SemanticCache()
        self.entity_guard = (
            entity_guard
            or SemanticCacheEntityGuard()
        )
        self.verifier = (
            verifier
            or SemanticCacheVerifier()
        )

        self.threshold = float(
            semantic_cache_conf.get(
                "verifier_threshold",
                0.50,
            )
        )

    def lookup(
        self,
        query: str,
    ) -> SemanticCacheDecision | None:
        if not self.cache.enable:
            return None

        if not query or not query.strip():
            return None

        if not self.cache.is_semantic_cache_safe_query(
            query
        ):
            logger.info(
                "[语义缓存决策]动态或个性化问题绕过，"
                f"query={query}"
            )
            return None

        started_at = time.perf_counter()

        candidates = self.cache.retrieve_candidates(
            query=query,
        )

        if not candidates:
            logger.info(
                "[语义缓存决策]Embedding无候选，"
                f"query={query}"
            )
            return None

        compatible_candidates = []
        blocked_count = 0

        for candidate in candidates:
            guard_result = self.entity_guard.check(
                query=query,
                cached_query=candidate.query,
            )

            if not guard_result.compatible:
                blocked_count += 1

                logger.info(
                    "[语义缓存决策]候选被实体校验拒绝，"
                    f"query={query}, "
                    f"cached_query={candidate.query}, "
                    f"dimensions="
                    f"{[item.dimension for item in guard_result.conflicts]}"
                )
                continue

            compatible_candidates.append(candidate)

        if not compatible_candidates:
            logger.info(
                "[语义缓存决策]所有候选均被实体校验拒绝，"
                f"query={query}"
            )
            return None

        selected = self.verifier.select_best(
            query=query,
            candidates=compatible_candidates,
            threshold=self.threshold,
        )

        if selected is None:
            logger.info(
                "[语义缓存决策]Cross-Encoder未通过，"
                f"query={query}, "
                f"threshold={self.threshold:.4f}"
            )
            return None

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        candidate = selected.candidate

        decision = SemanticCacheDecision(
            answer=candidate.answer,
            cached_query=candidate.query,
            embedding_score=candidate.score,
            verification_score=selected.score,
            candidate_count=len(candidates),
            entity_blocked_count=blocked_count,
            latency_ms=elapsed_ms,
            metadata=dict(candidate.metadata),
        )

        logger.info(
            "[语义缓存决策]最终命中，"
            f"query={query}, "
            f"cached_query={candidate.query}, "
            f"embedding_score={candidate.score:.4f}, "
            f"verification_score={selected.score:.4f}, "
            f"latency_ms={elapsed_ms:.1f}"
        )

        return decision