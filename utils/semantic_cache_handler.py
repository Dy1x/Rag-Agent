import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from model.factory import embed_model
from utils.config_handler import semantic_cache_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


@dataclass(frozen=True)
class SemanticCacheCandidate:
    """Embedding召回得到的语义缓存候选。"""

    query: str
    answer: str
    score: float
    created_at: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "score": self.score,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class SemanticCache:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
        ttl_seconds: int | None = None,
        candidate_threshold: float | None = None,
        candidate_top_k: int | None = None,
        candidate_fetch_k: int | None = None,
    ):
        self.enable = bool(
            semantic_cache_conf.get(
                "enable_semantic_cache",
                True,
            )
        )

        # 兼容旧版search()直接命中的严格阈值。
        self.threshold = float(
            threshold
            if threshold is not None
            else semantic_cache_conf.get(
                "similarity_threshold",
                0.85,
            )
        )

        # 第一阶段Embedding宽松召回阈值。
        self.candidate_threshold = float(
            candidate_threshold
            if candidate_threshold is not None
            else semantic_cache_conf.get(
                "candidate_threshold",
                0.55,
            )
        )

        configured_top_k = semantic_cache_conf.get(
            "candidate_top_k",
            semantic_cache_conf.get("top_k", 3),
        )

        # top_k参数用于兼容原有测试脚本。
        self.candidate_top_k = int(
            candidate_top_k
            if candidate_top_k is not None
            else top_k
            if top_k is not None
            else configured_top_k
        )

        self.candidate_fetch_k = int(
            candidate_fetch_k
            if candidate_fetch_k is not None
            else semantic_cache_conf.get(
                "candidate_fetch_k",
                10,
            )
        )

        self.ttl_seconds = int(
            ttl_seconds
            if ttl_seconds is not None
            else semantic_cache_conf.get(
                "cache_ttl_seconds",
                86400,
            )
        )

        self.collection_name = (
            collection_name
            or semantic_cache_conf["collection_name"]
        )

        configured_directory = (
            persist_directory
            or semantic_cache_conf["persist_directory"]
        )

        directory_path = Path(configured_directory)

        if not directory_path.is_absolute():
            configured_directory = get_abs_path(
                configured_directory
            )

        if self.candidate_top_k <= 0:
            raise ValueError(
                "candidate_top_k必须大于0"
            )

        if self.candidate_fetch_k < self.candidate_top_k:
            self.candidate_fetch_k = self.candidate_top_k

        if not 0 <= self.candidate_threshold <= 1:
            raise ValueError(
                "candidate_threshold必须位于0到1之间"
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "similarity_threshold必须位于0到1之间"
            )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embed_model,
            persist_directory=str(configured_directory),
        )

    @staticmethod
    def is_semantic_cache_safe_query(
        query: str,
    ) -> bool:
        """判断请求是否适合使用共享语义缓存。"""

        unsafe_keywords = [
            "我所在城市",
            "我所在地区",
            "所在城市",
            "所在地区",
            "城市",
            "天气",
            "晴天",
            "雨天",
            "高温",
            "干燥",
            "潮湿",
            "高湿度",
            "湿度",
            "回南天",
            "南方",
            "北方",
            "当地",
            "地区",
            "环境",
            "用户ID",
            "用户编号",
            "用户100",
            "清扫记录",
            "使用报告",
            "月度报告",
            "本月报告",
            "上月报告",
            "设备序列号",
            "设备固件",
            "固件版本",
            "我的设备",
            "我的订单",
            "我的账号",
            "订单状态",
        ]

        return not any(
            keyword in query
            for keyword in unsafe_keywords
        )

    def _is_expired(
        self,
        created_at: int | None,
    ) -> bool:
        """判断缓存是否过期。"""

        if not self.ttl_seconds:
            return False

        if not created_at:
            return False

        cache_age = int(time.time()) - int(created_at)

        return cache_age > self.ttl_seconds

    def add_cache(
        self,
        query: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """将问题和最终答案写入语义缓存。"""

        if not self.enable:
            return False

        if not query or not query.strip():
            return False

        if not answer or not answer.strip():
            return False

        if not self.is_semantic_cache_safe_query(query):
            logger.info(
                "[语义缓存]跳过写入非安全query，"
                f"query={query}"
            )
            return False

        try:
            cache_metadata = {
                "query": query,
                "answer": answer,
                "created_at": int(time.time()),
            }

            if metadata:
                cache_metadata.update(metadata)

            document = Document(
                page_content=query,
                metadata=cache_metadata,
            )

            document_id = str(uuid.uuid4())

            self.vector_store.add_documents(
                [document],
                ids=[document_id],
            )

            logger.info(
                "[语义缓存]写入成功，"
                f"query={query}, id={document_id}"
            )

            return True

        except Exception as exc:
            logger.warning(
                "[语义缓存]写入失败，"
                f"query={query}, error={exc}",
                exc_info=True,
            )
            return False

    def retrieve_candidates(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SemanticCacheCandidate]:
        """
        使用Embedding宽松召回语义缓存候选。

        注意：
        返回候选不代表可以直接返回答案。
        后续必须交给实体校验或Cross-Encoder验证。
        """

        if not self.enable:
            return []

        if not query or not query.strip():
            return []

        if not self.is_semantic_cache_safe_query(query):
            logger.info(
                "[语义缓存]跳过非安全query候选召回，"
                f"query={query}"
            )
            return []

        selected_top_k = (
            int(top_k)
            if top_k is not None
            else self.candidate_top_k
        )

        selected_min_score = (
            float(min_score)
            if min_score is not None
            else self.candidate_threshold
        )

        if selected_top_k <= 0:
            raise ValueError("top_k必须大于0")

        if not 0 <= selected_min_score <= 1:
            raise ValueError(
                "min_score必须位于0到1之间"
            )

        fetch_k = max(
            self.candidate_fetch_k,
            selected_top_k,
        )

        try:
            results = (
                self.vector_store
                .similarity_search_with_relevance_scores(
                    query,
                    k=fetch_k,
                )
            )

            candidates: list[
                SemanticCacheCandidate
            ] = []

            for document, raw_score in results:
                score = float(raw_score)
                metadata = dict(document.metadata)

                cached_query = str(
                    metadata.get(
                        "query",
                        document.page_content,
                    )
                )

                cached_answer = metadata.get("answer")
                created_at = metadata.get("created_at")

                if not cached_answer:
                    logger.warning(
                        "[语义缓存]候选缺少answer，跳过，"
                        f"cached_query={cached_query}"
                    )
                    continue

                if self._is_expired(created_at):
                    logger.info(
                        "[语义缓存]候选已过期，跳过，"
                        f"cached_query={cached_query}"
                    )
                    continue

                if score < selected_min_score:
                    continue

                candidates.append(
                    SemanticCacheCandidate(
                        query=cached_query,
                        answer=str(cached_answer),
                        score=score,
                        created_at=(
                            int(created_at)
                            if created_at
                            else None
                        ),
                        metadata=metadata,
                    )
                )

                if len(candidates) >= selected_top_k:
                    break

            logger.info(
                "[语义缓存]Embedding候选召回完成，"
                f"query={query}, "
                f"candidate_count={len(candidates)}, "
                f"top_k={selected_top_k}, "
                f"min_score={selected_min_score:.4f}"
            )

            for index, candidate in enumerate(
                candidates,
                start=1,
            ):
                logger.debug(
                    "[语义缓存]候选详情，"
                    f"rank={index}, "
                    f"score={candidate.score:.4f}, "
                    f"cached_query={candidate.query}"
                )

            return candidates

        except Exception as exc:
            logger.warning(
                "[语义缓存]候选召回失败，"
                f"query={query}, error={exc}",
                exc_info=True,
            )
            return []

    def search(
        self,
        query: str,
    ) -> dict[str, Any] | None:
        """
        兼容旧版语义缓存直接命中逻辑。

        第一天暂时继续使用严格similarity_threshold，
        避免宽松Top3候选直接影响正式RAG返回结果。
        """

        candidates = self.retrieve_candidates(
            query=query,
            top_k=1,
            min_score=self.threshold,
        )

        if not candidates:
            logger.info(
                "[语义缓存]严格模式未命中，"
                f"query={query}, "
                f"threshold={self.threshold:.4f}"
            )
            return None

        candidate = candidates[0]

        logger.info(
            "[语义缓存]严格模式命中，"
            f"query={query}, "
            f"cached_query={candidate.query}, "
            f"score={candidate.score:.4f}"
        )

        return candidate.to_dict()

    def safe_search(
        self,
        query: str,
    ) -> dict[str, Any] | None:
        """兼容原有RAG调用入口。"""

        if not self.is_semantic_cache_safe_query(query):
            logger.info(
                "[语义缓存]跳过非安全query，"
                f"query={query}"
            )
            return None

        return self.search(query)

    def delete_collection(self) -> None:
        """删除当前集合，主要用于隔离评估。"""

        try:
            self.vector_store.delete_collection()

        except Exception as exc:
            logger.warning(
                "[语义缓存]删除集合失败，"
                f"collection={self.collection_name}, "
                f"error={exc}"
            )