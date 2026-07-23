from rag.rerank.base import Reranker
from rag.rerank.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from rag.rerank.noop_reranker import NoopReranker
from utils.config_handler import retrieval_conf
from utils.logger_handler import logger


class RerankerFactory:
    @staticmethod
    def create() -> Reranker:
        enable_rerank = retrieval_conf.get(
            "enable_rerank",
            False,
        )

        if not enable_rerank:
            logger.info(
                "[RerankerFactory] Rerank未启用，"
                "使用NoopReranker"
            )
            return NoopReranker()

        rerank_type = str(
            retrieval_conf.get(
                "rerank_type",
                "cross_encoder",
            )
        ).strip().lower()

        if rerank_type == "noop":
            return NoopReranker()

        if rerank_type == "cross_encoder":
            logger.info(
                "[RerankerFactory] 使用本地CrossEncoder，"
                f"model={retrieval_conf.get('rerank_model_name')}"
            )

            return CrossEncoderReranker(
                model_name=retrieval_conf.get(
                    "rerank_model_name",
                    "",
                ),
                device=retrieval_conf.get(
                    "rerank_device",
                    "auto",
                ),
                batch_size=int(
                    retrieval_conf.get(
                        "rerank_batch_size",
                        8,
                    )
                ),
                max_length=int(
                    retrieval_conf.get(
                        "rerank_max_length",
                        512,
                    )
                ),
                score_threshold=retrieval_conf.get(
                    "rerank_score_threshold"
                ),
                fallback_enabled=bool(
                    retrieval_conf.get(
                        "rerank_fallback_enabled",
                        True,
                    )
                ),
            )

        raise ValueError(
            f"不支持的rerank_type：{rerank_type}"
        )