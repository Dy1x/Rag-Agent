from typing import Protocol

from langchain_core.documents import Document

from rag.hybrid_retriever import HybridRetrieverService
from rag.keyword_retriever import KeywordRetrieverService
from rag.vector_store import VectorStoreService
from utils.config_handler import retrieval_conf
from utils.logger_handler import logger


class RetrieverService(Protocol):
    def retrieve(self, query: str) -> list[Document]:
        ...


class VectorRetrieverService:
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.retriever = self.vector_store_service.get_retriever(
            k=retrieval_conf.get("vector_k")
        )

        logger.info(
            "[RetrieverFactory] 使用 vector 检索模式，"
            f"vector_k={retrieval_conf.get('vector_k')}"
        )

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)


class BM25OnlyRetrieverService:
    def __init__(self):
        self.keyword_retriever_service = KeywordRetrieverService()
        self.retriever = self.keyword_retriever_service.get_retriever()

        logger.info(
            "[RetrieverFactory] 使用 bm25 检索模式，"
            f"bm25_k={retrieval_conf.get('bm25_k')}"
        )

    def retrieve(self, query: str) -> list[Document]:
        if not self.retriever:
            logger.warning("[RetrieverFactory] BM25 retriever 不可用，返回空检索结果")
            return []

        return self.retriever.invoke(query)


class HybridRetrieverAdapter:
    def __init__(self):
        self.hybrid_retriever_service = HybridRetrieverService()

        logger.info(
            "[RetrieverFactory] 使用 hybrid 检索模式，"
            f"vector_k={retrieval_conf.get('vector_k')}, "
            f"bm25_k={retrieval_conf.get('bm25_k')}, "
            f"hybrid_k={retrieval_conf.get('hybrid_k')}, "
            f"vector_weight={retrieval_conf.get('vector_weight')}, "
            f"bm25_weight={retrieval_conf.get('bm25_weight')}"
        )

    def retrieve(self, query: str) -> list[Document]:
        return self.hybrid_retriever_service.retrieve(query)


class RetrieverFactory:
    @staticmethod
    def create() -> RetrieverService:
        mode = retrieval_conf.get("retrieval_mode", "vector")
        mode = str(mode).strip().lower()

        if mode == "vector":
            return VectorRetrieverService()

        if mode == "bm25":
            if not retrieval_conf.get("enable_bm25_retriever", True):
                logger.warning(
                    "[RetrieverFactory] retrieval_mode=bm25，"
                    "但 enable_bm25_retriever=false，降级为 vector"
                )
                return VectorRetrieverService()

            return BM25OnlyRetrieverService()

        if mode == "hybrid":
            if not retrieval_conf.get("enable_hybrid_retriever", True):
                logger.warning(
                    "[RetrieverFactory] retrieval_mode=hybrid，"
                    "但 enable_hybrid_retriever=false，降级为 vector"
                )
                return VectorRetrieverService()

            return HybridRetrieverAdapter()

        logger.warning(
            f"[RetrieverFactory] 未识别的 retrieval_mode={mode}，降级为 vector"
        )
        return VectorRetrieverService()