from langchain_classic.retrievers import EnsembleRetriever

from rag.keyword_retriever import KeywordRetrieverService
from rag.vector_store import VectorStoreService
from utils.config_handler import retrieval_conf
from utils.logger_handler import logger


class HybridRetrieverService:
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.keyword_retriever_service = KeywordRetrieverService()

        self.vector_retriever = self.vector_store_service.get_retriever(
            k=retrieval_conf.get("vector_k", 5)
        )
        self.bm25_retriever = self.keyword_retriever_service.get_retriever()

        self.retriever = self._build_hybrid_retriever()

    def _build_hybrid_retriever(self):
        if not self.bm25_retriever:
            logger.warning("[Hybrid检索] BM25 Retriever 不可用，降级为向量检索")
            return self.vector_retriever

        retriever = EnsembleRetriever(
            retrievers=[
                self.vector_retriever,
                self.bm25_retriever,
            ],
            weights=[
                retrieval_conf.get("vector_weight", 0.6),
                retrieval_conf.get("bm25_weight", 0.4),
            ],
        )

        logger.info(
            "[Hybrid检索] 构建成功，"
            f"vector_weight={retrieval_conf.get('vector_weight', 0.6)}, "
            f"bm25_weight={retrieval_conf.get('bm25_weight', 0.4)}"
        )

        return retriever

    def retrieve(self, query: str):
        docs = self.retriever.invoke(query)

        docs = self._deduplicate_docs(docs)

        hybrid_k = retrieval_conf.get("hybrid_k", 5)
        return docs[:hybrid_k]

    @staticmethod
    def _deduplicate_docs(docs):
        unique_docs = []
        seen_contents = set()

        for doc in docs:
            content_key = doc.page_content.strip()

            if not content_key or content_key in seen_contents:
                continue

            seen_contents.add(content_key)
            unique_docs.append(doc)

        return unique_docs