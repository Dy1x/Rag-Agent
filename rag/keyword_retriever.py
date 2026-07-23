import jieba

from langchain_community.retrievers import BM25Retriever

from rag.vector_store import VectorStoreService
from utils.config_handler import retrieval_conf
from utils.logger_handler import logger


def chinese_tokenize(text: str) -> list[str]:
    if not text:
        return []

    return [word.strip() for word in jieba.lcut(text) if word.strip()]


class KeywordRetrieverService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.documents = self.vector_store.get_all_chunks()
        self.documents = self._deduplicate_documents(self.documents)
        self.retriever = self._build_bm25_retriever()

    def _build_bm25_retriever(self):
        if not self.documents:
            logger.warning("[BM25检索]Chroma 中没有可用于构建 BM25 的 chunk")
            return None

        retriever = BM25Retriever.from_documents(
            self.documents,
            preprocess_func=chinese_tokenize,
        )
        retriever.k = retrieval_conf.get("bm25_k", 5)

        logger.info(f"[BM25检索]构建成功，chunk_count={len(self.documents)}")
        return retriever

    def get_retriever(self):
        return self.retriever

    def retrieve(self, query: str):
        if not self.retriever:
            return []

        return self.retriever.invoke(query)

    @staticmethod
    def _deduplicate_documents(documents):
        unique_docs = []
        seen_contents = set()

        for doc in documents:
            content_key = doc.page_content.strip()

            if not content_key:
                continue

            if content_key in seen_contents:
                continue

            seen_contents.add(content_key)
            unique_docs.append(doc)

        return unique_docs