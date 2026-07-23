"""
总结服务类：用户提问，搜索参考资料，将提问和参开资料提交给模型，让模型总结回复
"""
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.cache_handler import RedisCache
from utils.logger_handler import logger
from utils.semantic_cache_handler import SemanticCache
from rag.retriever_factory import RetrieverFactory
from rag.rerank.factory import RerankerFactory
from utils.config_handler import retrieval_conf
from rag.semantic_cache.decision_service import (
    SemanticCacheDecisionService,
)


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummerizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = RetrieverFactory.create()
        self.reranker = RerankerFactory.create()

        self.rerank_top_k = int(
            retrieval_conf.get("rerank_top_k", 5)
        )

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(
            self.prompt_text
        )
        self.model = chat_model
        self.chain = self._init_chain()
        self.cache = RedisCache()
        self.semantic_cache = SemanticCache()
        self.semantic_cache_decision = (
            SemanticCacheDecisionService(
                cache=self.semantic_cache,
            )
        )

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retrieve_candidates(
            self,
            query: str,
    ) -> list[Document]:
        """返回第一阶段粗召回结果。"""
        documents = self.retriever.retrieve(query)

        logger.info(
            "[RAG检索] 粗召回完成，"
            f"candidate_count={len(documents)}"
        )

        return documents

    def rerank_documents(
            self,
            query: str,
            documents: list[Document],
    ) -> list[Document]:
        """对粗召回结果执行重排序。"""
        results = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.rerank_top_k,
        )

        reranked_documents = []

        for result in results:
            document = result.document

            document.metadata = {
                **document.metadata,
                "rerank_score": result.score,
                "original_rank": result.original_rank,
                "rerank_rank": result.rerank_rank,
            }

            reranked_documents.append(document)

        return reranked_documents

    def retriever_docs(
            self,
            query: str,
    ) -> list[Document]:
        candidate_documents = self.retrieve_candidates(query)

        return self.rerank_documents(
            query=query,
            documents=candidate_documents,
        )

    def rag_summerize_without_cache(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】:参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

    # def rag_summerize(self, query: str) -> str:
    #     if not self.cache.enable:
    #         logger.info("[RAG缓存]缓存开关已关闭，跳过缓存读写")
    #         return self.rag_summerize_without_cache(query)
    #
    #     cached_answer = self.cache.get(query)
    #     if cached_answer:
    #         logger.info(f"[RAG缓存]命中缓存，query={query}")
    #         return cached_answer
    #
    #     logger.info(f"[RAG缓存]未命中缓存，执行RAG，query={query}")
    #     answer = self.rag_summerize_without_cache(query)
    #     if answer:
    #         saved = self.cache.set(query, answer)
    #         if saved:
    #             logger.info(f"[RAG缓存]写入缓存，query={query}")
    #
    #     return answer

    def _generate_and_cache(
            self,
            query: str,
    ) -> str:
        answer = self.rag_summerize_without_cache(
            query
        )

        if not answer:
            return answer

        redis_saved = self.cache.set(
            query,
            answer,
        )

        if redis_saved:
            logger.info(
                "[RAG缓存]写入Redis精确缓存，"
                f"query={query}"
            )

        semantic_saved = (
            self.semantic_cache.add_cache(
                query=query,
                answer=answer,
            )
        )

        if semantic_saved:
            logger.info(
                "[语义缓存]写入成功，"
                f"query={query}"
            )

        return answer

    def rag_summerize(
            self,
            query: str,
    ) -> str:
        if not query or not query.strip():
            raise ValueError("query不能为空")

        # 第一层：Redis精确缓存
        if self.cache.enable:
            cached_answer = self.cache.get(query)

            if cached_answer:
                logger.info(
                    "[RAG缓存]Redis精确命中，"
                    f"query={query}"
                )
                return cached_answer

        # 第二层：语义缓存组合决策
        semantic_decision = (
            self.semantic_cache_decision.lookup(
                query
            )
        )

        if semantic_decision:
            logger.info(
                "[RAG缓存]语义缓存命中，"
                f"query={query}, "
                f"cached_query="
                f"{semantic_decision.cached_query}, "
                f"embedding_score="
                f"{semantic_decision.embedding_score:.4f}, "
                f"verification_score="
                f"{semantic_decision.verification_score:.4f}"
            )

            return semantic_decision.answer

        # Redis关闭或锁关闭时直接执行RAG。
        # 语义缓存和Redis开关保持独立。
        if (
                not self.cache.enable
                or not self.cache.lock_enable
        ):
            logger.info(
                "[RAG缓存]未使用Redis重建锁，"
                f"执行完整RAG，query={query}"
            )

            return self._generate_and_cache(query)

        lock_token = str(uuid.uuid4())

        locked = self.cache.acquire_lock(
            query,
            lock_token,
        )

        if locked:
            logger.info(
                "[RAG缓存]获取重建锁成功，"
                f"query={query}"
            )

            try:
                # Double Check：
                # 等待锁期间可能已有请求写入缓存。
                cached_answer = self.cache.get(query)

                if cached_answer:
                    logger.info(
                        "[RAG缓存]加锁后二次命中Redis，"
                        f"query={query}"
                    )
                    return cached_answer

                semantic_decision = (
                    self.semantic_cache_decision.lookup(
                        query
                    )
                )

                if semantic_decision:
                    logger.info(
                        "[RAG缓存]加锁后二次命中语义缓存，"
                        f"query={query}"
                    )
                    return semantic_decision.answer

                return self._generate_and_cache(query)

            finally:
                self.cache.release_lock(
                    query,
                    lock_token,
                )

                logger.info(
                    "[RAG缓存]释放重建锁，"
                    f"query={query}"
                )

        logger.info(
            "[RAG缓存]未获取重建锁，等待缓存生成，"
            f"query={query}"
        )

        cached_answer = self.cache.wait_for_cache(
            query
        )

        if cached_answer:
            logger.info(
                "[RAG缓存]等待后命中Redis缓存，"
                f"query={query}"
            )
            return cached_answer

        # 其他请求可能使用了不同表达，因此再查一次语义缓存。
        semantic_decision = (
            self.semantic_cache_decision.lookup(
                query
            )
        )

        if semantic_decision:
            logger.info(
                "[RAG缓存]等待后命中语义缓存，"
                f"query={query}"
            )
            return semantic_decision.answer

        logger.warning(
            "[RAG缓存]等待超时，降级执行完整RAG，"
            f"query={query}"
        )

        return self.rag_summerize_without_cache(
            query
        )


if __name__ == '__main__':
    rag = RagSummerizeService()
    print(rag.rag_summerize("小户型适合哪些扫地机器人"))
