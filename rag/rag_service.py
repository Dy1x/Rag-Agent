"""在检索和重排结果上评估证据，生成带出处的知识库回答。"""

import json
import uuid
from collections.abc import Sequence
from functools import cached_property

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from model.factory import get_chat_model
from rag.evidence import (
    RAG_CACHE_RESULT_FIELD,
    RAG_CACHE_VERSION,
    RAG_CACHE_VERSION_FIELD,
    CachedRagAnswer,
    EvidenceAssessment,
    EvidenceChunk,
    RagResult,
    RetrievalBudget,
)
from rag.rerank.factory import RerankerFactory
from rag.retriever_factory import RetrieverFactory
from rag.semantic_cache.decision_service import SemanticCacheDecisionService
from utils.cache_handler import RedisCache
from utils.config_handler import retrieval_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_assessment_prompt, load_rag_prompts
from utils.semantic_cache_handler import SemanticCache


class RagSummerizeService:
    def __init__(self):
        self.rerank_top_k = int(retrieval_conf.get("rerank_top_k", 5))
        self.cache = RedisCache(namespace=RAG_CACHE_VERSION)

    @cached_property
    def retriever(self):
        return RetrieverFactory.create()

    @cached_property
    def reranker(self):
        return RerankerFactory.create()

    @cached_property
    def semantic_cache(self):
        return SemanticCache(metadata_filter={RAG_CACHE_VERSION_FIELD: RAG_CACHE_VERSION})

    @cached_property
    def semantic_cache_decision(self):
        return SemanticCacheDecisionService(cache=self.semantic_cache)

    @cached_property
    def assessment_prompt(self):
        return SystemMessage(content=load_rag_assessment_prompt())

    @cached_property
    def assessment_chain(self):
        return get_chat_model().bind(
            response_format={"type": "json_object"},
            max_tokens=800,
        ) | StrOutputParser()

    @cached_property
    def answer_chain(self):
        return (
            PromptTemplate.from_template(load_rag_prompts())
            | get_chat_model()
            | StrOutputParser()
        )

    def retrieve_candidates(self, query: str) -> list[Document]:
        """返回第一阶段粗召回结果。"""
        documents = self.retriever.retrieve(query)
        logger.info(f"[RAG检索] 粗召回完成，candidate_count={len(documents)}")
        return documents

    def rerank_documents(self, query: str, documents: list[Document]) -> list[Document]:
        """对粗召回结果执行重排序。"""
        results = self.reranker.rerank(
            query=query, documents=documents, top_k=self.rerank_top_k,
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

    def retriever_docs(self, query: str) -> list[Document]:
        return self.rerank_documents(query, self.retrieve_candidates(query))

    def _assess(
        self,
        question: str,
        searches: list[str],
        evidence: dict[str, EvidenceChunk],
        budget: RetrievalBudget,
    ) -> EvidenceAssessment | None:
        request = {
            "question": question,
            "search_queries": searches,
            "retrievals_left": budget.remaining,
            "evidence": [chunk.model_dump() for chunk in evidence.values()],
        }
        raw = self.assessment_chain.invoke([
            self.assessment_prompt,
            HumanMessage(content=json.dumps(request, ensure_ascii=False)),
        ])
        try:
            assessment = EvidenceAssessment.model_validate_json(raw)
            if set(assessment.selected_ids) - evidence.keys():
                raise ValueError("评估引用了不存在的片段")
            if assessment.next_step == "answer" and (
                not assessment.selected_ids or assessment.gaps
            ):
                raise ValueError("评估未提供足够的证据")
            if assessment.next_step == "retrieve" and (
                not assessment.followup_query or not assessment.gaps
            ):
                raise ValueError("评估未提供缺口或补查词")
            if assessment.next_step == "clarify" and not assessment.clarification_question:
                raise ValueError("评估未提供追问")
            logger.info(
                f"[Agentic RAG] 证据评估 next_step={assessment.next_step}, "
                f"selected_count={len(assessment.selected_ids)}, gap_count={len(assessment.gaps)}"
            )
            return assessment
        except ValueError:
            logger.warning("[Agentic RAG] 证据评估结果无效")
            return None

    @staticmethod
    def _insufficient(gaps: Sequence[str] = (), reason: str = "") -> RagResult:
        text = "现有知识库资料不足，无法确认这个问题。"
        missing = "、".join(gap for gap in gaps if gap)
        if missing:
            text += f"仍缺少以下证据：{missing}。"
        return RagResult(status="insufficient", text=text + reason)

    def _generate_answer(
        self,
        query: str,
        evidence: dict[str, EvidenceChunk],
        assessment: EvidenceAssessment,
    ) -> RagResult:
        selected = [evidence[chunk_id] for chunk_id in dict.fromkeys(assessment.selected_ids)]
        text = self.answer_chain.invoke({
            "input": query,
            "context": json.dumps([chunk.model_dump() for chunk in selected], ensure_ascii=False),
        }).strip()
        try:
            return RagResult(status="answered", text=text, evidence=selected)
        except ValidationError:
            logger.warning("[Agentic RAG] 生成的回答缺少有效引用")
            return RagResult(status="insufficient", text="本次未能生成带有效出处的回答，暂时无法确认。")

    def _retrieve_and_answer(self, query: str, budget: RetrievalBudget) -> RagResult:
        evidence: dict[str, EvidenceChunk] = {}
        seen_text: set[str] = set()
        searches: list[str] = []
        gaps: list[str] = []
        search_query = query

        while budget.claim():
            searches.append(search_query)
            logger.info(f"[Agentic RAG] 开始知识检索，剩余名额={budget.remaining}")
            for doc in self.retriever_docs(search_query):
                text_key = " ".join(doc.page_content.split())
                if text_key and text_key not in seen_text:
                    seen_text.add(text_key)
                    chunk_id = f"E{len(evidence) + 1}"
                    evidence[chunk_id] = EvidenceChunk.from_document(chunk_id, doc)

            assessment = self._assess(query, searches, evidence, budget)
            if assessment is None:
                return RagResult(status="insufficient", text="本次未能完成证据评估，暂时无法确认。")
            if assessment.next_step == "answer":
                return self._generate_answer(query, evidence, assessment)
            if assessment.next_step == "clarify":
                return RagResult(status="clarify", text=assessment.clarification_question)

            gaps = assessment.gaps
            if assessment.next_step == "stop":
                return self._insufficient(gaps)

            search_query = assessment.followup_query
            prior_searches = {" ".join(search.casefold().split()) for search in searches}
            if " ".join(search_query.casefold().split()) in prior_searches:
                return self._insufficient(gaps)

        return self._insufficient(gaps, reason="本轮知识库检索次数已用完。")

    @staticmethod
    def _read_cached(raw: str | None) -> CachedRagAnswer | None:
        if not raw:
            return None
        try:
            return CachedRagAnswer.model_validate_json(raw)
        except ValidationError:
            return None

    def _exact_hit(self, query: str, raw: str | None) -> RagResult | None:
        cached = self._read_cached(raw)
        if cached and cached.query == query:
            logger.info("[Agentic RAG] 精确缓存命中")
            return cached.result
        return None

    def _generate_and_cache(
        self, query: str, budget: RetrievalBudget, *, write_cache: bool = True,
    ) -> RagResult:
        decision = self.semantic_cache_decision.lookup(query)
        if decision:
            cached = self._read_cached(decision.metadata.get(RAG_CACHE_RESULT_FIELD))
            if cached and cached.query == decision.cached_query:
                evidence = {chunk.id: chunk for chunk in cached.result.evidence}
                assessment = self._assess(query, [], evidence, budget)
                if assessment and assessment.next_step == "answer":
                    result = self._generate_answer(query, evidence, assessment)
                    if result.status == "answered":
                        logger.info("[Agentic RAG] 缓存片段支持当前问题")
                        if write_cache:
                            self._save(query, result, semantic=False)
                        return result

        result = self._retrieve_and_answer(query, budget)
        if write_cache and result.status == "answered":
            self._save(query, result, semantic=True)
        return result

    def _save(self, query: str, result: RagResult, *, semantic: bool) -> None:
        raw = CachedRagAnswer(version=RAG_CACHE_VERSION, query=query, result=result).model_dump_json()
        self.cache.set(query, raw)
        if semantic:
            self.semantic_cache.add_cache(
                query=query,
                answer=result.format(),
                metadata={RAG_CACHE_VERSION_FIELD: RAG_CACHE_VERSION, RAG_CACHE_RESULT_FIELD: raw},
            )

    @staticmethod
    def _clean_query(query: str) -> str:
        if not query or not query.strip():
            raise ValueError("query不能为空")
        return query.strip()

    def answer_question(self, query: str, budget: RetrievalBudget | None = None) -> RagResult:
        query = self._clean_query(query)
        budget = budget if budget is not None else RetrievalBudget()

        hit = self._exact_hit(query, self.cache.get(query))
        if hit:
            return hit
        if not self.cache.enable or not self.cache.lock_enable:
            return self._generate_and_cache(query, budget)

        lock_token = str(uuid.uuid4())
        if self.cache.acquire_lock(query, lock_token):
            try:
                hit = self._exact_hit(query, self.cache.get(query))
                return hit or self._generate_and_cache(query, budget)
            finally:
                self.cache.release_lock(query, lock_token)

        hit = self._exact_hit(query, self.cache.wait_for_cache(query))
        return hit or self._generate_and_cache(query, budget, write_cache=False)

    def rag_summerize_without_cache(self, query: str) -> str:
        return self._retrieve_and_answer(self._clean_query(query), RetrievalBudget()).format()

    def rag_summerize(self, query: str, budget: RetrievalBudget | None = None) -> str:
        return self.answer_question(query, budget).format()


if __name__ == "__main__":
    rag = RagSummerizeService()
    print(rag.rag_summerize("第一次使用公共充电桩需要注意什么？"))
