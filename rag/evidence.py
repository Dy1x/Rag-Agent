import re
from pathlib import PureWindowsPath
from threading import Lock
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator


RAG_CACHE_VERSION = "agentic_v1"
RAG_CACHE_VERSION_FIELD = "rag_cache_version"
RAG_CACHE_RESULT_FIELD = "rag_result"
RAG_RETRIEVAL_LIMIT = 2
_CITATION = re.compile(r"【(E[0-9]+)】")


class RetrievalBudget:
    """每个请求共享的预算。并行工具调用必须先领取一次检索名额。"""

    def __init__(self):
        self._remaining = RAG_RETRIEVAL_LIMIT
        self._lock = Lock()

    def claim(self) -> bool:
        with self._lock:
            if self._remaining == 0:
                return False
            self._remaining -= 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining


class EvidenceChunk(BaseModel):
    id: str = Field(pattern=r"^E[1-9][0-9]*$")
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @classmethod
    def from_document(cls, evidence_id: str, doc: Document) -> "EvidenceChunk":
        metadata = doc.metadata
        path = metadata.get("filename") or metadata.get("source")
        source = PureWindowsPath(str(path)).name if path else "知识库"

        page_label = metadata.get("page_label")
        if page_label:
            source += f"，第 {page_label} 页"
        elif isinstance(metadata.get("page"), int):
            source += f"，第 {metadata['page'] + 1} 页"

        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            source += f"，片段 {chunk_id}"
        elif metadata.get("chunk_index") is not None:
            source += f"，片段 {metadata['chunk_index']}"

        return cls(id=evidence_id, source=source, text=doc.page_content.strip())


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    next_step: Literal["answer", "retrieve", "clarify", "stop"]
    selected_ids: list[str]
    gaps: list[str]
    followup_query: str
    clarification_question: str


class RagResult(BaseModel):
    status: Literal["answered", "clarify", "insufficient"]
    text: str = Field(min_length=1)
    evidence: list[EvidenceChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citations(self) -> "RagResult":
        if self.status == "answered":
            cited = set(_CITATION.findall(self.text))
            evidence_ids = {chunk.id for chunk in self.evidence}
            if not cited or not cited <= evidence_ids or len(evidence_ids) != len(self.evidence):
                raise ValueError("回答必须引用已选中的证据片段")
        return self

    @property
    def sources(self) -> list[dict[str, str]]:
        cited = set(_CITATION.findall(self.text))
        return [
            {"id": chunk.id, "source": chunk.source}
            for chunk in self.evidence if chunk.id in cited
        ]

    def format(self) -> str:
        sources = self.sources
        if not sources:
            return self.text
        references = "\n".join(
            f"- 【{item['id']}】{item['source']}" for item in sources
        )
        return f"{self.text}\n\n**知识库出处**\n{references}"


class CachedRagAnswer(BaseModel):
    version: Literal["agentic_v1"]
    query: str = Field(min_length=1)
    result: RagResult

    @model_validator(mode="after")
    def validate_answer(self) -> "CachedRagAnswer":
        if self.result.status != "answered":
            raise ValueError("仅缓存有证据支持的回答")
        return self
