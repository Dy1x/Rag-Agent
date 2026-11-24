from mcp.server.fastmcp import FastMCP

from rag.rag_service import RagSummerizeService
from datetime import date, datetime
from decimal import Decimal

mcp = FastMCP("rag-query-server")
rag_service = RagSummerizeService()

def to_jsonable(data):
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]

    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}

    if isinstance(data, (datetime, date)):
        return data.isoformat()

    if isinstance(data, Decimal):
        return float(data)

    return data

def doc_to_dict(doc, rank: int) -> dict:
    return {
        "rank": rank,
        "content": doc.page_content,
        "metadata": doc.metadata,
    }

@mcp.tool()
def rag_summerize(query: str) -> dict:
    """基于知识库资料回答用户问题。适用于产品故障、维护保养、选购建议等知识库问答。"""
    if not query or not query.strip():
        return {
            "found": False,
            "message": "query不能为空",
            "data": None,
        }

    result = rag_service.answer_question(query)
    response = {
        "found": result.status == "answered",
        "status": result.status,
        "data": {
            "query": query,
            "answer": result.format(),
            "sources": result.sources,
        },
    }
    if result.status != "answered":
        response["message"] = result.text
    return response

@mcp.tool()
def retrieve_docs(query: str) -> dict:
    """检索知识库相关文档片段，用于查看 RAG 召回内容。"""
    if not query:
        return {
            "found": False,
            "message": "query不能为空",
            "data": [],
        }

    docs = rag_service.retriever_docs(query)

    if not docs:
        return {
            "found": False,
            "message": "未检索到相关文档",
            "data": [],
        }

    data = [
        doc_to_dict(doc, index)
        for index, doc in enumerate(docs, start=1)
    ]

    return {
        "found": True,
        "data": to_jsonable(data),
    }

if __name__ == "__main__":
    mcp.run()
