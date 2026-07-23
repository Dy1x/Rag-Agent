import json
import time
from pathlib import Path

from rag.rag_service import RagSummerizeService

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "rag_eval_dataset.json"
REPORT_DIR = BASE_DIR / "reports"
JSON_REPORT_PATH = REPORT_DIR / "rag_eval_report.json"
MD_REPORT_PATH = REPORT_DIR / "rag_eval_report.md"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def source_hit(docs, expected_sources: list[str]) -> bool:
    for doc in docs:
        source = doc.metadata.get("source", "")
        for expected in expected_sources:
            if expected in source:
                return True
    return False


def keyword_hit(docs, expected_keywords: list[str]) -> bool:
    content = "\n".join(doc.page_content for doc in docs)
    return any(keyword in content for keyword in expected_keywords)


def keyword_hits(answer: str, expected_keywords: list[str]):
    hit_keywords = []
    missed_keywords = []

    for keyword in expected_keywords:
        if keyword in answer:
            hit_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    recall = len(hit_keywords) / len(expected_keywords) if expected_keywords else 0.0
    return recall, hit_keywords, missed_keywords


def build_retrieved_docs(docs):
    retrieved_docs = []

    for index, doc in enumerate(docs, start=1):
        retrieved_docs.append({
            "rank": index,
            "source": doc.metadata.get("source", ""),
            "content_preview": doc.page_content[:200].replace("\n", " "),
        })

    return retrieved_docs


def write_json_report(summary: dict, results: list[dict]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": summary,
        "details": results,
    }

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_markdown_report(summary: dict, results: list[dict]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# RAG 评估报告")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 总问题数：{summary['total']}")
    lines.append(f"- Source Hit@K：{summary['source_hit_at_k']:.2f}")
    lines.append(f"- Retrieval Keyword Hit@K：{summary['retrieval_keyword_hit_at_k']:.2f}")
    lines.append(f"- 平均检索耗时：{summary['avg_retrieval_elapsed']:.3f}s")
    lines.append(f"- 平均答案关键词召回率：{summary['avg_answer_keyword_recall']:.2f}")
    lines.append(f"- 平均回答耗时：{summary['avg_answer_elapsed']:.3f}s")
    lines.append("")
    lines.append("## 明细")
    lines.append("")

    for result in results:
        lines.append(f"### {result['id']} {result['question']}")
        lines.append("")
        lines.append(f"- 来源命中：{result['source_hit']}")
        lines.append(f"- 检索关键词命中：{result['retrieval_keyword_hit']}")
        lines.append(f"- 检索耗时：{result['retrieval_elapsed']:.3f}s")
        lines.append(f"- 答案关键词召回率：{result['answer_keyword_recall']:.2f}")
        lines.append(f"- 命中答案关键词：{', '.join(result['hit_answer_keywords']) or '无'}")
        lines.append(f"- 未命中答案关键词：{', '.join(result['missed_answer_keywords']) or '无'}")
        lines.append(f"- 回答耗时：{result['answer_elapsed']:.3f}s")
        lines.append("")
        lines.append("检索结果：")
        lines.append("")

        for doc in result["retrieved_docs"]:
            lines.append(f"- Top{doc['rank']}：{doc['source']}")
            lines.append(f"  - {doc['content_preview']}")

        lines.append("")
        lines.append("答案：")
        lines.append("")
        lines.append(result["answer"])
        lines.append("")

    with open(MD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    dataset = load_dataset()
    rag = RagSummerizeService()

    results = []

    for item in dataset:
        question = item["question"]

        expected_sources = item.get("expected_sources", [])
        expected_keywords = item.get("expected_keywords", [])
        expected_answer_keywords = item.get("expected_answer_keywords", [])

        retrieval_start = time.time()
        docs = rag.retriever_docs(question)
        retrieval_elapsed = time.time() - retrieval_start

        is_source_hit = source_hit(docs, expected_sources)
        is_keyword_hit = keyword_hit(docs, expected_keywords)

        answer_start = time.time()
        answer = rag.rag_summerize(question)
        answer_elapsed = time.time() - answer_start

        answer_recall, hit_answer_keywords, missed_answer_keywords = keyword_hits(
            answer,
            expected_answer_keywords,
        )

        result = {
            "id": item["id"],
            "question": question,
            "source_hit": is_source_hit,
            "retrieval_keyword_hit": is_keyword_hit,
            "retrieval_elapsed": retrieval_elapsed,
            "retrieved_docs": build_retrieved_docs(docs),
            "answer": answer,
            "answer_keyword_recall": answer_recall,
            "hit_answer_keywords": hit_answer_keywords,
            "missed_answer_keywords": missed_answer_keywords,
            "answer_elapsed": answer_elapsed,
        }

        results.append(result)

        print("=" * 80)
        print(f"ID: {item['id']}")
        print(f"问题: {question}")
        print(f"来源命中: {is_source_hit}")
        print(f"检索关键词命中: {is_keyword_hit}")
        print(f"检索耗时: {retrieval_elapsed:.3f}s")
        print(f"答案关键词召回率: {answer_recall:.2f}")
        print(f"命中答案关键词: {hit_answer_keywords}")
        print(f"未命中答案关键词: {missed_answer_keywords}")
        print(f"回答耗时: {answer_elapsed:.3f}s")
        print(f"答案: {answer}")

    total = len(results)

    if total == 0:
        print("评估集为空")
        return

    source_hit_rate = sum(1 for r in results if r["source_hit"]) / total
    retrieval_keyword_hit_rate = sum(1 for r in results if r["retrieval_keyword_hit"]) / total
    avg_retrieval_elapsed = sum(r["retrieval_elapsed"] for r in results) / total
    avg_answer_keyword_recall = sum(r["answer_keyword_recall"] for r in results) / total
    avg_answer_elapsed = sum(r["answer_elapsed"] for r in results) / total

    summary = {
        "total": total,
        "source_hit_at_k": source_hit_rate,
        "retrieval_keyword_hit_at_k": retrieval_keyword_hit_rate,
        "avg_retrieval_elapsed": avg_retrieval_elapsed,
        "avg_answer_keyword_recall": avg_answer_keyword_recall,
        "avg_answer_elapsed": avg_answer_elapsed,
    }

    print("=" * 80)
    print("RAG 评估汇总")
    print(f"总问题数: {total}")
    print(f"Source Hit@K: {source_hit_rate:.2f}")
    print(f"Retrieval Keyword Hit@K: {retrieval_keyword_hit_rate:.2f}")
    print(f"平均检索耗时: {avg_retrieval_elapsed:.3f}s")
    print(f"平均答案关键词召回率: {avg_answer_keyword_recall:.2f}")
    print(f"平均回答耗时: {avg_answer_elapsed:.3f}s")

    write_json_report(summary, results)
    write_markdown_report(summary, results)

    print(f"JSON评估报告已生成: {JSON_REPORT_PATH}")
    print(f"Markdown评估报告已生成: {MD_REPORT_PATH}")


if __name__ == "__main__":
    main()
