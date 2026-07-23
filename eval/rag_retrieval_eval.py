import json
import time
from pathlib import Path

from rag.rag_service import RagSummerizeService

DATASET_PATH = Path(__file__).parent / "rag_eval_dataset.json"


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


def main():
    dataset = load_dataset()
    rag = RagSummerizeService()

    total = len(dataset)
    source_hit_count = 0
    keyword_hit_count = 0
    elapsed_list = []

    for item in dataset:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_keywords = item.get("expected_keywords", [])

        start = time.time()
        docs = rag.retriever_docs(question)
        elapsed = time.time() - start
        elapsed_list.append(elapsed)

        is_source_hit = source_hit(docs, expected_sources)
        is_keyword_hit = keyword_hit(docs, expected_keywords)

        source_hit_count += 1 if is_source_hit else 0
        keyword_hit_count += 1 if is_keyword_hit else 0

        print("=" * 80)
        print(f"ID: {item['id']}")
        print(f"问题: {question}")
        print(f"来源命中: {is_source_hit}")
        print(f"关键词命中: {is_keyword_hit}")
        print(f"检索耗时: {elapsed:.3f}s")

        print("检索结果:")
        for index, doc in enumerate(docs, start=1):
            print(f"[Top{index}] source={doc.metadata.get('source')}")
            print(doc.page_content[:200].replace("\n", " "))
            print("-" * 40)

    print("=" * 80)
    print("评估汇总")
    print(f"总问题数: {total}")
    print(f"Source Hit@K: {source_hit_count / total:.2f}")
    print(f"Keyword Hit@K: {keyword_hit_count / total:.2f}")
    print(f"平均检索耗时: {sum(elapsed_list) / total:.3f}s")


if __name__ == "__main__":
    main()
