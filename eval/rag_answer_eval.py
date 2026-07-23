import json
import time
from pathlib import Path

from rag.rag_service import RagSummerizeService

DATASET_PATH = Path(__file__).parent / "rag_eval_dataset.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_hits(answer: str, expected_keywords: list[str]):
    hit_keywords = []
    missed_keywords = []

    for keyword in expected_keywords:
        if keyword in answer:
            hit_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    if not expected_keywords:
        recall = 0.0
    else:
        recall = len(hit_keywords) / len(expected_keywords)

    return recall, hit_keywords, missed_keywords


def main():
    dataset = load_dataset()
    rag = RagSummerizeService()

    total = len(dataset)
    recall_list = []
    elapsed_list = []

    for item in dataset:
        question = item["question"]
        expected_answer_keywords = item.get("expected_answer_keywords", [])

        start = time.time()
        answer = rag.rag_summerize(question)
        elapsed = time.time() - start

        recall, hit_keywords, missed_keywords = keyword_hits(
            answer,
            expected_answer_keywords
        )

        recall_list.append(recall)
        elapsed_list.append(elapsed)

        print("=" * 80)
        print(f"ID: {item['id']}")
        print(f"问题: {question}")
        print(f"答案: {answer}")
        print(f"命中关键词: {hit_keywords}")
        print(f"关键词召回率: {recall:.2f}")
        print(f"未命中关键词: {missed_keywords}")
        print(f"回答耗时: {elapsed:.3f}s")

    print("=" * 80)
    print("答案评估汇总")
    print(f"总问题数: {total}")
    print(f"平均关键词召回率: {sum(recall_list) / total:.2f}")
    print(f"平均回答耗时: {sum(elapsed_list) / total:.3f}s")


if __name__ == "__main__":
    main()
