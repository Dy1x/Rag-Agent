import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from rag.rag_service import RagSummerizeService
from model.factory import get_chat_model, get_embed_model


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "rag_eval_dataset.json"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "ragas_eval_report.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ragas_dataset():
    raw_items = load_dataset()
    rag = RagSummerizeService()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in raw_items:
        question = item["question"]

        docs = rag.retriever_docs(question)
        retrieved_contexts = [doc.page_content for doc in docs]

        answer = rag.rag_summerize(question)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(item.get("reference", ""))

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    dataset = build_ragas_dataset()

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=get_chat_model(),
        embeddings=get_embed_model(),
    )

    print(result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result.to_pandas().to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"RAGAS评估报告已生成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
