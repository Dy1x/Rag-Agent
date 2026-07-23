import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, context_recall

from rag.rag_service import RagSummerizeService
from model.factory import chat_model, embed_model


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "rag_eval_dataset.json"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "ragas_correctness_report.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset():
    raw_items = load_dataset()
    rag = RagSummerizeService()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in raw_items:
        question = item["question"]
        reference = item.get("reference", "")

        if not reference:
            continue

        docs = rag.retriever_docs(question)
        answer = rag.rag_summerize(question)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(reference)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    dataset = build_dataset()

    result = evaluate(
        dataset=dataset,
        metrics=[
            answer_correctness,
            context_recall,
        ],
        llm=chat_model,
        embeddings=embed_model,
    )

    print(result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = result.to_pandas().to_dict(orient="records")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"RAGAS correctness/context_recall 报告已生成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
