import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.retrieval_benchmark import (  # noqa: E402
    evaluate_document,
    first_relevant_rank,
    format_percent,
    load_dataset,
    percentile,
)
from rag.rerank.factory import RerankerFactory  # noqa: E402
from rag.retriever_factory import RetrieverFactory  # noqa: E402
from utils.config_handler import retrieval_conf  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "retrieval_eval_dataset_50.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "rerank_benchmark"
K_VALUES = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 Hybrid 原始排序与 Cross-Encoder Rerank 排序效果"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只测试前 N 条，适合快速验证",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=0,
        help="控制台显示前 N 条重排结果，默认不显示",
    )
    return parser.parse_args()


def validate_config() -> tuple[int, int]:
    mode = str(retrieval_conf.get("retrieval_mode", "")).lower()
    if mode != "hybrid":
        raise ValueError(
            "第三天评估要求 retrieval_mode=hybrid，"
            f"当前配置为 {mode!r}"
        )

    if not retrieval_conf.get("enable_rerank", False):
        raise ValueError("第三天评估要求 enable_rerank=true")

    rerank_type = str(
        retrieval_conf.get("rerank_type", "")
    ).lower()
    if rerank_type != "cross_encoder":
        raise ValueError(
            "第三天评估要求 rerank_type=cross_encoder，"
            f"当前配置为 {rerank_type!r}"
        )

    candidate_k = int(retrieval_conf.get("hybrid_k", 20))
    rerank_top_k = int(retrieval_conf.get("rerank_top_k", 5))

    if candidate_k < rerank_top_k:
        raise ValueError("hybrid_k 不能小于 rerank_top_k")
    if rerank_top_k < max(K_VALUES):
        raise ValueError("为了计算 Hit@5，rerank_top_k 不能小于 5")

    return candidate_k, rerank_top_k


def warm_up_reranker(reranker: Any) -> float:
    from langchain_core.documents import Document

    started_at = time.perf_counter()
    reranker.rerank(
        query="边刷不转怎么办？",
        documents=[
            Document(
                page_content="边刷不转时检查转轴毛发和边刷电机。",
                metadata={"source": "warmup.txt"},
            ),
            Document(
                page_content="滤网应定期清洗并彻底晾干。",
                metadata={"source": "warmup.txt"},
            ),
        ],
        top_k=1,
    )
    return (time.perf_counter() - started_at) * 1000


def evaluate_docs(
    docs: list[Any],
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            **evaluate_document(doc, item),
        }
        for rank, doc in enumerate(docs, start=1)
    ]


def build_reranked_details(
    rerank_results: list[Any],
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    details = []

    for result in rerank_results:
        evaluated = evaluate_document(result.document, item)
        details.append(
            {
                "rank": result.rerank_rank,
                "original_rank": result.original_rank,
                "rerank_score": result.score,
                **evaluated,
            }
        )

    return details


def add_hit_fields(
    result: dict[str, Any],
    before_rank: int | None,
    after_rank: int | None,
) -> None:
    for k in K_VALUES:
        result[f"before_hit_at_{k}"] = bool(
            before_rank is not None and before_rank <= k
        )
        result[f"after_hit_at_{k}"] = bool(
            after_rank is not None and after_rank <= k
        )


def classify_rank_change(
    before_rank: int | None,
    after_rank: int | None,
) -> str:
    if before_rank is None and after_rank is None:
        return "missed"
    if before_rank is None and after_rank is not None:
        return "recovered"
    if before_rank is not None and after_rank is None:
        return "lost"
    if after_rank < before_rank:
        return "improved"
    if after_rank > before_rank:
        return "degraded"
    return "unchanged"


def evaluate_item(
    item: dict[str, Any],
    retriever: Any,
    reranker: Any,
    candidate_k: int,
    rerank_top_k: int,
) -> dict[str, Any]:
    retrieval_started = time.perf_counter()
    candidate_docs = list(
        retriever.retrieve(item["question"]) or []
    )[:candidate_k]
    retrieval_ms = (
        time.perf_counter() - retrieval_started
    ) * 1000

    candidate_details = evaluate_docs(candidate_docs, item)
    before_docs = candidate_docs[:rerank_top_k]
    before_details = evaluate_docs(before_docs, item)

    rerank_started = time.perf_counter()
    rerank_results = reranker.rerank(
        query=item["question"],
        documents=candidate_docs,
        top_k=rerank_top_k,
    )
    rerank_ms = (
        time.perf_counter() - rerank_started
    ) * 1000
    after_details = build_reranked_details(
        rerank_results,
        item,
    )

    candidate_rank = first_relevant_rank(
        candidate_details,
        "strict_relevant",
        candidate_k,
    )
    before_rank = first_relevant_rank(
        before_details,
        "strict_relevant",
        rerank_top_k,
    )
    after_rank = first_relevant_rank(
        after_details,
        "strict_relevant",
        rerank_top_k,
    )

    result = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_sources": item["expected_sources"],
        "expected_keywords": item["expected_keywords"],
        "candidate_count": len(candidate_docs),
        "candidate_first_relevant_rank": candidate_rank,
        "candidate_recall": candidate_rank is not None,
        "before_first_relevant_rank": before_rank,
        "after_first_relevant_rank": after_rank,
        "before_reciprocal_rank": (
            1 / before_rank if before_rank else 0.0
        ),
        "after_reciprocal_rank": (
            1 / after_rank if after_rank else 0.0
        ),
        "rank_change": (
            before_rank - after_rank
            if before_rank is not None and after_rank is not None
            else None
        ),
        "rank_status": classify_rank_change(
            before_rank,
            after_rank,
        ),
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "total_ms": retrieval_ms + rerank_ms,
        "before_top_k": before_details,
        "after_top_k": after_details,
    }
    add_hit_fields(result, before_rank, after_rank)
    return result


def category_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result["category"]].append(result)

    summary = {}
    for category, items in sorted(groups.items()):
        total = len(items)
        summary[category] = {
            "total": total,
            "before_hit_at_5": (
                sum(item["before_hit_at_5"] for item in items)
                / total
            ),
            "after_hit_at_5": (
                sum(item["after_hit_at_5"] for item in items)
                / total
            ),
        }
    return summary


def build_summary(
    results: list[dict[str, Any]],
    candidate_k: int,
    rerank_top_k: int,
    warmup_ms: float,
) -> dict[str, Any]:
    total = len(results)
    retrieval_times = [item["retrieval_ms"] for item in results]
    rerank_times = [item["rerank_ms"] for item in results]
    total_times = [item["total_ms"] for item in results]

    summary: dict[str, Any] = {
        "total": total,
        "candidate_k": candidate_k,
        "rerank_top_k": rerank_top_k,
        "model_name": retrieval_conf.get("rerank_model_name"),
        "device": retrieval_conf.get("rerank_device"),
        "batch_size": retrieval_conf.get("rerank_batch_size"),
        "max_length": retrieval_conf.get("rerank_max_length"),
        "warmup_ms": warmup_ms,
        "candidate_recall_count": sum(
            item["candidate_recall"] for item in results
        ),
        "candidate_recall_rate": sum(
            item["candidate_recall"] for item in results
        ) / total,
        "before_mrr_at_5": sum(
            item["before_reciprocal_rank"] for item in results
        ) / total,
        "after_mrr_at_5": sum(
            item["after_reciprocal_rank"] for item in results
        ) / total,
        "improved_count": sum(
            item["rank_status"] in ("improved", "recovered")
            for item in results
        ),
        "unchanged_count": sum(
            item["rank_status"] == "unchanged"
            for item in results
        ),
        "degraded_count": sum(
            item["rank_status"] in ("degraded", "lost")
            for item in results
        ),
        "missed_count": sum(
            item["rank_status"] == "missed"
            for item in results
        ),
        "avg_retrieval_ms": sum(retrieval_times) / total,
        "avg_rerank_ms": sum(rerank_times) / total,
        "p50_rerank_ms": percentile(rerank_times, 0.50),
        "p95_rerank_ms": percentile(rerank_times, 0.95),
        "avg_total_ms": sum(total_times) / total,
        "p95_total_ms": percentile(total_times, 0.95),
        "category_metrics": category_summary(results),
    }

    for k in K_VALUES:
        before_count = sum(
            item[f"before_hit_at_{k}"] for item in results
        )
        after_count = sum(
            item[f"after_hit_at_{k}"] for item in results
        )
        summary[f"before_hit_count_at_{k}"] = before_count
        summary[f"after_hit_count_at_{k}"] = after_count
        summary[f"before_hit_rate_at_{k}"] = before_count / total
        summary[f"after_hit_rate_at_{k}"] = after_count / total

    return summary


def write_json_report(
    output_dir: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    dataset_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rerank_report.json"

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path.resolve()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "summary": summary,
        "improved_cases": [
            item
            for item in results
            if item["rank_status"] in ("improved", "recovered")
        ],
        "degraded_cases": [
            item
            for item in results
            if item["rank_status"] in ("degraded", "lost")
        ],
        "candidate_misses": [
            item for item in results if not item["candidate_recall"]
        ],
        "details": results,
    }

    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def metric_row(
    label: str,
    before: float,
    after: float,
    percentage: bool = True,
) -> str:
    if percentage:
        before_text = format_percent(before)
        after_text = format_percent(after)
        change_text = f"{(after - before) * 100:+.2f}个百分点"
    else:
        before_text = f"{before:.4f}"
        after_text = f"{after:.4f}"
        change_text = f"{after - before:+.4f}"

    return (
        f"| {label} | {before_text} | "
        f"{after_text} | {change_text} |"
    )


def write_markdown_report(
    output_dir: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rerank_comparison.md"

    lines = [
        "# Rerank 前后对比报告",
        "",
        f"- 测试样本：{summary['total']}条",
        f"- 模型：`{summary['model_name']}`",
        f"- 设备：`{summary['device']}`",
        f"- Hybrid候选数：Top{summary['candidate_k']}",
        f"- 最终上下文数：Top{summary['rerank_top_k']}",
        "",
        "## 核心指标",
        "",
        "| 指标 | Hybrid原始排序 | Cross-Encoder Rerank | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for k in K_VALUES:
        lines.append(
            metric_row(
                f"严格Hit@{k}",
                summary[f"before_hit_rate_at_{k}"],
                summary[f"after_hit_rate_at_{k}"],
            )
        )
    lines.append(
        metric_row(
            "MRR@5",
            summary["before_mrr_at_5"],
            summary["after_mrr_at_5"],
            percentage=False,
        )
    )

    lines.extend(
        [
            "",
            "## 候选与排名变化",
            "",
            f"- Candidate Recall@{summary['candidate_k']}："
            f"{summary['candidate_recall_count']}/{summary['total']} "
            f"({format_percent(summary['candidate_recall_rate'])})",
            f"- 提升或恢复：{summary['improved_count']}条",
            f"- 排名不变：{summary['unchanged_count']}条",
            f"- 退化或丢失：{summary['degraded_count']}条",
            f"- 候选中仍未命中：{summary['missed_count']}条",
            "",
            "## 性能",
            "",
            f"- 模型预热：{summary['warmup_ms']:.1f} ms",
            f"- 平均Hybrid检索：{summary['avg_retrieval_ms']:.1f} ms",
            f"- 平均Rerank：{summary['avg_rerank_ms']:.1f} ms",
            f"- Rerank P50：{summary['p50_rerank_ms']:.1f} ms",
            f"- Rerank P95：{summary['p95_rerank_ms']:.1f} ms",
            f"- 平均检索加重排：{summary['avg_total_ms']:.1f} ms",
            f"- 检索加重排P95：{summary['p95_total_ms']:.1f} ms",
            "",
            "## 分类Hit@5",
            "",
            "| 类别 | Rerank前 | Rerank后 |",
            "| --- | ---: | ---: |",
        ]
    )

    for category, metrics in summary["category_metrics"].items():
        lines.append(
            f"| {category} | "
            f"{format_percent(metrics['before_hit_at_5'])} | "
            f"{format_percent(metrics['after_hit_at_5'])} |"
        )

    improved = [
        item
        for item in results
        if item["rank_status"] in ("improved", "recovered")
    ][:5]
    degraded = [
        item
        for item in results
        if item["rank_status"] in ("degraded", "lost")
    ][:5]

    lines.extend(["", "## 典型提升案例", ""])
    if not improved:
        lines.append("- 无")
    for item in improved:
        lines.append(
            f"- `{item['id']}` {item['question']}："
            f"{item['before_first_relevant_rank']} → "
            f"{item['after_first_relevant_rank']}"
        )

    lines.extend(["", "## 典型退化案例", ""])
    if not degraded:
        lines.append("- 无")
    for item in degraded:
        lines.append(
            f"- `{item['id']}` {item['question']}："
            f"{item['before_first_relevant_rank']} → "
            f"{item['after_first_relevant_rank']}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_result(
    index: int,
    total: int,
    result: dict[str, Any],
    show_top: int,
) -> None:
    print(
        f"[{index:02d}/{total}] {result['id']} "
        f"before={result['before_first_relevant_rank']} "
        f"after={result['after_first_relevant_rank']} "
        f"status={result['rank_status']} "
        f"rerank={result['rerank_ms']:.1f}ms"
    )

    for doc in result["after_top_k"][:show_top]:
        marker = "*" if doc["strict_relevant"] else " "
        print(
            f"  {marker} Top{doc['rank']} "
            f"original={doc['original_rank']} "
            f"score={doc['rerank_score']:.4f} "
            f"source={Path(doc['source']).name}"
        )


def main() -> None:
    args = parse_args()
    candidate_k, rerank_top_k = validate_config()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    dataset = load_dataset(dataset_path, args.limit)

    retriever = RetrieverFactory.create()
    reranker = RerankerFactory.create()

    print("正在预热Cross-Encoder模型...")
    warmup_ms = warm_up_reranker(reranker)
    print(f"模型预热完成：{warmup_ms:.1f} ms")

    results = []
    for index, item in enumerate(dataset, start=1):
        result = evaluate_item(
            item=item,
            retriever=retriever,
            reranker=reranker,
            candidate_k=candidate_k,
            rerank_top_k=rerank_top_k,
        )
        results.append(result)
        print_result(index, len(dataset), result, args.show_top)

    summary = build_summary(
        results=results,
        candidate_k=candidate_k,
        rerank_top_k=rerank_top_k,
        warmup_ms=warmup_ms,
    )
    json_path = write_json_report(
        output_dir,
        summary,
        results,
        dataset_path,
    )
    markdown_path = write_markdown_report(
        output_dir,
        summary,
        results,
    )

    print()
    print("=" * 80)
    print(
        f"Hit@1: "
        f"{format_percent(summary['before_hit_rate_at_1'])} → "
        f"{format_percent(summary['after_hit_rate_at_1'])}"
    )
    print(
        f"Hit@3: "
        f"{format_percent(summary['before_hit_rate_at_3'])} → "
        f"{format_percent(summary['after_hit_rate_at_3'])}"
    )
    print(
        f"Hit@5: "
        f"{format_percent(summary['before_hit_rate_at_5'])} → "
        f"{format_percent(summary['after_hit_rate_at_5'])}"
    )
    print(
        f"MRR@5: {summary['before_mrr_at_5']:.4f} → "
        f"{summary['after_mrr_at_5']:.4f}"
    )
    print(f"JSON报告：{json_path}")
    print(f"Markdown报告：{markdown_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
