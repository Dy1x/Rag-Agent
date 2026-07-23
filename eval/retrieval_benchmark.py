import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "retrieval_eval_dataset_50.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "retrieval_benchmark"
SUPPORTED_MODES = ("vector", "bm25", "hybrid")
DEFAULT_K_VALUES = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "离线评估 vector、BM25 和 hybrid 检索效果，"
            "生成 Hit@K、MRR、延迟及失败案例报告。"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"评估数据集路径，默认：{DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"报告输出目录，默认：{DEFAULT_REPORT_DIR}",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=list(SUPPORTED_MODES),
        help="需要评估的检索模式，默认同时测试 vector bm25 hybrid",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=5,
        help="每种模式最终返回的候选文档数，默认 5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只测试前 N 条，适合快速调试；默认测试完整数据集",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=3,
        help="控制台为每个问题展示的 TopN 结果数量，默认 3；设为 0 可关闭",
    )
    return parser.parse_args()


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("评估数据集的根节点必须是 JSON 数组")

    required_fields = {
        "id",
        "category",
        "question",
        "expected_sources",
        "expected_keywords",
    }
    seen_ids: set[str] = set()

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条数据不是 JSON 对象")

        missing_fields = required_fields - set(item)
        if missing_fields:
            raise ValueError(
                f"第 {index} 条数据缺少字段：{sorted(missing_fields)}"
            )

        item_id = str(item["id"])
        if item_id in seen_ids:
            raise ValueError(f"评估数据存在重复 id：{item_id}")
        seen_ids.add(item_id)

        if not item["expected_sources"]:
            raise ValueError(f"{item_id} 的 expected_sources 不能为空")
        if not item["expected_keywords"]:
            raise ValueError(f"{item_id} 的 expected_keywords 不能为空")

        min_matches = int(item.get("min_keyword_matches", 1))
        if min_matches < 1 or min_matches > len(item["expected_keywords"]):
            raise ValueError(
                f"{item_id} 的 min_keyword_matches 必须位于 1 到关键词数量之间"
            )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0")
        dataset = dataset[:limit]

    if not dataset:
        raise ValueError("评估数据集为空")

    return dataset


def normalize_source(source: Any) -> str:
    return str(source or "").replace("\\", "/").lower()


def evaluate_document(doc: Any, item: dict[str, Any]) -> dict[str, Any]:
    source = normalize_source(getattr(doc, "metadata", {}).get("source", ""))
    content = str(getattr(doc, "page_content", "") or "")

    matched_sources = [
        expected
        for expected in item["expected_sources"]
        if normalize_source(expected) in source
    ]
    matched_keywords = [
        keyword
        for keyword in item["expected_keywords"]
        if keyword.lower() in content.lower()
    ]

    min_keyword_matches = int(item.get("min_keyword_matches", 1))
    source_relevant = bool(matched_sources)
    strict_relevant = (
        source_relevant and len(matched_keywords) >= min_keyword_matches
    )

    return {
        "source": source,
        "content_preview": content[:240].replace("\n", " "),
        "matched_sources": matched_sources,
        "matched_keywords": matched_keywords,
        "source_relevant": source_relevant,
        "strict_relevant": strict_relevant,
    }


def first_relevant_rank(
    evaluated_docs: list[dict[str, Any]],
    relevance_field: str,
    max_k: int,
) -> int | None:
    for rank, doc in enumerate(evaluated_docs[:max_k], start=1):
        if doc[relevance_field]:
            return rank
    return None


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def calculate_category_metrics(
    results: list[dict[str, Any]],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)

    category_metrics: dict[str, dict[str, Any]] = {}
    for category, category_results in sorted(grouped.items()):
        total = len(category_results)
        metrics: dict[str, Any] = {"total": total}

        for k in k_values:
            strict_hits = sum(
                result[f"strict_hit_at_{k}"] for result in category_results
            )
            metrics[f"strict_hit_count_at_{k}"] = strict_hits
            metrics[f"strict_hit_rate_at_{k}"] = strict_hits / total

        category_metrics[category] = metrics

    return category_metrics


def calculate_summary(
    mode: str,
    results: list[dict[str, Any]],
    k_values: tuple[int, ...],
    candidate_k: int,
) -> dict[str, Any]:
    total = len(results)
    latencies = [result["latency_ms"] for result in results]
    summary: dict[str, Any] = {
        "mode": mode,
        "total": total,
        "candidate_k": candidate_k,
        "avg_latency_ms": sum(latencies) / total,
        "p95_latency_ms": percentile(latencies, 0.95),
        "mrr_at_k": sum(result["reciprocal_rank"] for result in results) / total,
    }

    for k in k_values:
        source_hits = sum(result[f"source_hit_at_{k}"] for result in results)
        strict_hits = sum(result[f"strict_hit_at_{k}"] for result in results)

        summary[f"source_hit_count_at_{k}"] = source_hits
        summary[f"source_hit_rate_at_{k}"] = source_hits / total
        summary[f"strict_hit_count_at_{k}"] = strict_hits
        summary[f"strict_hit_rate_at_{k}"] = strict_hits / total

    summary["category_metrics"] = calculate_category_metrics(results, k_values)
    return summary


def build_retriever(mode: str, candidate_k: int):
    try:
        from rag.retriever_factory import RetrieverFactory
        from utils.config_handler import retrieval_conf
    except ImportError as exc:
        raise RuntimeError(
            "无法导入项目检索模块。请先在项目使用的 Python 环境中安装依赖，"
            "并从项目根目录执行本脚本。"
        ) from exc

    retrieval_conf["retrieval_mode"] = mode
    retrieval_conf["vector_k"] = candidate_k
    retrieval_conf["bm25_k"] = candidate_k
    retrieval_conf["hybrid_k"] = candidate_k

    if mode == "bm25":
        retrieval_conf["enable_bm25_retriever"] = True
    if mode == "hybrid":
        retrieval_conf["enable_bm25_retriever"] = True
        retrieval_conf["enable_hybrid_retriever"] = True

    return RetrieverFactory.create(), retrieval_conf


def evaluate_mode(
    mode: str,
    dataset: list[dict[str, Any]],
    candidate_k: int,
    show_top: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever, _ = build_retriever(mode, candidate_k)
    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= candidate_k)
    max_k = max(k_values)
    results: list[dict[str, Any]] = []

    print()
    print("=" * 90)
    print(f"开始评估模式：{mode}，样本数：{len(dataset)}，TopK：{candidate_k}")
    print("=" * 90)

    for index, item in enumerate(dataset, start=1):
        started_at = time.perf_counter()
        docs = retriever.retrieve(item["question"])
        latency_ms = (time.perf_counter() - started_at) * 1000
        docs = list(docs or [])[:candidate_k]

        evaluated_docs = [
            {
                "rank": rank,
                **evaluate_document(doc, item),
            }
            for rank, doc in enumerate(docs, start=1)
        ]

        source_rank = first_relevant_rank(
            evaluated_docs,
            "source_relevant",
            max_k,
        )
        strict_rank = first_relevant_rank(
            evaluated_docs,
            "strict_relevant",
            max_k,
        )

        result: dict[str, Any] = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_sources": item["expected_sources"],
            "expected_keywords": item["expected_keywords"],
            "min_keyword_matches": item.get("min_keyword_matches", 1),
            "latency_ms": latency_ms,
            "returned_count": len(evaluated_docs),
            "first_source_relevant_rank": source_rank,
            "first_strict_relevant_rank": strict_rank,
            "reciprocal_rank": 1 / strict_rank if strict_rank else 0.0,
            "retrieved_docs": evaluated_docs,
        }

        for k in k_values:
            result[f"source_hit_at_{k}"] = bool(
                source_rank is not None and source_rank <= k
            )
            result[f"strict_hit_at_{k}"] = bool(
                strict_rank is not None and strict_rank <= k
            )

        results.append(result)

        hit_text = "命中" if result[f"strict_hit_at_{max_k}"] else "未命中"
        rank_text = strict_rank if strict_rank is not None else "-"
        print(
            f"[{index:02d}/{len(dataset)}] {item['id']} "
            f"{hit_text} rank={rank_text} latency={latency_ms:.1f}ms "
            f"| {item['question']}"
        )

        if show_top > 0:
            for doc in evaluated_docs[:show_top]:
                source_name = Path(doc["source"]).name
                marker = "*" if doc["strict_relevant"] else " "
                print(
                    f"  {marker} Top{doc['rank']} {source_name} "
                    f"keywords={doc['matched_keywords']}"
                )

    summary = calculate_summary(mode, results, k_values, candidate_k)
    return summary, results


def write_mode_json_report(
    output_dir: Path,
    mode: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    dataset_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{mode}_retrieval_report.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path.resolve()),
        "summary": summary,
        "failures": [
            result
            for result in results
            if not result.get(
                f"strict_hit_at_{min(5, summary['candidate_k'])}",
                False,
            )
        ],
        "details": results,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report_path


def write_comparison_json(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    dataset_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retrieval_comparison.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "dataset": str(dataset_path.resolve()),
                "summaries": summaries,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report_path


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_comparison_markdown(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    dataset_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retrieval_comparison.md"
    k_values = tuple(
        k
        for k in DEFAULT_K_VALUES
        if all(k <= summary["candidate_k"] for summary in summaries)
    )

    lines = [
        "# 检索召回评估报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 数据集：`{dataset_path.resolve()}`",
        f"- 测试样本数：{summaries[0]['total']}",
        f"- 候选文档数：Top{summaries[0]['candidate_k']}",
        "",
        "## 模式对比",
        "",
    ]

    headers = ["模式"]
    for k in k_values:
        headers.append(f"严格 Hit@{k}")
    headers.extend(["MRR", "平均延迟", "P95 延迟"])

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for summary in summaries:
        row = [summary["mode"]]
        for k in k_values:
            row.append(
                f"{summary[f'strict_hit_count_at_{k}']}/{summary['total']} "
                f"({format_percent(summary[f'strict_hit_rate_at_{k}'])})"
            )
        row.extend(
            [
                f"{summary['mrr_at_k']:.4f}",
                f"{summary['avg_latency_ms']:.1f} ms",
                f"{summary['p95_latency_ms']:.1f} ms",
            ]
        )
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "严格命中定义：TopK 中至少存在一个 chunk，"
            "其来源属于标注文件，并且命中样本要求数量的答案关键词。",
            "",
            "## 分类表现",
            "",
        ]
    )

    target_k = max(k_values)
    categories = sorted(
        {
            category
            for summary in summaries
            for category in summary["category_metrics"]
        }
    )
    category_headers = ["类别"] + [
        f"{summary['mode']} Hit@{target_k}" for summary in summaries
    ]
    lines.append("| " + " | ".join(category_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(category_headers)) + " |")

    for category in categories:
        row = [category]
        for summary in summaries:
            metrics = summary["category_metrics"].get(category)
            if not metrics:
                row.append("-")
                continue
            row.append(
                f"{metrics[f'strict_hit_count_at_{target_k}']}/"
                f"{metrics['total']} "
                f"({format_percent(metrics[f'strict_hit_rate_at_{target_k}'])})"
            )
        lines.append("| " + " | ".join(row) + " |")

    best_summary = max(
        summaries,
        key=lambda item: (
            item[f"strict_hit_rate_at_{target_k}"],
            item["mrr_at_k"],
            -item["avg_latency_ms"],
        ),
    )
    lines.extend(
        [
            "",
            "## 结果摘要",
            "",
            f"- 当前严格 Hit@{target_k} 最佳模式："
            f"**{best_summary['mode']}**，"
            f"{best_summary[f'strict_hit_count_at_{target_k}']}/"
            f"{best_summary['total']}，"
            f"命中率 {format_percent(best_summary[f'strict_hit_rate_at_{target_k}'])}。",
            f"- 最佳模式 MRR@{target_k}：{best_summary['mrr_at_k']:.4f}。",
            f"- 最佳模式平均检索耗时：{best_summary['avg_latency_ms']:.1f} ms。",
            "",
            "各模式的完整查询结果和失败案例位于同目录 JSON 报告中。",
        ]
    )

    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return report_path


def print_summary(summary: dict[str, Any]) -> None:
    candidate_k = summary["candidate_k"]
    target_k = min(5, candidate_k)

    print()
    print("-" * 90)
    print(f"{summary['mode']} 评估完成")
    for k in DEFAULT_K_VALUES:
        if k > candidate_k:
            continue
        print(
            f"严格 Hit@{k}: "
            f"{summary[f'strict_hit_count_at_{k}']}/{summary['total']} "
            f"({format_percent(summary[f'strict_hit_rate_at_{k}'])})"
        )
    print(f"MRR@{target_k}: {summary['mrr_at_k']:.4f}")
    print(f"平均延迟: {summary['avg_latency_ms']:.1f} ms")
    print(f"P95 延迟: {summary['p95_latency_ms']:.1f} ms")
    print("-" * 90)


def main() -> None:
    args = parse_args()
    if args.candidate_k < 1:
        raise ValueError("--candidate-k 必须大于等于 1")
    if args.candidate_k < max(DEFAULT_K_VALUES):
        raise ValueError(
            f"为了计算 Hit@{max(DEFAULT_K_VALUES)}，"
            f"--candidate-k 不能小于 {max(DEFAULT_K_VALUES)}"
        )
    if args.show_top < 0:
        raise ValueError("--show-top 不能小于 0")

    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    dataset = load_dataset(dataset_path, args.limit)
    summaries: list[dict[str, Any]] = []

    try:
        from utils.config_handler import retrieval_conf
    except ImportError as exc:
        raise RuntimeError(
            "无法加载项目配置，请在项目依赖环境中运行本脚本"
        ) from exc

    original_config = dict(retrieval_conf)

    try:
        for mode in args.modes:
            summary, results = evaluate_mode(
                mode=mode,
                dataset=dataset,
                candidate_k=args.candidate_k,
                show_top=args.show_top,
            )
            summaries.append(summary)
            report_path = write_mode_json_report(
                output_dir=output_dir,
                mode=mode,
                summary=summary,
                results=results,
                dataset_path=dataset_path,
            )
            print_summary(summary)
            print(f"详细报告：{report_path}")
    finally:
        retrieval_conf.clear()
        retrieval_conf.update(original_config)

    comparison_json = write_comparison_json(
        output_dir,
        summaries,
        dataset_path,
    )
    comparison_markdown = write_comparison_markdown(
        output_dir,
        summaries,
        dataset_path,
    )

    print()
    print("=" * 90)
    print("全部评估完成")
    print(f"对比 JSON：{comparison_json}")
    print(f"对比 Markdown：{comparison_markdown}")
    print("=" * 90)


if __name__ == "__main__":
    main()
