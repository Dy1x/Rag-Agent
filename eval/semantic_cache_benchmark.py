import argparse
import json
import math
import platform
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.semantic_cache_handler import SemanticCache  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "semantic_cache_eval_dataset_100.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports" / "semantic_cache_benchmark"
DEFAULT_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估语义缓存命中准确率、召回率、误命中率、安全拦截和延迟"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--rag-baseline-ms",
        type=float,
        default=None,
        help="已知的完整RAG平均耗时，单位毫秒",
    )
    parser.add_argument(
        "--measure-rag-baseline",
        action="store_true",
        help="实际调用完整RAG测量基线耗时，会调用检索、Rerank和LLM",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=3,
        help="实测完整RAG时使用的基准问题数量，默认3",
    )
    parser.add_argument(
        "--keep-test-collection",
        action="store_true",
        help="运行结束后保留隔离的测试集合",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base_entries = data.get("base_entries")
    queries = data.get("queries")

    if not isinstance(base_entries, list) or not isinstance(queries, list):
        raise ValueError("数据集必须包含base_entries和queries数组")

    base_ids = [item.get("base_id") for item in base_entries]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base_entries存在重复base_id")

    query_ids = [item.get("id") for item in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("queries存在重复id")

    type_counts = defaultdict(int)
    for item in queries:
        query_type = item.get("type")
        if query_type not in {
            "semantic_hit",
            "semantic_miss",
            "safety_bypass",
        }:
            raise ValueError(
                f"不支持的查询类型：{query_type}"
            )
        type_counts[query_type] += 1

        if (
            query_type == "semantic_hit"
            and item.get("expected_base_id") not in base_ids
        ):
            raise ValueError(
                f"{item.get('id')}引用了不存在的expected_base_id"
            )

    expected_counts = {
        "semantic_hit": 50,
        "semantic_miss": 30,
        "safety_bypass": 20,
    }
    if dict(type_counts) != expected_counts:
        raise ValueError(
            f"评估集类型数量错误：{dict(type_counts)}，"
            f"期望：{expected_counts}"
        )

    return data


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent) - 1)
    return ordered[index]


def seed_cache(
    cache: SemanticCache,
    base_entries: list[dict[str, Any]],
) -> None:
    for item in base_entries:
        saved = cache.add_cache(
            query=item["query"],
            answer=item["answer"],
            metadata={"base_id": item["base_id"]},
        )
        if not saved:
            raise RuntimeError(
                f"基准缓存写入失败：{item['base_id']}"
            )


def raw_lookup(
    cache: SemanticCache,
    query: str,
) -> tuple[dict[str, Any] | None, float]:
    started_at = time.perf_counter()
    results = cache.vector_store.similarity_search_with_relevance_scores(
        query,
        k=1,
    )
    latency_ms = (time.perf_counter() - started_at) * 1000

    if not results:
        return None, latency_ms

    doc, score = results[0]
    return {
        "cached_query": doc.metadata.get("query"),
        "cached_base_id": doc.metadata.get("base_id"),
        "score": float(score),
    }, latency_ms


def collect_lookup_results(
    cache: SemanticCache,
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for index, item in enumerate(queries, start=1):
        safe = cache.is_semantic_cache_safe_query(item["query"])

        if not safe:
            lookup = None
            latency_started = time.perf_counter()
            cache.safe_search(item["query"])
            latency_ms = (
                time.perf_counter() - latency_started
            ) * 1000
        else:
            lookup, latency_ms = raw_lookup(cache, item["query"])

        result = {
            **item,
            "safe_query": safe,
            "lookup_latency_ms": latency_ms,
            "nearest_cached_query": (
                lookup["cached_query"] if lookup else None
            ),
            "nearest_base_id": (
                lookup["cached_base_id"] if lookup else None
            ),
            "similarity_score": (
                lookup["score"] if lookup else None
            ),
        }
        results.append(result)

        score_text = (
            f"{result['similarity_score']:.4f}"
            if result["similarity_score"] is not None
            else "-"
        )
        print(
            f"[{index:03d}/{len(queries)}] "
            f"{item['id']} type={item['type']} "
            f"safe={safe} score={score_text} "
            f"latency={latency_ms:.1f}ms"
        )

    return results


def evaluate_threshold(
    lookup_results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    positive_items = [
        item
        for item in lookup_results
        if item["type"] == "semantic_hit"
    ]
    negative_items = [
        item
        for item in lookup_results
        if item["type"] == "semantic_miss"
    ]
    unsafe_items = [
        item
        for item in lookup_results
        if item["type"] == "safety_bypass"
    ]

    correct_hits = 0
    wrong_base_hits = 0
    missed_positives = 0
    negative_false_hits = 0
    negative_correct_misses = 0
    total_cache_hits = 0

    details = []
    for item in lookup_results:
        score = item["similarity_score"]
        cache_hit = bool(
            item["safe_query"]
            and score is not None
            and score >= threshold
        )
        correct = False
        outcome = ""

        if item["type"] == "semantic_hit":
            if cache_hit:
                total_cache_hits += 1
                correct = (
                    item["nearest_base_id"]
                    == item["expected_base_id"]
                )
                if correct:
                    correct_hits += 1
                    outcome = "correct_hit"
                else:
                    wrong_base_hits += 1
                    outcome = "wrong_base_hit"
            else:
                missed_positives += 1
                outcome = "missed_positive"

        elif item["type"] == "semantic_miss":
            if cache_hit:
                total_cache_hits += 1
                negative_false_hits += 1
                outcome = "false_hit"
            else:
                negative_correct_misses += 1
                correct = True
                outcome = "correct_miss"

        else:
            correct = not item["safe_query"] and not cache_hit
            outcome = (
                "safety_bypassed"
                if correct
                else "safety_failure"
            )

        details.append(
            {
                **item,
                "threshold": threshold,
                "cache_hit": cache_hit,
                "correct": correct,
                "outcome": outcome,
            }
        )

    precision = (
        correct_hits / total_cache_hits
        if total_cache_hits
        else 0.0
    )
    recall = correct_hits / len(positive_items)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    false_hit_rate = (
        negative_false_hits / len(negative_items)
    )
    safety_bypass_count = sum(
        not item["safe_query"] for item in unsafe_items
    )
    safety_bypass_rate = (
        safety_bypass_count / len(unsafe_items)
    )

    return {
        "threshold": threshold,
        "positive_total": len(positive_items),
        "negative_total": len(negative_items),
        "unsafe_total": len(unsafe_items),
        "correct_hits": correct_hits,
        "wrong_base_hits": wrong_base_hits,
        "missed_positives": missed_positives,
        "negative_false_hits": negative_false_hits,
        "negative_correct_misses": negative_correct_misses,
        "total_cache_hits": total_cache_hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_hit_rate": false_hit_rate,
        "safety_bypass_count": safety_bypass_count,
        "safety_bypass_rate": safety_bypass_rate,
        "workload_avoidance_rate": (
            correct_hits / len(lookup_results)
        ),
        "details": details,
    }


def measure_full_rag_baseline(
    base_entries: list[dict[str, Any]],
    sample_count: int,
) -> tuple[float, list[dict[str, Any]]]:
    from rag.rag_service import RagSummerizeService

    if sample_count <= 0:
        raise ValueError("baseline_samples必须大于0")

    rag = RagSummerizeService()
    samples = []

    for item in base_entries[:sample_count]:
        started_at = time.perf_counter()
        rag.rag_summerize_without_cache(item["query"])
        latency_ms = (time.perf_counter() - started_at) * 1000
        samples.append(
            {
                "base_id": item["base_id"],
                "query": item["query"],
                "latency_ms": latency_ms,
            }
        )
        print(
            f"[RAG基线] {item['base_id']} "
            f"latency={latency_ms:.1f}ms"
        )

    average_ms = (
        sum(item["latency_ms"] for item in samples)
        / len(samples)
    )
    return average_ms, samples


def build_latency_summary(
    lookup_results: list[dict[str, Any]],
    rag_baseline_ms: float | None,
) -> dict[str, Any]:
    safe_latencies = [
        item["lookup_latency_ms"]
        for item in lookup_results
        if item["safe_query"]
    ]
    bypass_latencies = [
        item["lookup_latency_ms"]
        for item in lookup_results
        if not item["safe_query"]
    ]

    average_lookup_ms = sum(safe_latencies) / len(safe_latencies)
    summary = {
        "avg_semantic_lookup_ms": average_lookup_ms,
        "p50_semantic_lookup_ms": percentile(
            safe_latencies,
            0.50,
        ),
        "p95_semantic_lookup_ms": percentile(
            safe_latencies,
            0.95,
        ),
        "avg_safety_bypass_ms": (
            sum(bypass_latencies) / len(bypass_latencies)
        ),
        "rag_baseline_ms": rag_baseline_ms,
        "latency_reduction_rate": None,
    }

    if rag_baseline_ms and rag_baseline_ms > 0:
        summary["latency_reduction_rate"] = (
            rag_baseline_ms - average_lookup_ms
        ) / rag_baseline_ms

    return summary


def select_recommended_threshold(
    threshold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        item
        for item in threshold_results
        if item["precision"] >= 0.95
        and item["false_hit_rate"] <= 0.05
    ]
    candidates = eligible or threshold_results
    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["precision"],
            -item["false_hit_rate"],
        ),
    )


def write_json_report(
    output_dir: Path,
    dataset_path: Path,
    threshold_results: list[dict[str, Any]],
    latency_summary: dict[str, Any],
    recommended: dict[str, Any],
    rag_baseline_samples: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "semantic_cache_report.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path.resolve()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "recommended_threshold": recommended["threshold"],
        "latency": latency_summary,
        "rag_baseline_samples": rag_baseline_samples,
        "threshold_results": threshold_results,
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_markdown_report(
    output_dir: Path,
    threshold_results: list[dict[str, Any]],
    latency: dict[str, Any],
    recommended: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "semantic_cache_comparison.md"

    lines = [
        "# 语义缓存离线评估报告",
        "",
        "## 阈值对比",
        "",
        "| 阈值 | Precision | Recall | F1 | 负样本错误命中率 | 安全拦截率 | 正确调用避免率 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in threshold_results:
        lines.append(
            f"| {item['threshold']:.2f} "
            f"| {fmt_percent(item['precision'])} "
            f"| {fmt_percent(item['recall'])} "
            f"| {item['f1']:.4f} "
            f"| {fmt_percent(item['false_hit_rate'])} "
            f"| {fmt_percent(item['safety_bypass_rate'])} "
            f"| {fmt_percent(item['workload_avoidance_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 推荐结果",
            "",
            f"- 推荐阈值：**{recommended['threshold']:.2f}**",
            f"- 缓存命中准确率："
            f"{recommended['correct_hits']}/"
            f"{recommended['total_cache_hits']} "
            f"({fmt_percent(recommended['precision'])})",
            f"- 相似改写召回率："
            f"{recommended['correct_hits']}/"
            f"{recommended['positive_total']} "
            f"({fmt_percent(recommended['recall'])})",
            f"- 负样本错误命中："
            f"{recommended['negative_false_hits']}/"
            f"{recommended['negative_total']} "
            f"({fmt_percent(recommended['false_hit_rate'])})",
            f"- 动态问题安全拦截："
            f"{recommended['safety_bypass_count']}/"
            f"{recommended['unsafe_total']} "
            f"({fmt_percent(recommended['safety_bypass_rate'])})",
            f"- 100条混合负载中正确避免完整RAG调用："
            f"{recommended['correct_hits']}/100 "
            f"({fmt_percent(recommended['workload_avoidance_rate'])})",
            "",
            "## 延迟",
            "",
            f"- 语义缓存平均查询耗时："
            f"{latency['avg_semantic_lookup_ms']:.1f} ms",
            f"- 语义缓存P50："
            f"{latency['p50_semantic_lookup_ms']:.1f} ms",
            f"- 语义缓存P95："
            f"{latency['p95_semantic_lookup_ms']:.1f} ms",
        ]
    )

    if latency["rag_baseline_ms"]:
        lines.extend(
            [
                f"- 完整RAG平均耗时："
                f"{latency['rag_baseline_ms']:.1f} ms",
                f"- 命中语义缓存后的延迟降低："
                f"{fmt_percent(latency['latency_reduction_rate'])}",
            ]
        )
    else:
        lines.append(
            "- 未提供完整RAG基线，暂不计算延迟降低比例。"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    data = load_dataset(dataset_path)

    thresholds = sorted(set(args.thresholds))
    if any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("threshold必须位于0到1之间")

    collection_name = (
        f"semantic_cache_benchmark_{uuid.uuid4().hex[:12]}"
    )
    cache = SemanticCache(
        collection_name=collection_name,
        persist_directory="chroma_db_semantic_cache_benchmark",
        threshold=min(thresholds),
        top_k=1,
        ttl_seconds=86400,
    )

    rag_baseline_ms = args.rag_baseline_ms
    rag_baseline_samples = []

    try:
        print(f"测试集合：{collection_name}")
        seed_cache(cache, data["base_entries"])
        print("基准缓存写入完成，开始评估100条查询...")

        lookup_results = collect_lookup_results(
            cache,
            data["queries"],
        )

        threshold_results = [
            evaluate_threshold(lookup_results, threshold)
            for threshold in thresholds
        ]
        recommended = select_recommended_threshold(
            threshold_results
        )

        if args.measure_rag_baseline:
            rag_baseline_ms, rag_baseline_samples = (
                measure_full_rag_baseline(
                    data["base_entries"],
                    args.baseline_samples,
                )
            )

        latency_summary = build_latency_summary(
            lookup_results,
            rag_baseline_ms,
        )
        json_path = write_json_report(
            output_dir,
            dataset_path,
            threshold_results,
            latency_summary,
            recommended,
            rag_baseline_samples,
        )
        markdown_path = write_markdown_report(
            output_dir,
            threshold_results,
            latency_summary,
            recommended,
        )

        print()
        print("=" * 80)
        print(f"推荐阈值：{recommended['threshold']:.2f}")
        print(
            f"Precision：{fmt_percent(recommended['precision'])}"
        )
        print(
            f"Recall：{fmt_percent(recommended['recall'])}"
        )
        print(
            "错误命中率："
            f"{fmt_percent(recommended['false_hit_rate'])}"
        )
        print(
            "安全拦截率："
            f"{fmt_percent(recommended['safety_bypass_rate'])}"
        )
        print(f"JSON报告：{json_path}")
        print(f"Markdown报告：{markdown_path}")
        print("=" * 80)

    finally:
        if not args.keep_test_collection:
            cache.delete_collection()


if __name__ == "__main__":
    main()
