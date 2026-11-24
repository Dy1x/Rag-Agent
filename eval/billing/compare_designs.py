import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from uuid import uuid4

from .checks import CASES
from .mutations import HARD, MATRIX
from .run_evaluation import REPORTS, evaluate_case, evaluation_database, model_settings


def ratio(n, d):
    return {"n": n, "d": d, "rate": n / d if d else None}


def summarize(rows):
    clean = [r for r in rows if not r["mutation"] and r.get("first_correct")]
    injected = [r for r in rows if r.get("injected")]
    tokens = sum(r.get("tokens", 0) for r in rows)
    seconds = sum(r.get("seconds", 0) for r in rows)

    def interception(group):
        return ratio(sum(r["first_rejected"] for r in group), len(group))

    return {
        "correct_outcome": ratio(
            sum(r.get("correct_outcome", False) for r in rows), len(rows)
        ),
        "all_checks": ratio(sum(r.get("passed", False) for r in rows), len(rows)),
        "mutation_interception": interception(injected),
        "hard_interception": interception(
            [r for r in injected if r["mutation"] in HARD]
        ),
        "per_mutation": {
            m: interception([r for r in injected if r["mutation"] == m]) for m in MATRIX
        },
        "false_interception": interception(clean),
        "not_injected": sum(
            bool(r["mutation"]) and not r.get("injected") for r in rows
        ),
        "unauthorized_calls": sum(r.get("unauthorized_calls", 0) for r in rows),
        "mean_revisions": sum(r.get("revisions", 0) for r in rows) / len(rows),
        "llm_calls": sum(r.get("llm_calls", 0) for r in rows),
        "tokens": tokens,
        "seconds": round(seconds, 2),
        "avg_tokens": round(tokens / len(rows), 2),
        "avg_seconds": round(seconds / len(rows), 2),
    }


def markdown(metrics, metadata):
    def fraction(m):
        return f"{m['n']}/{m['d']}"

    lines = [
        "# 充电费用争议架构对照",
        "",
        f"规划模式 `{metadata['planner']}`，配置模型 `{metadata['chat_model_name']}`，每组并发 `{metadata['workers']}`。",
        f"制单只读预算 `{metadata['maker_max_tool_calls']}`，复核只读预算 `{metadata['checker_max_tool_calls']}`，最大修订 `{metadata['max_revisions']}`。",
        "",
        "每组包含 11 个原始场景和 23 个变异样本。A 使用单 Agent 共享对话自查，B 使用制单与硬校验，C 在 B 的基础上加入独立软审核。DeepSeek 模式的 C 另含车主说明正则兜底。",
        "",
        "| 组别 | 结论正确 | 全部检查通过 | 错误拦截 | 硬性错误拦截 | 误拦截 | 未注入 | 越权调用 | 平均修订 | 模型调用 | Token | 秒 | 平均 Token | 平均秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for design, m in metrics.items():
        items = [
            design,
            *[
                fraction(m[k])
                for k in (
                    "correct_outcome",
                    "all_checks",
                    "mutation_interception",
                    "hard_interception",
                    "false_interception",
                )
            ],
            m["not_injected"],
            m["unauthorized_calls"],
            round(m["mean_revisions"], 2),
            m["llm_calls"],
            m["tokens"],
            m["seconds"],
            m["avg_tokens"],
            m["avg_seconds"],
        ]
        lines.append("| " + " | ".join(map(str, items)) + " |")
    lines.extend(["", "| 变异 | A | B | C |", "|---|---:|---:|---:|"])
    for mutation in MATRIX:
        lines.append(
            "| "
            + " | ".join(
                [
                    mutation,
                    *[fraction(m["per_mutation"][mutation]) for m in metrics.values()],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "错误拦截按注入后的首版审核计数。误拦截分母是未变异、已进入审核且首版符合场景期望的方案。结论正确按最终状态、动作目标和金额计数；全部检查还覆盖证据与车主说明。",
            "未注入样本保留在组内总样本数中。Token 和秒数为样本记录之和，均值按该组全部样本计算。秒数包含同组并行样本各自的耗时。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_design(engine, samples, planner, design, workers):
    def run(name, mutation):
        started = perf_counter()
        try:
            return evaluate_case(
                engine, name, planner, design=design, mutation=mutation
            )
        except Exception as exc:
            return dict(
                name=name,
                design=design,
                mutation=mutation,
                passed=False,
                errors=[f"{type(exc).__name__}: {exc}"],
                seconds=round(perf_counter() - started, 3),
                **(
                    {"not_injected_reason": "样本执行异常，注入状态未记录"}
                    if mutation
                    else {}
                ),
            )

    rows = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(run, name, mutation): index
            for index, (name, mutation) in enumerate(samples)
        }
        for completed, future in enumerate(as_completed(pending), 1):
            row = future.result()
            rows[pending[future]] = row
            print(
                f"[{design} {completed}/{len(samples)}] {row['name']} {row['mutation'] or 'clean'} "
                f"rejected={row.get('first_rejected', False)} "
                f"outcome={row.get('correct_outcome', False)} "
                f"{row['seconds']:.1f}s {row.get('tokens', 0)}tok",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["offline", "deepseek"], default="offline")
    parser.add_argument(
        "--workers", type=int, help="每组并发数（DeepSeek 默认 3，离线默认 1）"
    )
    args = parser.parse_args()
    workers = (
        args.workers
        if args.workers is not None
        else (3 if args.planner == "deepseek" else 1)
    )
    if workers < 1:
        parser.error("--workers 必须为正整数")
    metadata = dict(planner=args.planner, workers=workers, **model_settings())
    samples = [(s, None) for s in CASES] + [
        (s, m) for m, names in MATRIX.items() for s in names
    ]
    results = {}
    with evaluation_database() as engine:
        for design in "ABC":
            results[design] = run_design(engine, samples, args.planner, design, workers)
    metrics = {d: summarize(rows) for d, rows in results.items()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"billing-compare-{args.planner}-{uuid4().hex[:8]}"
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                planner=args.planner, metadata=metadata, metrics=metrics, runs=results
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text(markdown(metrics, metadata), encoding="utf-8")
    print(path.with_suffix(".md"))
    full = metrics["C"]
    return int(
        args.planner == "offline"
        and (
            full["hard_interception"]["rate"] != 1
            or full["false_interception"]["rate"] != 0
            or full["all_checks"]["rate"] != 1
            or full["not_injected"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
