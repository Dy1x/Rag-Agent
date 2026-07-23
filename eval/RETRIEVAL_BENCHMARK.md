# 检索召回评估使用说明

## 评估内容

`retrieval_benchmark.py` 使用同一份 50 条业务问题数据，依次评估：

- Vector 向量检索
- BM25 关键词检索
- Hybrid 混合检索

脚本只测试检索召回，不调用大模型生成答案。

主要指标：

- Hit@1：第一条结果是否包含正确 chunk
- Hit@3：前三条结果是否包含正确 chunk
- Hit@5：前五条结果是否包含正确 chunk
- MRR@5：正确 chunk 的排名是否足够靠前
- 平均检索耗时
- P95 检索耗时

严格命中需要同时满足：

1. chunk 的 `source` 属于样本标注的正确知识库文件。
2. chunk 内容命中样本要求数量的答案关键词。

## 运行前准备

1. 使用 Python 3.10 或更高版本。
2. 安装项目现有依赖。
3. 确认 Chroma 知识库已经完成入库。
4. 确认 `config/retrieval.yml` 中 BM25 和 Hybrid 开关已开启。

## 一键运行

在项目根目录执行：

```powershell
python .\eval\retrieval_benchmark.py
```

该命令会完整测试 50 条问题和三种检索模式。

快速测试前 5 条：

```powershell
python .\eval\retrieval_benchmark.py --limit 5
```

只测试 Hybrid：

```powershell
python .\eval\retrieval_benchmark.py --modes hybrid
```

关闭控制台 Top3 详情：

```powershell
python .\eval\retrieval_benchmark.py --show-top 0
```

将每种模式的候选数量调整为 Top10：

```powershell
python .\eval\retrieval_benchmark.py --candidate-k 10
```

## 输出文件

报告默认生成到：

```text
eval/reports/retrieval_benchmark/
```

其中包括：

```text
vector_retrieval_report.json
bm25_retrieval_report.json
hybrid_retrieval_report.json
retrieval_comparison.json
retrieval_comparison.md
```

优先查看 `retrieval_comparison.md`。各模式未命中的具体问题和召回内容，可以在对应 JSON 报告的 `failures` 字段中查看。

## 简历描述模板

请在实际运行后替换数字：

> 构建包含 50 条业务问题的离线检索评估集，引入 Hit@K、MRR 和 P95 延迟指标，对比 Vector、BM25 与 Hybrid 检索方案；Hybrid 检索 Hit@5 达到 XX%，较单向量检索提升 XX 个百分点。
