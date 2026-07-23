# 语义缓存量化评估使用说明

## 一、评估目标

本评估用于回答以下问题：

1. 相同语义、不同表达的问题，语义缓存能否正确命中。
2. 含义相近但答案不同的问题，是否会发生错误命中。
3. 用户、设备、订单、天气等动态问题，是否会绕过语义缓存。
4. 不同相似度阈值下，Precision、Recall、F1 和误命中率如何变化。
5. 命中语义缓存后，相比完整 RAG 调用能降低多少延迟。

## 二、评估数据组成

`semantic_cache_eval_dataset_100.json` 包含：

- 10 条已缓存的标准问题和答案。
- 50 条应命中的语义改写问题。
- 30 条不应命中的相似负样本。
- 20 条必须绕过缓存的动态或个性化问题。

脚本会先将 10 条标准问题写入独立的 Chroma 测试集合，再评估 100 条查询。
测试集合与正式语义缓存集合隔离，默认在运行结束后自动删除。

## 三、核心指标

- `Precision`：所有缓存命中中，答案来源正确的比例。该指标越高，误答风险越低。
- `Recall`：50 条语义改写问题中，被正确命中的比例。
- `F1`：Precision 和 Recall 的综合指标。
- `负样本错误命中率`：30 条不应命中的问题中，被错误命中的比例。
- `安全绕过率`：20 条动态或个性化问题中，成功绕过语义缓存的比例。
- `正确调用避免率`：100 条混合请求中，正确复用缓存并避免完整 RAG 的比例。
- `P50/P95 延迟`：语义缓存查询延迟的中位数和高分位值。

## 四、运行前检查

1. 激活项目使用的 Python 虚拟环境。
2. 确认项目 Embedding 模型配置可用。
3. 确认已安装 `langchain-chroma` 和项目现有依赖。
4. 不需要清空正式缓存，测试使用独立集合。

## 五、运行命令

仅测试语义缓存准确性和查询延迟：

```powershell
python .\eval\semantic_cache_benchmark.py
```

同时实际测量 3 条完整 RAG 请求的平均延迟：

```powershell
python .\eval\semantic_cache_benchmark.py --measure-rag-baseline --baseline-samples 3
```

如果已经知道完整 RAG 的平均延迟，例如 4200 毫秒：

```powershell
python .\eval\semantic_cache_benchmark.py --rag-baseline-ms 4200
```

自定义候选阈值：

```powershell
python .\eval\semantic_cache_benchmark.py --thresholds 0.82 0.85 0.88 0.90 0.92
```

## 六、输出文件

报告默认生成到：

```text
eval/reports/semantic_cache_benchmark/
```

主要文件：

```text
semantic_cache_comparison.md
semantic_cache_report.json
```

优先查看 `semantic_cache_comparison.md`。JSON 报告包含每条查询的最近缓存问题、
相似度、是否命中、是否正确和最终分类，适合排查误命中样本。

## 七、选择阈值

脚本优先从满足以下条件的阈值中选择 F1 最高者：

- Precision 不低于 95%。
- 负样本错误命中率不高于 5%。

如果没有阈值满足条件，则从全部候选阈值中选择 F1 最高者。

得到推荐值后，修改 `config/semantic_cache.yml`：

```yaml
similarity_threshold: 0.90
```

这里的 `0.90` 必须替换为报告实际推荐的阈值。

## 八、结果验收标准

建议至少满足：

- Precision >= 95%。
- 负样本错误命中率 <= 5%。
- 安全绕过率 = 100%。
- Recall 在保证低误命中的前提下尽可能高。
- P95 查询延迟稳定，没有明显长尾异常。

如果 Precision 不达标，应提高阈值并增加容易混淆的负样本。
如果 Precision 很高但 Recall 偏低，可小幅降低阈值并重新运行。

## 九、简历描述模板

运行完成后，用真实结果替换数字：

> 构建包含 100 条查询的语义缓存离线评估集，覆盖 50 条语义改写、
> 30 条相似负样本及 20 条动态/个性化请求；通过多阈值对比将缓存命中
> Precision 提升至 XX%，误命中率控制在 XX%，安全绕过率达到 XX%，
> 命中缓存后端到端延迟较完整 RAG 降低 XX%。

面试时需要说明：该结果来自固定离线评估集，不等同于线上真实流量收益。
上线后还应继续监控命中率、误命中率、节省的 LLM 调用次数和延迟分位值。
