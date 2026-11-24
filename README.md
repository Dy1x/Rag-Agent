# RAG Agent 智能客服

一个基于 Streamlit、LangChain/LangGraph、Chroma 和 Redis 的智能客服 Agent Demo。项目当前知识库聚焦新能源充电桩场景，支持知识库检索、会话记忆、流式回复、语义缓存和工具调用。

## 架构概览

Streamlit 聊天入口将车主或运维提问路由到三条业务链路。**Agentic RAG** 检索手册和 FAQ、评估证据够不够回答，并返回段落引用或追问。**站点掉线调查** 由固件与站点专家分别分析，再由协调节点汇总。若 MySQL 证据确认网关重启中断了充电会话，调查会关联费用子案件。**钱包费用 Maker-Checker** 读取计费账本与政策，制单后经过硬校验和独立复核，涉及资金的方案由人工批准后执行。MySQL 持久化案件和轨迹，运营可在 Streamlit 查看，也可导出 OTLP 供 OpenTelemetry 工具读取。

## 功能特点

- Streamlit 聊天界面，支持流式输出和 URL session_id 会话隔离
- 基于 LangGraph checkpointer 的多轮对话恢复
- Agentic RAG 知识库问答，评估并引用新能源充电桩 FAQ、故障排查、维护保养、运营运维和选型资料
- 支持向量检索、BM25 检索、Hybrid 检索和 Cross-Encoder 重排序
- Redis 精确缓存复用带出处的答案，Chroma 语义缓存复用片段并对当前问题重新评估
- 主聊天支持充电桩知识问答、费用案件、站点掉线调查和会话记忆
- 本地配置与敏感信息通过 example 文件展示，真实配置不提交

## 项目结构

```text
.
├── app.py                 # Streamlit 应用入口
├── agent/                 # Agent 定义、工具和中间件
├── rag/                   # RAG 检索、重排序、向量库和语义缓存逻辑
├── model/                 # 模型工厂，配置聊天模型和 embedding 模型
├── utils/                 # 配置、缓存、记忆、日志、文件处理等工具
├── service/               # 知识文件与异步任务相关服务
├── tasks/                 # Celery 任务
├── mcp_servers/           # MCP 服务端/客户端示例
├── config/                # 项目配置与 example 配置
├── data/                  # 当前知识库原始资料
├── eval/                  # 检索与语义缓存评估脚本
└── prompts/               # 系统提示词与 RAG 提示词
```

## 环境准备

建议使用 Python 3.11 或更新版本创建虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

主 Agent、证据评估和答案生成默认使用 DeepSeek。知识库和语义缓存继续使用阿里云百炼的 `text-embedding-v4`，已有向量库可以直接复用。在运行应用的 PowerShell 会话中配置两个环境变量。

```powershell
$env:DEEPSEEK_API_KEY="<DeepSeek API Key>"
$env:DASHSCOPE_API_KEY="<DashScope API Key>"
```

从 example 文件创建缺失的本地配置，保留已有配置。启动页面前，在 `config/cache.yml` 中配置可访问的 Redis Stack（需 RedisJSON 和 RediSearch）及其密码。MySQL 和 Celery 配置供对应功能使用。

```text
config/cache.example.yml  -> config/cache.yml
config/db.example.yml     -> config/db.yml
config/celery.example.yml -> config/celery.yml
```

真实的 `cache.yml`、`db.yml`、`celery.yml` 已加入 `.gitignore`，不要提交到 GitHub。

## 知识库入库

当前知识库原始文件位于 `data/`，主题是新能源充电桩。

```powershell
python -m rag.vector_store
```

向量库会生成到 `config/chroma.yml` 中的 `persist_directory`：

```yaml
persist_directory: chroma_db_charging_pile
```

本地 Chroma 向量库不提交到 GitHub，需要在新环境中重新入库。

## 启动项目

```powershell
streamlit run app.py
```

默认访问：

```text
http://localhost:8501
```

也可用 `python -m mcp_servers.rag_mcp_server` 从标准输入输出运行知识库 MCP 服务。服务的 `retrieve_docs` 返回原始检索和重排结果。问答工具另返回 `status`（`answered`、`clarify` 或 `insufficient`）及引用出处。

## 验证 Agentic RAG

安装依赖并建好本地配置文件后，可在没有 API Key、Redis 服务和向量索引的环境运行模拟测试。

```powershell
python -m unittest discover -s tests -v
```

每个用户请求最多执行两次知识库检索，Agent 在同一轮发出的多个知识库工具调用共享这个额度。评估认为缺少型号、告警码等必要信息时会追问。证据充分时返回行内片段引用及出处，补查仍无充分证据时会说明缺口。旧版纯文本缓存会被跳过。

配置好密钥后，可先验证 DeepSeek 通路，再启动页面提问。检查回答的片段引用和出处，并在应用日志中查看每次实际知识库检索对应的 `[Agentic RAG] 开始知识检索` 记录。

```powershell
python -c "from model.factory import get_chat_model; print(get_chat_model().invoke('请只回复 OK').content)"
python -m streamlit run app.py
```

## 推荐演示问题

可以在页面中输入：

```text
用户反馈：我在公共充电站扫码后无法启动充电，设备好像显示离线，车已经插枪了。

请你作为新能源充电桩客服 Agent 完成：
1. 提取用户问题
2. 判断可能原因
3. 给出安全、稳妥的排查步骤
4. 列出需要用户补充的信息
5. 生成一段客服回复话术
```

也可以测试：

```text
第一次使用公共充电桩需要注意什么？
```

```text
充电时闻到焦糊味怎么办？请用客服口吻回答，并给出安全提醒。
```

## 说明

这是一个用于学习和展示的 Agent + RAG 项目，重点展示从知识库检索到客服回复生成的完整链路。项目中的向量库、缓存、日志、本地密码配置和临时测试目录均不提交，便于保持公开仓库干净。

## 充电站运维异常调查 · 第一天离线基线

在项目根目录运行以下命令。该入口使用 Python 标准库和临时 SQLite 数据库，无需 API Key、向量库或其他服务。在本机可以使用 `E:\anaconda\python.exe` 替换命令中的 `python`。

```powershell
python -m eval.investigation.run_baseline --case firmware
python -m eval.investigation.run_baseline --case site
python -m eval.investigation.run_baseline --case missing
python -m unittest discover -s tests -p test_investigation_data.py -v
```

| 样例 | 基线预期 |
| --- | --- |
| `firmware` | 共 7/16 台掉线。A 站 4/8、B 站 3/8。目标版本组 6/8、对照组 1/8。8 条掉线事件按设备去重为 7 台。 |
| `site` | A 站 6/8、B 站 0/8。目标版本组和对照组均为 3/8。 |
| `missing` | A 站 3/8、B 站 2/8。固件来源标记为 `unavailable`，统计保留设备及站点结果，并列出固件证据缺口。 |

四个查询入口共用 `InvestigationScope`，包含站点 ID、UTC 开始与结束时间、固件来源状态和可选的缺失说明。告警窗口包含起点、排除终点，仅 `offline` 事件计入掉线设备数。固件查询覆盖开始前两小时，包含窗口起点、排除调查开始时刻。每台设备按窗口内最后一次成功升级分组，同一时刻的记录按升级 ID 排序。对照组包含其余范围内的台账设备。掉线率以台账设备数为分母，分母为零时返回 `null`。

结果包含设备、告警和固件记录的来源 ID。固件来源可用但没有升级记录时，返回 `available` 和空记录，仍能计算对照组。固件来源不可用时，升级记录和分组结果为 `null`，`missing_evidence` 记录原因。基线输出的站点分布和组间差异用于比较观测关联，原因仍需额外证据核实。

### 聊天中的多 Agent 调查

聊天工具 `investigate_outage` 接收站点 ID 列表、明确的 UTC 起止时间和具体问题。服务同步完成固件调查、站点调查及必要复核，返回报告和本次执行的 `run_id`。多 Agent 的 `run_id` 用于关联本次执行。原单 Agent 的 SQLite 恢复仍由 `run_agent.py` 提供。

如需复现聊天路径，先在启动页面的 PowerShell 会话中生成模拟库。准备命令要求目标文件尚不存在。

```powershell
$env:INVESTIGATION_DB_PATH = Join-Path ([IO.Path]::GetTempPath()) ("outage-" + [guid]::NewGuid().ToString("N") + ".sqlite3")
python -m eval.investigation.run_multi_agent --case rollout_site_overlap --prepare-db $env:INVESTIGATION_DB_PATH
$env:INVESTIGATION_FIRMWARE_SOURCE = "available"
$env:INVESTIGATION_SIMULATED_DATA = "1"
$env:INVESTIGATION_PLANNER = "offline"
python -m streamlit run app.py
```

这些变量由应用服务端设置。`INVESTIGATION_DB_PATH` 指向已有 SQLite 文件。`INVESTIGATION_FIRMWARE_SOURCE` 可以是 `available` 或 `unavailable`，缺失原因可放在 `INVESTIGATION_FIRMWARE_SOURCE_REASON`。`INVESTIGATION_SIMULATED_DATA=1` 会在报告中标记模拟数据。`INVESTIGATION_PLANNER` 默认使用 `deepseek`。`offline` 只替换调查子图的规划器与知识库，聊天主 Agent 仍使用项目配置的模型和会话存储。无需模型密钥或 Redis 的验收可直接使用下面的 CLI 和测试。

可在聊天输入「请调查充电站 A 和 B 在 UTC 2026-11-21T14:00Z 至 2026-11-22T00:00Z 的掉线影响，说明优先排查方向，并保留关键证据 ID 和 run_id。」这份模拟库应得到 15/30 台掉线，A 站网关事件 `gw-A-01` 比同站首条掉线早两分钟。按站点复核后，A 站两组都是 80%，B 站两组都是 20%，因此优先核验 A 站网关，故障归因仍待验证。只说「昨晚有站点掉线」会触发范围确认，通用手册问题走现有 Agentic RAG。

### 单 Agent 和多 Agent 对照评测

```powershell
python -m eval.investigation.run_multi_agent --case rollout_site_overlap --planner offline
python -m eval.investigation.compare_investigation_agents --planner offline
python -m eval.investigation.compare_investigation_agents --planner deepseek
python -m unittest discover -s tests -v
```

两种架构调查同一问题和同一份模拟 SQLite 数据。每次均含两次初查，最多五次后续工具调用，其中知识库最多一次。单 Agent 的评测配置开放站点事件和分站固件对比。多 Agent 的后续预算分配给两位专家各两次，协调后最多一次定向复核。输出包含逐案分数、失败原因、复核轨迹、调用次数和耗时。调用总数计入逻辑工具调用，知识库工具内部的检索仍采用现有预算。耗时为当次运行观测值。

`offline` 使用规则与模拟知识库。`deepseek` 使用已配置的模型和现有知识库，需相应模型密钥和可用依赖。两种模式的设备、升级和告警数据均为模拟数据，输出会注明规划器和知识库模式。

## 充电站运维异常调查 · 第二天调查图

在项目根目录用安装了 LangGraph 和 Pydantic 的 Python 运行。本机可以使用已有的 `.\.venv\agentic-rag\Scripts\python.exe`。三个离线案例使用同一个默认问题「昨晚多个站点掉线，优先查哪里？」。也可以用 `--question` 传入问题，站点和 UTC 窗口来自所选样例。

```powershell
python -m eval.investigation.run_agent --case firmware --planner offline
python -m eval.investigation.run_agent --case site --planner offline
python -m eval.investigation.run_agent --case missing --planner offline
python -m unittest discover -s tests -p test_investigation_graph.py -v
```

调查图先并行获取告警与固件记录，合流后最多执行两次补查，最多重新规划一次。输出包含 `run_id`、实际动作及理由和调查报告。离线规划仅使用已取得的证据，知识库响应带 `simulated: true` 标记，无需 API Key、向量库或 Redis。

配置现有 DeepSeek、知识库及相关服务后，可将参数改为 `--planner deepseek`。规划使用 DeepSeek 的 JSON 输出，知识库动作调用现有 Agentic RAG。规划异常时会在条件允许的情况下执行站点对比。缺失或失败来源会保留在报告里，并保留其他来源的可用证据。

## 充电站运维异常调查 · 第三天恢复与评测

安装 `requirements.txt` 后在项目根目录运行。每次 `start` 先在标准错误流显示 `run_id`，并将该次调查的台账、告警、升级记录、运行参数和 LangGraph 检查点保存到同一个 SQLite 文件。默认存储目录为 `eval/reports/investigation-runs`。可用 `--runs-dir` 指定目录，同一次调查的各条命令使用同一目录。

```powershell
$saved = python -m eval.investigation.run_agent start --case firmware --planner offline --pause-after compare_firmware_cohorts | ConvertFrom-Json
python -m eval.investigation.run_agent status --run-id $saved.run_id
python -m eval.investigation.run_agent resume --run-id $saved.run_id
```

`--pause-after` 在指定动作落盘后结束本次进程。省略后调查会继续运行至收口。`status` 查看已保存的进度。`resume` 从检查点继续，已完成的调查会直接返回原报告。第二天的 `--case ... --planner ...` 用法继续使用临时数据库。

六案例评测使用独立的 `expected_results.json`。每项指标以通过的案例数为分子、参加评测的案例数为分母。评测逐例给出实际值和失败原因，任一指标未满时退出码为 1。离线模式使用模拟知识库，DeepSeek 模式与现有模型和 RAG 配置联动。

```powershell
python -m eval.investigation.run_evaluation --planner offline
python -m eval.investigation.run_evaluation --planner deepseek
python -m unittest discover -s tests -p 'test_investigation_*.py' -v
```

## 充电费用争议处理

模拟运营台用钱包流水核查重复扣款、冻结、充值、时段计费和占位费争议。金额单位为分，电量单位为 Wh。计费使用启动时所在的本地时段，数据库保存 UTC 时间。制单与复核分别运行，退款和解冻经人工审批后由代码执行。

### 初始化与启动

在项目根目录、已安装依赖的 Python 3.11+ 环境运行。计费服务读取 `config/db.yml` 的 MySQL 连接参数，默认创建独立的 `charging_billing` 库。初始化账号需拥有该库的建库、建表和读写权限。重复运行初始化会保留已有业务记录。

```powershell
python -m scripts.init_billing_db
python -m streamlit run app.py
```

需要自定义配置时，将 `config/billing.example.yml` 复制为 `config/billing.yml`，设置库名、本地时区、planner、修订次数和 SQLite 检查点路径。默认配置使用 `offline` 和 `Asia/Shanghai`。命令行可用 `--database NAME` 指定库。显式传入 `--reset` 会清空该库的计费业务和案件记录，并重新写入种子数据。

### 从模拟扣款到审批

1. 在「车主模拟器」选择 S2，点击「生成场景」。这会新建车主、充值、生成一笔 560 分的充电订单，并模拟扣款两次。也可以分别操作充值、启动、结束、读数补传与人工退款，查看钱包投影和流水。
2. 在「一键生成场景」中选择本次规划模式（默认取计费配置，可选 `offline` 或 `deepseek`），点击「直接发起费用核查」。模拟器用场景车主、建议问题和定位线索直接调用计费处理器，展示处理状态、车主答复和可复制的案件号、运行号。也可以点击「切换到该车主去聊天」，用建议问题和订单号发起核查。首页切换车主会创建新的聊天会话。
3. 在「资金审批台」按案件号查看证据、各版方案及复核意见，可以通过、拒绝或调低退款金额。S2 通过后应向第二笔扣款退回 560 分。在「车主模拟器」核对退款流水，也可在聊天中查询本人案件状态。
4. 在「案件运行轨迹」输入案件号或运行号，查看工具调用、制单、复核、挂起、恢复与执行结果，并下载 JSON。重启后恢复待审批案件需要保留业务库和 `data/checkpoints/billing_cases.sqlite`。

主聊天沿用现有聊天模型和 Redis Stack。离线计费图、审批台、轨迹页和模拟器使用 MySQL，单独运行模拟器可执行 `python -m streamlit run "pages/1_车主模拟器.py"`。DeepSeek 计费模式使用项目配置的 DeepSeek 模型及 `DEEPSEEK_API_KEY`。

聊天入口默认提供四类工具：手册、选型与通用排障使用 `rag_summerize`；个人费用争议和案件进度使用 `handle_billing_dispute`、`get_dispute_case_status`，并可调用已注册的业务处理器；具体站点的批量掉线或升级异常使用 `investigate_outage`；会话记忆使用 `get_memory_context`、`save_memory_context` 和 `clear_memory_context`。

DeepSeek 复核员默认可执行 8 次只读查询，提交结构化意见另有余量。计费 FAQ 查询保留引用与证据；知识库暂不可用时，返回证据不足提示，计费核查可继续使用账本与适用政策。

### 测试与评测

```powershell
$env:BILLING_TEST_MYSQL = "1"
python -m unittest discover -s tests -v
python -m eval.billing.run_evaluation --planner offline
python -m eval.billing.compare_designs --planner offline
python -m eval.billing.run_evaluation --planner deepseek
python -m eval.billing.compare_designs --planner deepseek --workers 3
```

MySQL 单测使用独立的 `cb_test_` 临时库，未启用环境开关时跳过。评测创建 `cb_eval_` 临时库，模拟人工批准并在结束时清理。场景评测支持 `--keep-db`，有失败时写出逐例结果并以退出码 1 结束。对照评测通过 `--workers` 设置每组并发，离线默认 1，DeepSeek 默认 3。真实评测使用 `config/rag.yml` 中的模型和当前会话的 `DEEPSEEK_API_KEY`。

### 2026-11-24 实测结果

开启 MySQL 8.0.43 集成测试后 **108/108** 通过。最新离线对照中，C 组全检查 **34/34**、硬性变异拦截 **14/14**、误拦截 **0/7**。[离线对照表](eval/reports/billing-compare-offline-332e2d74.md) · [离线逐样本数据](eval/reports/billing-compare-offline-332e2d74.json)

#### 真实 DeepSeek 复测与 A/B/C 消融

模型为 `deepseek-flash`，制单只读工具预算 3，复核只读工具预算 8，最多修订 2 次。11 个未变异场景复测 **11/11** 通过。与最近一次历史 11 场景报告相比，S4 和 S10 的首次审核均由拒绝变为通过。历史报告没有配置元数据，这些数字是两次实际运行的观测差异。[本次场景报告](eval/reports/billing-eval-deepseek-05f1a2d4.json) · [历史场景报告](eval/reports/billing-eval-deepseek-7b7095d6.json)

| 场景 | 修订次数（历史 → 本次） | Token（历史 → 本次） | Token 变化 |
| --- | ---: | ---: | ---: |
| S4 错误峰价 | 1 → 0 | 31,348 → 20,842 | −33.5% |
| S10 宽限期内误收占位费 | 1 → 0 | 31,983 → 21,049 | −34.2% |

真实对照每组使用相同矩阵，包含 11 个原始场景和 23 个首版变异，合计 102 个样本。A 组共享制单对话，自查时能看到变异前的制单输出。B 组只做确定性硬校验。C 组在 B 的基础上加入独立软审核和车主说明正则兜底。误拦截分母为各组未变异、进入审核且首版符合评分断言的样本。

| 组别 | 最终结论正确 | 全部检查通过 | 首版变异拦截 | 硬性拦截 | 误拦截 | 平均 Token | 平均秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 单 Agent 自查 | 33/34 | 32/34 | 23/23 | 14/14 | 0/5 | 35,521.00 | 15.15 |
| B 制单加硬校验 | 30/34 | 24/34 | 14/23 | 14/14 | 0/7 | 16,001.47 | 12.00 |
| C 硬校验加独立复核 | 33/34 | 33/34 | 22/23 | 14/14 | 0/6 | 30,748.09 | 19.68 |

23 项首版变异在三组均成功注入，三组各记录 0 次越权工具调用。平均 Token 以运行轨迹中记录的模型用量计算，平均秒数按样本计算。[真实对照表与逐变异统计](eval/reports/billing-compare-deepseek-8513d5f7.md) · [真实逐样本数据](eval/reports/billing-compare-deepseek-8513d5f7.json)

- **A** 本轮拦截全部 23 项变异，包括 5 项过度承诺。原始 S2 的退款执行正确，但首版车主说明带有完整订单号，LLM 自查予以放行。原始 S8 收到一次工具消息协议 API 400，转人工并计入失败。
- **B** 拦截全部 14 项硬性变异，放行 4 项遗漏补救和 5 项过度承诺。原始 S9 的单价用「每分钟 0.5 元」表述，现有断言因未匹配 `50` 将其计为失败，表中保留原始分数。
- **C** 拦截 22/23 项变异，全部检查通过 33/34。漏检的 S1 变异移除了 5,000 分解冻动作，但说明仍称已提交解冻，独立复核放行后实际未执行解冻。C 的平均 Token 较 B 高 92.2%，较 A 低 13.4%。

在完整对照之后，针对 C 组 S1 的两个边界问题做了两条真实 DeepSeek 定向复测。原始场景首版通过，历史「保证金」原句也通过修正后的规则检查。遗漏解冻的变异首版以 `S_MISSED_REMEDY` 驳回，修订一次后解冻 5,000 分成功。上表保留完整对照的原始结果。[S1 定向复测](eval/reports/billing-s1-targeted-deepseek-e2660e46.json)

## 掉线调查的 MySQL 数据源

聊天工具 `investigate_outage` 和注册的 `outage_investigation` 处理器可以读取 MySQL 调查快照。默认仍使用 `INVESTIGATION_DB_PATH` 指定的 SQLite 文件。两种方式共用调查图、固件分组、网关时间线与证据 ID。

在项目根目录运行以下命令，可把既有 SQLite 证据复制到与计费案件相同的 MySQL 库。导入以只读方式打开源文件，在一个事务内写入快照。相同数据集和内容可以重复导入。内容发生变化时使用新的数据集名称。旧库缺少站点事件表时，查询会保留“来源不可用”的语义。

```powershell
python -m scripts.migrate_investigation --from-sqlite .\investigation.sqlite3 --dataset site_snapshot_01
$env:INVESTIGATION_BACKEND = "mysql"
$env:INVESTIGATION_DATASET = "site_snapshot_01"
$env:INVESTIGATION_FIRMWARE_SOURCE = "available"
$env:INVESTIGATION_PLANNER = "offline"
python -m streamlit run app.py
```

若导入时用 `--database NAME` 指定其他库，运行调查时将 `INVESTIGATION_MYSQL_DATABASE` 设为同一个名称。切回旧数据源时设置 `INVESTIGATION_BACKEND=sqlite` 并指定原文件路径。使用模拟数据时设置 `INVESTIGATION_SIMULATED_DATA=1`。

MySQL 调查把案件、报告和工具轨迹写入共享的 `cb_dispute_cases`、`cb_case_runs` 和 `cb_trace_events`，按返回的案件号或运行号可在「案件运行轨迹」查看。掉线调查沿用原有运维复核流程。验收覆盖 7 组样例的双库逐项对照、微秒时间边界、大小写敏感站点、旧库缺表、重复导入与失败回滚。本机的演示数据集 `demo_rollout_site_overlap` 包含 30 台设备，离线调查核实其中 15 台发生掉线。

### 网关重启后的费用联动

调查快照可包含设备侧已确认的 `session_interruptions` 表。导入器识别以下列。源表的 `interruption_id` 为主键，其余列为非空文本。

| 列 | 内容 |
| --- | --- |
| `interruption_id`、`device_id` | 中断凭据 ID、调查设备 ID |
| `charger_id`、`session_id` | 设备记录关联的计费桩与充电会话 |
| `site_event_id`、`alarm_event_id` | 本数据集的网关重启事件和该设备的掉线告警 |
| `cause`、`occurred_at` | 已确认原因和带 UTC 偏移的发生时间。网关重启使用 `gateway_restart` |

MySQL 调查完成时会核对站点、设备、桩、事件先后、调查窗口及充电会话起止时间。匹配的每个会话最多创建一个费用子案件，车主取自服务端账本。多次调查可以关联同一个子案件。仅包含告警和网关时间线的快照可以用于运维判断；触发费用核查需要设备侧会话中断凭据。

聊天工具返回联动案件数；父运行轨迹的 `child_case` 事件记录中断凭据和子案件号，可以在轨迹页继续查看。订单缺失或结束读数待补时子案件进入待补证据；涉及资金的方案进入现有审批台。

### 导出 OpenTelemetry 轨迹

在「案件运行轨迹」输入案件号并选择运行，或直接输入运行号，点击「下载 OpenTelemetry 轨迹」。也可在项目根目录运行以下命令，输出路径需为新文件。

```powershell
python -m scripts.export_case_trace --run-id RUN-2063FFF854CC --output .\outage-demo.otlp.pb
```

页面和命令行均从同一 MySQL 事务读取运行状态和事件，导出 OTLP Protobuf 的 `ExportTraceServiceRequest`。每个案件对应一个 trace，每次运行对应一个 span。事件按数据库序号排列，保留 UTC 时间、节点和完整 JSON 载荷；父案件的 `child_case` 事件同时关联到子案件的 span。审批等待中的运行可导出截至最后一条事件的快照，审批后可再次导出同一运行。[本地掉线调查导出样例](eval/reports/RUN-2063FFF854CC.otlp.pb)含 13 条事件。
