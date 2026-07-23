# RAG Agent 智能客服

一个基于 Streamlit、LangChain/LangGraph、Chroma 和 Redis 的智能客服 Agent Demo。项目当前知识库聚焦新能源充电桩场景，支持知识库检索、会话记忆、流式回复、语义缓存和工具调用。

## 功能特点

- Streamlit 聊天界面，支持流式输出和 URL session_id 会话隔离
- 基于 LangGraph checkpointer 的多轮对话恢复
- RAG 知识库问答，当前资料为新能源充电桩 FAQ、故障排查、维护保养、运营运维和选型指南
- 支持向量检索、BM25 检索、Hybrid 检索和 Cross-Encoder 重排序
- Redis 精确缓存 + Chroma 语义缓存，减少重复问题的生成成本
- Agent 工具调用能力，包括 RAG 查询、记忆读写、数据库/MCP 查询等
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

建议使用 Python 虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

配置阿里云百炼 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
```

如果需要 Redis、MySQL、Celery，请根据 example 文件创建本地配置：

```text
config/cache.example.yml  -> config/cache.yml
config/db.example.yml     -> config/db.yml
config/celery.example.yml -> config/celery.yml
```

真实的 `cache.yml`、`db.yml`、`celery.yml` 已加入 `.gitignore`，不要提交到 GitHub。

## 知识库入库

当前知识库原始文件位于 `data/`，主题是新能源充电桩。

```powershell
python .\rag\vector_store.py
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
