import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from uuid import uuid4

from chromadb.config import Settings
from httpx import Client, MockTransport, Response
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import rag_summerize
from mcp_servers import rag_mcp_server
from model.factory import ChatModelFactory
from rag.evidence import RAG_CACHE_RESULT_FIELD, RAG_CACHE_VERSION, RAG_CACHE_VERSION_FIELD
from rag.rag_service import RagSummerizeService
from utils.semantic_cache_handler import SemanticCache


def assessment(step, *, ids=(), gaps=(), followup="", question=""):
    return json.dumps({
        "next_step": step,
        "selected_ids": list(ids),
        "gaps": list(gaps),
        "followup_query": followup,
        "clarification_question": question,
    }, ensure_ascii=False)


def document(text, **metadata):
    return Document(page_content=text, metadata=metadata)


def tool_call(call_id, query):
    return {"name": "rag_summerize", "id": call_id, "args": {"query": query}}


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgenticRagTest(unittest.TestCase):
    def setUp(self):
        self.service = RagSummerizeService()
        self.service.cache = Mock(enable=True, lock_enable=False)
        self.service.cache.get.return_value = None
        self.service.semantic_cache = Mock()
        self.service.semantic_cache_decision = Mock()
        self.service.semantic_cache_decision.lookup.return_value = None
        self.service.retriever_docs = Mock(return_value=[])
        self.assessments = Mock()
        self.answers = Mock()
        self.assessment_inputs = []
        self.answer_prompts = []
        model = patch("rag.rag_service.get_chat_model", return_value=RunnableLambda(self.respond))
        model.start()
        self.addCleanup(model.stop)

    def respond(self, prompt, **kwargs):
        if isinstance(prompt, list):
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})
            self.assertIn("JSON", prompt[0].content)
            self.assessment_inputs.append(json.loads(prompt[-1].content))
            return AIMessage(content=self.assessments())

        self.answer_prompts.append(prompt.to_string())
        return AIMessage(content=self.answers())

    def test_single_retrieval_uses_only_selected_evidence_and_keeps_source(self):
        self.service.retriever_docs.return_value = [
            document("确认电子锁闭合。", source=r"C:\kb\使用手册.pdf", page=2),
            document("本周站点营收。", source="报表.txt"),
        ]
        self.assessments.return_value = assessment("answer", ids=["E1"])
        self.answers.return_value = "启动充电前确认电子锁闭合【E1】。"

        result = self.service.answer_question("启动充电前要确认什么？")

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.sources, [{"id": "E1", "source": "使用手册.pdf，第 3 页"}])
        self.assertIn("【E1】使用手册.pdf", result.format())
        self.assertNotIn("营收", self.answer_prompts[0])
        self.service.retriever_docs.assert_called_once_with("启动充电前要确认什么？")

    def test_rewrite_merges_evidence_and_evaluates_the_original_question(self):
        question = "A 型充电桩 E12 的含义和处理办法是什么？"
        followup = "A 型 E12 处置办法"
        self.service.retriever_docs.side_effect = [
            [document("A 型 E12 表示通信异常。", filename="故障码.txt")],
            [
                document(" A 型  E12 表示通信异常。 ", filename="故障码.txt", rerank_rank=2),
                document("A 型 E12 应检查通信指示灯。", filename="运维指南.txt"),
            ],
        ]
        self.assessments.side_effect = [
            assessment("retrieve", ids=["E1"], gaps=["缺少处置步骤"], followup=followup),
            assessment("answer", ids=["E1", "E2"]),
        ]
        self.answers.return_value = "E12 表示通信异常【E1】，请检查通信指示灯【E2】。"

        answer = self.service.rag_summerize_without_cache(question)

        self.assertEqual(self.service.retriever_docs.call_args_list, [call(question), call(followup)])
        second = self.assessment_inputs[1]
        self.assertEqual(second["question"], question)
        self.assertEqual(second["search_queries"], [question, followup])
        self.assertEqual(second["retrievals_left"], 0)
        self.assertEqual([chunk["id"] for chunk in second["evidence"]], ["E1", "E2"])
        self.assertIn("【E2】运维指南.txt", answer)
        self.service.cache.get.assert_not_called()

    def test_missing_user_information_returns_question_without_caching(self):
        self.assessments.return_value = assessment(
            "clarify", gaps=["型号"], question="请提供充电桩型号。",
        )
        result = self.service.answer_question("E12 怎么处理？")

        self.assertEqual((result.status, result.text), ("clarify", "请提供充电桩型号。"))
        self.service.retriever_docs.assert_called_once()
        self.service.cache.set.assert_not_called()
        self.service.semantic_cache.add_cache.assert_not_called()
        self.answers.assert_not_called()

    def test_exhausted_budget_rejects_a_third_search(self):
        self.assessments.side_effect = [
            assessment("retrieve", gaps=["缺少型号说明"], followup="A 型说明书"),
            assessment("retrieve", gaps=["缺少故障处理步骤"], followup="A 型故障手册"),
        ]
        result = self.service.answer_question("A 型告警怎么修？")

        self.assertEqual(result.status, "insufficient")
        self.assertIn("缺少故障处理步骤", result.text)
        self.assertIn("检索次数已用完", result.text)
        self.assertEqual(self.service.retriever_docs.call_count, 2)
        self.answers.assert_not_called()
        self.service.cache.set.assert_not_called()

    def test_parallel_and_later_agent_tools_share_budget_reset_on_next_turn(self):
        self.assessments.return_value = assessment("stop", gaps=["知识库无型号手册"])
        parallel_calls = [tool_call(f"parallel-{i}", "A 型说明") for i in range(3)]
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=parallel_calls),
            AIMessage(content="", tool_calls=[tool_call("later", "B 型说明")]),
            AIMessage(content="本轮结束"),
            AIMessage(content="", tool_calls=[tool_call("next-turn", "C 型说明")]),
            AIMessage(content="下一轮结束"),
        ])
        checkpointer = InMemorySaver()
        with (
            patch("agent.react_agent.get_chat_model", return_value=model),
            patch("agent.react_agent.get_chat_checkpointer", return_value=checkpointer),
            patch("agent.tools.agent_tools.rag", self.service),
        ):
            agent = ReactAgent("offline-test")
            self.assertEqual(list(agent.execute_stream("第一轮")), ["本轮结束"])
            self.assertEqual(self.service.retriever_docs.call_args_list, [call("A 型说明")] * 2)
            saved = checkpointer.get(agent.get_checkpoint_config())["channel_values"]
            self.assertNotIn("rag_budget", saved)
            self.assertEqual(
                sum("检索次数已用完" in message.content for message in saved["messages"]),
                2,
            )
            self.assertEqual(list(agent.execute_stream("第二轮")), ["下一轮结束"])
            self.service.retriever_docs.assert_called_with("C 型说明")
            self.assertEqual(self.service.retriever_docs.call_count, 3)
            self.assertEqual(
                set(rag_summerize.tool_call_schema.model_json_schema()["properties"]), {"query"},
            )

    def test_legacy_cache_is_ignored_and_exact_hit_keeps_validated_citations(self):
        self.service.cache.get.return_value = "没有出处的旧缓存回答"
        self.service.semantic_cache_decision.lookup.return_value = SimpleNamespace(
            cached_query="启动要求", answer="没有出处的旧语义缓存回答", metadata={},
        )
        self.service.retriever_docs.return_value = [document("先插枪。", filename="手册.txt")]
        self.assessments.return_value = assessment("answer", ids=["E1"])
        self.answers.return_value = "先插枪【E1】。"

        original = self.service.answer_question("启动要求")
        raw = self.service.cache.set.call_args.args[1]
        self.assertEqual(json.loads(raw)["version"], RAG_CACHE_VERSION)
        self.service.cache.get.return_value = raw
        cached = self.service.answer_question("启动要求")

        self.assertEqual(cached, original)
        self.assertIn("【E1】手册.txt", cached.format())
        self.service.retriever_docs.assert_called_once()
        self.assessments.assert_called_once()

    def test_semantic_hit_reassesses_current_question_and_preserves_source(self):
        self.service.retriever_docs.return_value = [document("先插枪。", filename="手册.txt")]
        self.assessments.return_value = assessment("answer", ids=["E1"])
        self.answers.side_effect = ["先插枪【E1】。", "启动前先插枪【E1】。"]
        self.service.answer_question("启动要求")
        metadata = self.service.semantic_cache.add_cache.call_args.kwargs["metadata"]
        self.assertIn(RAG_CACHE_RESULT_FIELD, metadata)
        self.service.semantic_cache_decision.lookup.return_value = SimpleNamespace(
            cached_query="启动要求", metadata=metadata,
        )

        result = self.service.answer_question("充电开始前需要做什么？")

        self.assertEqual(result.text, "启动前先插枪【E1】。")
        self.assertEqual(result.sources, [{"id": "E1", "source": "手册.txt"}])
        self.assertEqual(self.assessment_inputs[1]["question"], "充电开始前需要做什么？")
        self.assertEqual(self.assessment_inputs[1]["search_queries"], [])
        self.service.retriever_docs.assert_called_once()
        self.service.semantic_cache.add_cache.assert_called_once()

    def test_semantic_evidence_insufficient_for_new_question_triggers_fresh_search(self):
        self.service.retriever_docs.return_value = [document("先插枪。", filename="手册.txt")]
        self.assessments.side_effect = [
            assessment("answer", ids=["E1"]),
            assessment("retrieve", gaps=["缺少巡检周期"], followup="巡检周期"),
            assessment("stop", gaps=["无巡检周期"]),
        ]
        self.answers.return_value = "先插枪【E1】。"
        self.service.answer_question("启动要求")
        metadata = self.service.semantic_cache.add_cache.call_args.kwargs["metadata"]
        self.service.semantic_cache_decision.lookup.return_value = SimpleNamespace(
            cached_query="启动要求", metadata=metadata,
        )

        result = self.service.answer_question("设备巡检周期是多久？")

        self.assertEqual(result.status, "insufficient")
        self.service.retriever_docs.assert_called_with("设备巡检周期是多久？")
        self.answers.assert_called_once()
        self.service.cache.set.assert_called_once()

    def test_semantic_search_filters_legacy_rows_in_the_same_chroma_collection(self):
        store = Chroma(
            collection_name=f"rag_test_{uuid4().hex}",
            embedding_function=DeterministicFakeEmbedding(size=16),
            client_settings=Settings(anonymized_telemetry=False),
        )
        with (
            patch("utils.semantic_cache_handler.Chroma", return_value=store),
            patch("utils.semantic_cache_handler.get_embed_model"),
        ):
            semantic_cache = SemanticCache(
                metadata_filter={RAG_CACHE_VERSION_FIELD: RAG_CACHE_VERSION},
            )
        self.assertTrue(semantic_cache.add_cache("公共桩启动要求", "旧缓存"))
        self.assertTrue(semantic_cache.add_cache(
            "公共桩启动要求", "新格式缓存",
            metadata={RAG_CACHE_VERSION_FIELD: RAG_CACHE_VERSION},
        ))
        candidates = semantic_cache.retrieve_candidates("公共桩启动要求")
        self.assertEqual([candidate.answer for candidate in candidates], ["新格式缓存"])

    def test_invalid_assessment_does_not_generate_or_cache(self):
        self.assessments.side_effect = ["{unparseable", assessment("answer", ids=["E99"])]
        for _ in range(2):
            self.assertEqual(self.service.answer_question("该怎么处理？").status, "insufficient")
        self.answers.assert_not_called()
        self.service.cache.set.assert_not_called()

    def test_invalid_answer_citations_fail_closed(self):
        self.service.retriever_docs.return_value = [document("先插枪。", filename="手册.txt")]
        self.assessments.return_value = assessment("answer", ids=["E1"])
        self.answers.return_value = "先插枪【E1】，然后做其他操作【E9】。"

        self.assertEqual(self.service.answer_question("启动要求").status, "insufficient")
        self.service.cache.set.assert_not_called()

    def test_mcp_answer_has_status_and_raw_retrieval_still_returns_metadata(self):
        self.assessments.return_value = assessment("clarify", question="请提供型号。")
        self.service.retriever_docs.return_value = [
            document("片段内容", filename="手册.txt", rerank_rank=1),
        ]
        with patch.object(rag_mcp_server, "rag_service", self.service):
            answer = rag_mcp_server.rag_summerize("故障码含义")
            raw = rag_mcp_server.retrieve_docs("故障码含义")
        self.assertEqual(answer["status"], "clarify")
        self.assertEqual(answer["data"]["answer"], "请提供型号。")
        self.assertTrue(raw["found"])
        self.assertEqual(raw["data"][0]["metadata"]["rerank_rank"], 1)

    def test_chat_provider_requires_named_key_and_configures_deepseek_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                ChatModelFactory().generate()

        def respond(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            return Response(200, json={
                "id": "offline-chat",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                    "finish_reason": "stop",
                }],
            })

        with (
            Client(transport=MockTransport(respond)) as transport,
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-placeholder"}, clear=True),
            patch(
                "model.factory.ChatOpenAI",
                side_effect=lambda **options: ChatOpenAI(http_client=transport, **options),
            ),
        ):
            model = ChatModelFactory().generate()
            result = model.bind(response_format={"type": "json_object"}).invoke("Return JSON.")
        self.assertEqual(json.loads(result.content), {"ok": True})


if __name__ == "__main__":
    unittest.main()
