import os
from abc import ABC, abstractmethod
from functools import cache
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """通过 provider 的 OpenAI 兼容接口创建聊天模型。"""

    def generate(self) -> BaseChatModel:
        provider = rag_conf.get("chat_model_provider", "qwen")
        key_names = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}
        if provider not in key_names:
            raise ValueError(f"不支持的聊天模型 provider：{provider}")

        key_name = key_names[provider]
        api_key = os.getenv(key_name)
        if not api_key:
            raise RuntimeError(f"请设置 {key_name} 环境变量。")

        # 使用非思考模式，兼容通过 ChatOpenAI 往返传递工具调用。
        options = (
            {"extra_body": {"thinking": {"type": "disabled"}}}
            if provider == "deepseek"
            else {}
        )

        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=api_key,
            base_url=rag_conf["chat_model_base_url"],
            temperature=0,
            timeout=60,
            max_retries=2,
            **options,
        )


class EmbeddingsFactory(BaseModelFactory):
    """
    创建向量模型。

    保留原来的 DashScopeEmbeddings，
    与现有 Chroma 向量库保持一致。
    """

    def generate(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"]
        )


@cache
def get_chat_model() -> BaseChatModel:
    return ChatModelFactory().generate()


@cache
def get_embed_model() -> Embeddings:
    return EmbeddingsFactory().generate()
