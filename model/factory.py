import os
from abc import ABC, abstractmethod
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
    """
    创建聊天模型。

    qwen3.7-plus 通过阿里云百炼的 OpenAI 兼容接口调用，
    不再使用 ChatTongyi。
    """

    def generate(self) -> BaseChatModel:
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "没有读取到 DASHSCOPE_API_KEY，"
                "请先配置阿里云百炼 API Key。"
            )

        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=api_key,
            base_url=rag_conf["chat_model_base_url"],
            temperature=0,
            timeout=60,
            max_retries=2,
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


chat_model = ChatModelFactory().generate()
embed_model = EmbeddingsFactory().generate()