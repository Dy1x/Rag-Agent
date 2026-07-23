###识别文件类型,加载 txt/pdf/docx,清洗文本,质量检查,返回 Document 列表

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from utils.config_handler import document_parser_conf
from utils.logger_handler import logger


#先按后缀判断
def detect_file_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower().lstrip(".")

    if not suffix:
        return "unknown"

    return suffix

#实现支持类型检查
def is_allowed_file_type(file_type: str) -> bool:
    allowed = document_parser_conf.get("allowed_file_types", [])
    return file_type in allowed

#实现文本清洗:去掉多余空格,去掉空行,去掉首尾空白
def clean_text(text: str) -> str:
    if not text:
        return ""

    if document_parser_conf.get("normalize_whitespace", True):
        text = text.replace("\u3000", " ")
        text = re.sub(r"[ \t]+", " ", text)

    if document_parser_conf.get("remove_empty_lines", True):
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        text = "\n".join(lines)

    return text.strip()

#实现 Document 清洗和过滤
def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned_docs = []
    min_len = document_parser_conf.get("min_chunk_text_length", 5)

    for doc in documents:
        text = doc.page_content

        if document_parser_conf.get("clean_text", True):
            text = clean_text(text)

        if len(text) < min_len:
            continue

        cleaned_docs.append(
            Document(
                page_content=text,
                metadata=doc.metadata,
            )
        )

    return cleaned_docs

#实现不同类型 loader
def load_txt(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


def load_pdf(file_path: str) -> list[Document]:
    return PyPDFLoader(file_path).load()


def load_docx(file_path: str) -> list[Document]:
    return Docx2txtLoader(file_path).load()



#实现统一入口
def load_documents_by_type(file_path: str) -> list[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"路径不是文件：{file_path}")

    file_type = detect_file_type(file_path)

    if not is_allowed_file_type(file_type):
        raise ValueError(f"不支持的文件类型：{file_type}")

    if file_type == "txt":
        documents = load_txt(file_path)
    elif file_type == "pdf":
        documents = load_pdf(file_path)
    elif file_type == "docx":
        documents = load_docx(file_path)
    else:
        raise ValueError(f"暂未实现该文件类型解析：{file_type}")

    documents = clean_documents(documents)
    validate_documents(documents, file_path)

    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    for index, doc in enumerate(documents):
        doc.metadata["source"] = file_path
        doc.metadata["filename"] = filename
        doc.metadata["file_type"] = file_type
        doc.metadata["file_size"] = file_size
        doc.metadata["parser"] = f"{file_type}_parser"
        doc.metadata["document_index"] = index

    logger.info(
        f"[文档解析]解析成功，file_path={file_path}, file_type={file_type}, doc_count={len(documents)}"
    )

    return documents


#实现质量检查:防止空文件、扫描 PDF、解析失败文件误入库
def validate_documents(documents: list[Document], file_path: str):
    if not documents:
        raise ValueError(f"文件未解析出有效文本：{file_path}")

    total_text = "\n".join(doc.page_content for doc in documents)
    min_total_len = document_parser_conf.get("min_total_text_length", 20)

    if len(total_text) < min_total_len:
        raise ValueError(
            f"文件解析文本过短，可能是空文件、图片扫描件或解析失败：{file_path}, text_length={len(total_text)}"
        )
