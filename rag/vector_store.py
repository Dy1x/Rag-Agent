import os.path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf, retrieval_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from model.factory import get_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.document_parser import load_documents_by_type


class VectorStoreService:
    def __init__(self):
        self.vectors_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),

        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        vector_k = k or retrieval_conf.get("vector_k", chroma_conf["k"])

        return self.vectors_store.as_retriever(
            search_kwargs={"k": vector_k}
        )

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return:None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_sex_store"])):
                open(get_abs_path(chroma_conf["md5_sex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_sex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_sex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_file_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]内容已经存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vectors_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")

            except Exception as e:
                # exc_info为true的话，会记录详细的报错堆栈，为False的话，仅记录报错信息本身
                logger.error(f"[加载知识库]{path}内容加载失败：{str(e)}", exc_info=True)
                continue

    def load_single_document(self, file_path: str, file_id: int | None = None, version: int = 1) -> dict:

        documents = load_documents_by_type(file_path)

        if not documents:
            raise ValueError(f"文件没有有效文本内容：{file_path}")

        split_documents = self.spliter.split_documents(documents)

        if not split_documents:
            raise ValueError(f"文件分片后没有有效内容：{file_path}")

        chunk_count = len(split_documents)
        chunk_lengths = [len(doc.page_content) for doc in split_documents]

        for index, doc in enumerate(split_documents):
            doc.metadata["chunk_index"] = index
            doc.metadata["chunk_count"] = chunk_count

            if file_id is not None:
                doc.metadata["file_id"] = file_id
                doc.metadata["version"] = version
                doc.metadata["chunk_id"] = f"file_{file_id}_v{version}_chunk_{index}"

        self.vectors_store.add_documents(split_documents)

        report = {
            "file_path": file_path,
            "file_id": file_id,
            "version": version,
            "raw_doc_count": len(documents),
            "chunk_count": chunk_count,
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_chunk_length": round(sum(chunk_lengths) / chunk_count, 2),
        }

        logger.info(f"[加载知识库]{file_path}内容加载成功，report={report}")
        return report


    def get_chunks_by_file_id(self, file_id: int) -> list[dict]:
        result = self.vectors_store.get(
            where={"file_id": file_id}
        )

        chunks = []

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for index, chunk_id in enumerate(ids):
            chunks.append({
                "id": chunk_id,
                "content": documents[index],
                "metadata": metadatas[index],
            })
        return chunks

    def delete_chunks_by_file_id(self, file_id: int) -> int:
        result = self.vectors_store.get(
            where={"file_id": file_id}
        )

        ids = result.get("ids", [])

        if not ids:
            logger.info(f"[知识库更新]未找到需要删除的旧chunk，file_id={file_id}")
            return 0

        self.vectors_store.delete(ids=ids)

        logger.info(f"[知识库更新]删除旧chunk成功，file_id={file_id}, deleted_count={len(ids)}")
        return len(ids)

    def delete_chunks_by_ids(self, ids: list[str]) -> int:
        if not ids:
            logger.info("[知识库更新]没有需要删除的旧chunk")
            return 0

        self.vectors_store.delete(ids=ids)

        logger.info(f"[知识库更新]按ids删除旧chunk成功，deleted_count={len(ids)}")
        return len(ids)

    def get_all_chunks(self) -> list[Document]:
        result = self.vectors_store.get()

        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks = []

        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}

            if not content:
                continue

            chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata or {},
                )
            )

        return chunks

if __name__ == '__main__':
    service = VectorStoreService()
    service.load_document()
    retriever = service.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("*" * 20)
