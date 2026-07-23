# 实现知识库文件更新：对比 MD5、删除旧 chunk、重新入库并更新 MySQL 状态。
import os

from rag.vector_store import VectorStoreService
from service.knowledge_file_service import (
    get_file_by_id,
    update_file_status,
    mark_file_failed,
    update_knowledge_file_record,
)
from utils.file_handler import get_file_md5_hex


def update_knowledge_file(file_id: int, new_file_path: str) -> dict:
    old_file = get_file_by_id(file_id)

    if not old_file:
        raise ValueError(f"知识库文件记录不存在，file_id={file_id}")

    new_md5 = get_file_md5_hex(new_file_path)
    if not new_md5:
        raise ValueError(f"新文件 MD5 计算失败：{new_file_path}")

    old_md5 = old_file["md5"]

    if new_md5 == old_md5:
        return {
            "file_id": file_id,
            "status": "skipped",
            "reason": "文件内容未变化，跳过更新",
            "old_md5": old_md5,
            "new_md5": new_md5,
        }

    filename = os.path.basename(new_file_path)
    file_type = filename.rsplit(".", 1)[-1].lower()

    update_file_status(file_id, "updating")

    service = VectorStoreService()

    try:
        old_chunks = service.get_chunks_by_file_id(file_id)
        old_chunk_ids = [chunk["id"] for chunk in old_chunks]

        index_report = service.load_single_document(
            new_file_path,
            file_id=file_id,
            version=1,
        )

        deleted_count = service.delete_chunks_by_ids(old_chunk_ids)

        update_knowledge_file_record(
            file_id=file_id,
            filename=filename,
            file_path=new_file_path,
            file_type=file_type,
            md5=new_md5,
            chunk_count=index_report["chunk_count"],
        )

        return {
            "file_id": file_id,
            "status": "success",
            "old_md5": old_md5,
            "new_md5": new_md5,
            "old_chunk_count": len(old_chunk_ids),
            "deleted_chunk_count": deleted_count,
            "new_chunk_count": index_report["chunk_count"],
            "index_report": index_report,
        }

    except Exception as e:
        mark_file_failed(file_id, str(e))
        raise