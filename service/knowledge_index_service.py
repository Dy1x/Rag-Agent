from service.knowledge_file_service import (
    update_file_status,
    mark_file_success,
    mark_file_failed,
)
from utils.db_handler import fetch_one
from rag.vector_store import VectorStoreService


def index_knowledge_file(file_id: int):
    file_record = fetch_one(
        "SELECT * FROM knowledge_files WHERE id = :file_id",
        {"file_id": file_id},
    )

    if not file_record:
        raise ValueError(f"知识库文件记录不存在，file_id={file_id}")

    update_file_status(file_id, "indexing")

    try:
        service = VectorStoreService()
        index_report = service.load_single_document(
            file_record["file_path"],
            file_id=file_id,
            version=1,
        )

        mark_file_success(file_id, index_report["chunk_count"])

        return {
            "file_id": file_id,
            "status": "success",
            "chunk_count": index_report["chunk_count"],
            "index_report": index_report,
        }

    except Exception as e:
        mark_file_failed(file_id, str(e))
        raise
