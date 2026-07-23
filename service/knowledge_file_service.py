from utils.db_handler import fetch_one, fetch_all, execute
import os

from utils.file_handler import get_file_md5_hex

def get_file_by_md5(md5: str):
    return fetch_one(
        "SELECT * FROM knowledge_files WHERE md5 = :md5",
        {"md5": md5},
    )


def create_knowledge_file(filename: str, file_path: str, file_type: str, md5: str):
    sql = """
    INSERT INTO knowledge_files (filename, file_path, file_type, md5, status)
    VALUES (:filename, :file_path, :file_type, :md5, 'pending')
    """
    return execute(sql, {
        "filename": filename,
        "file_path": file_path,
        "file_type": file_type,
        "md5": md5,
    })


def update_file_status(file_id: int, status: str, error_message: str | None = None):
    sql = """
    UPDATE knowledge_files
    SET status = :status, error_message = :error_message
    WHERE id = :file_id
    """
    return execute(sql, {
        "file_id": file_id,
        "status": status,
        "error_message": error_message,
    })


def register_knowledge_file(file_path: str):
    filename = os.path.basename(file_path)
    file_type = filename.rsplit(".", 1)[-1].lower()
    md5 = get_file_md5_hex(file_path)

    if not md5:
        raise ValueError(f"文件MD5计算失败：{file_path}")

    existing = get_file_by_md5(md5)

    if existing:
        status = existing["status"]

        if status == "success":
            return {
                "created": False,
                "should_index": False,
                "reason": "文件已成功入库",
                "file": existing,
            }

        if status in ("pending", "indexing"):
            return {
                "created": False,
                "should_index": False,
                "reason": f"文件已存在，当前状态为{status}",
                "file": existing,
            }

        if status == "failed":
            update_file_status(existing["id"], "pending", None)
            return {
                "created": False,
                "should_index": True,
                "reason": "文件曾入库失败，已重置为pending，可重试",
                "file": get_file_by_md5(md5),
            }

    create_knowledge_file(filename, file_path, file_type, md5)
    created = get_file_by_md5(md5)

    return {
        "created": True,
        "should_index": True,
        "reason": "文件记录创建成功",
        "file": created,
    }


def mark_file_success(file_id: int, chunk_count: int):
    sql = """
    UPDATE knowledge_files
    SET status = 'success',
        chunk_count = :chunk_count,
        error_message = NULL,
        indexed_at = NOW()
    WHERE id = :file_id
    """
    return execute(sql, {
        "file_id": file_id,
        "chunk_count": chunk_count,
    })


def mark_file_failed(file_id: int, error_message: str):
    sql = """
    UPDATE knowledge_files
    SET status = 'failed',
        error_message = :error_message
    WHERE id = :file_id
    """
    return execute(sql, {
        "file_id": file_id,
        "error_message": error_message,
    })


def list_knowledge_files():
    return fetch_all(
        "SELECT * FROM knowledge_files ORDER BY created_at DESC"
    )

def get_file_by_id(file_id: int):
    return fetch_one(
        "SELECT * FROM knowledge_files WHERE id = :file_id",
        {"file_id": file_id},
    )


def update_knowledge_file_record(
    file_id: int,
    filename: str,
    file_path: str,
    file_type: str,
    md5: str,
    chunk_count: int,
):
    sql = """
    UPDATE knowledge_files
    SET filename = :filename,
        file_path = :file_path,
        file_type = :file_type,
        md5 = :md5,
        chunk_count = :chunk_count,
        status = 'success',
        error_message = NULL,
        indexed_at = NOW()
    WHERE id = :file_id
    """
    return execute(sql, {
        "file_id": file_id,
        "filename": filename,
        "file_path": file_path,
        "file_type": file_type,
        "md5": md5,
        "chunk_count": chunk_count,
    })
