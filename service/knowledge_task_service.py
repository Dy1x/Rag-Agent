from tasks.knowledge_tasks import index_knowledge_file_task
from celery.result import AsyncResult
from tasks.celery_app import celery_app

def submit_index_task(file_id: int):
    async_result = index_knowledge_file_task.delay(file_id)

    return {
        "file_id": file_id,
        "task_id": async_result.id,
        "status": "submitted",
    }

def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
    }