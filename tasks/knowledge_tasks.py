from tasks.celery_app import celery_app
from service.knowledge_index_service import index_knowledge_file
from utils.logger_handler import logger


@celery_app.task(bind=True, max_retries=3)
def index_knowledge_file_task(self, file_id: int):
    try:
        logger.info(f"[Celery知识库入库]开始处理任务，file_id={file_id}")
        result = index_knowledge_file(file_id)
        logger.info(f"[Celery知识库入库]任务处理成功，file_id={file_id}, result={result}")
        return result

    except Exception as e:
        logger.error(f"[Celery知识库入库]任务处理失败，file_id={file_id}, error={str(e)}", exc_info=True)

        try:
            raise self.retry(exc=e, countdown=10)
        except self.MaxRetriesExceededError:
            logger.error(f"[Celery知识库入库]任务超过最大重试次数，file_id={file_id}")
            raise
