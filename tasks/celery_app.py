from celery import Celery

from utils.config_handler import celery_conf

celery_app = Celery(
    "agent_tasks",
    broker=celery_conf["broker_url"],
    backend=celery_conf["result_backend"],
    include=["tasks.knowledge_tasks"],
)

celery_app.conf.update(
    timezone=celery_conf.get("timezone", "Asia/Shanghai"),
    task_track_started=celery_conf.get("task_track_started", True),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
