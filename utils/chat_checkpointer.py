"""
LangGraph Redis聊天Checkpointer。

使用CentOS上端口6380的Redis Stack。

ShallowRedisSaver只保存每个thread_id的最新状态，
最近10轮消息由middleware.py负责控制。
"""

from langgraph.checkpoint.redis.shallow import (
    ShallowRedisSaver,
)
from redis import Redis
from redis.exceptions import RedisError

from utils.config_handler import cache_conf
from utils.logger_handler import logger


def get_checkpoint_password() -> str:
    """
    从cache.yml读取Redis Stack密码。
    """

    password = cache_conf.get(
        "chat_checkpoint_redis_password"
    )

    if not password:
        raise RuntimeError(
            "cache.yml中没有配置"
            "chat_checkpoint_redis_password"
        )

    return str(password)


def create_chat_checkpointer(
) -> ShallowRedisSaver:
    """
    创建并初始化Redis Checkpointer。
    """

    redis_host = cache_conf.get(
        "chat_checkpoint_redis_host",
        "127.0.0.1",
    )

    redis_port = int(
        cache_conf.get(
            "chat_checkpoint_redis_port",
            6380,
        )
    )

    redis_db = int(
        cache_conf.get(
            "chat_checkpoint_redis_db",
            0,
        )
    )

    redis_timeout = float(
        cache_conf.get(
            "chat_checkpoint_redis_timeout",
            5,
        )
    )

    ttl_minutes = int(
        cache_conf.get(
            "chat_checkpoint_ttl_minutes",
            120,
        )
    )

    refresh_on_read = bool(
        cache_conf.get(
            "chat_checkpoint_refresh_on_read",
            True,
        )
    )

    checkpoint_prefix = cache_conf.get(
        "chat_checkpoint_prefix",
        "agent:checkpoint",
    )

    checkpoint_write_prefix = (
        cache_conf.get(
            "chat_checkpoint_write_prefix",
            "agent:checkpoint:write",
        )
    )

    redis_client = Redis(
        host=redis_host,
        port=redis_port,
        password=get_checkpoint_password(),
        db=redis_db,
        socket_connect_timeout=redis_timeout,
        socket_timeout=redis_timeout,
        health_check_interval=30,

        # RedisSaver要求使用bytes
        decode_responses=False,
    )

    try:
        redis_client.ping()

        logger.info(
            "[Chat Checkpointer] "
            "Redis Stack连接成功："
            f"{redis_host}:{redis_port}/{redis_db}"
        )

        checkpointer = ShallowRedisSaver(
            redis_client=redis_client,
            ttl={
                "default_ttl": ttl_minutes,
                "refresh_on_read": (
                    refresh_on_read
                ),
            },
            checkpoint_prefix=(
                checkpoint_prefix
            ),
            checkpoint_write_prefix=(
                checkpoint_write_prefix
            ),
        )

        # 首次使用时创建RedisJSON和RediSearch索引。
        # 该方法可以安全地重复调用。
        checkpointer.setup()

        logger.info(
            "[Chat Checkpointer] "
            "初始化成功："
            f"ttl_minutes={ttl_minutes}, "
            "refresh_on_read="
            f"{refresh_on_read}"
        )

        return checkpointer

    except RedisError as error:
        raise RuntimeError(
            "Redis Stack连接失败："
            f"{redis_host}:{redis_port}/{redis_db}，"
            f"原因：{error}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            "LangGraph Redis Checkpointer"
            "初始化失败。"
            "请确认RedisJSON和RediSearch可用。"
            f"原因：{error}"
        ) from error


chat_checkpointer = (
    create_chat_checkpointer()
)