import json
import redis

from utils.config_handler import cache_conf
from utils.logger_handler import logger


class AgentMemory:
    def __init__(self):
        self.enable = cache_conf.get("enable_agent_memory", True)
        self.prefix = cache_conf.get("agent_memory_prefix", "agent:memory:")
        self.ttl = cache_conf.get("agent_memory_ttl", 7200)

        self.client = redis.Redis(
            host=cache_conf["redis_host"],
            port=cache_conf["redis_port"],
            password=cache_conf.get("redis_password"),
            db=cache_conf.get("redis_db", 0),
            socket_timeout=cache_conf.get("redis_timeout", 5),
            decode_responses=True,
        )

    def build_key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def get_memory(self, session_id: str) -> dict:
        if not self.enable:
            return {}

        try:
            value = self.client.get(self.build_key(session_id))
            if not value:
                return {}

            return json.loads(value)

        except Exception as e:
            logger.warning(f"[Agent记忆]读取失败：{str(e)}")
            return {}

    def set_memory(self, session_id: str, memory: dict) -> bool:
        if not self.enable:
            return False

        try:
            self.client.setex(
                self.build_key(session_id),
                self.ttl,
                json.dumps(memory, ensure_ascii=False),
            )
            return True

        except Exception as e:
            logger.warning(f"[Agent记忆]写入失败：{str(e)}")
            return False

    def update_memory(self, session_id: str, **kwargs) -> dict:
        memory = self.get_memory(session_id)

        for key, value in kwargs.items():
            if value is not None:
                memory[key] = value

        self.set_memory(session_id, memory)

        return memory

    def clear_memory(self, session_id: str) -> bool:
        if not self.enable:
            return False

        try:
            self.client.delete(self.build_key(session_id))
            return True

        except Exception as e:
            logger.warning(f"[Agent记忆]清理失败：{str(e)}")
            return False