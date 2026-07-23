import hashlib
import redis
import time

from utils.config_handler import cache_conf
from utils.logger_handler import logger


class RedisCache:
    def __init__(self):
        self.enable = cache_conf.get("enable_rag_cache", True)
        self.prefix = cache_conf.get("rag_cache_prefix", "agent:rag:")
        self.ttl = cache_conf.get("rag_cache_ttl", 3600)

        self.client = redis.Redis(
            host=cache_conf["redis_host"],
            port=cache_conf["redis_port"],
            password=cache_conf.get("redis_password"),
            db=cache_conf.get("redis_db", 0),
            socket_timeout=cache_conf.get("redis_timeout", 5),
            decode_responses=True,
        )
        self.lock_enable = cache_conf.get("enable_rag_lock", True)
        self.lock_prefix = cache_conf.get("rag_lock_prefix", "agent:rag:lock:")
        self.lock_ttl = cache_conf.get("rag_lock_ttl", 30)
        self.lock_wait_timeout = cache_conf.get("rag_lock_wait_timeout", 10)
        self.lock_retry_interval = cache_conf.get("rag_lock_retry_interval", 0.2)

    def build_key(self, query: str) -> str:
        normalized_query = query.strip().lower()
        query_hash = hashlib.md5(normalized_query.encode("utf-8")).hexdigest()
        return f"{self.prefix}{query_hash}"

    def get(self, query: str) -> str | None:
        if not self.enable:
            return None

        try:
            return self.client.get(self.build_key(query))
        except Exception as e:
            logger.warning(f"[Redis缓存]读取失败：{str(e)}")
            return None

    def set(self, query: str, value: str) -> bool:
        if not self.enable:
            return False

        try:
            key = self.build_key(query)
            self.client.setex(key, self.ttl, value)
            return True
        except Exception as e:
            logger.warning(f"[Redis缓存]写入失败：{str(e)}")
            return False

    def build_lock_key(self, query: str) -> str:
        cache_key = self.build_key(query)
        return cache_key.replace(self.prefix, self.lock_prefix, 1)

    def acquire_lock(self, query: str, token: str) -> bool:
        if not self.enable or not self.lock_enable:
            return False

        try:
            lock_key = self.build_lock_key(query)
            return bool(
                self.client.set(
                    lock_key,
                    token,
                    nx=True,
                    ex=self.lock_ttl,
                )
            )
        except Exception as e:
            logger.warning(f"[Redis锁]获取锁失败：{str(e)}")
            return False

    def release_lock(self, query: str, token: str) -> bool:
        if not self.enable or not self.lock_enable:
            return False

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            lock_key = self.build_lock_key(query)
            return bool(self.client.eval(lua_script, 1, lock_key, token))
        except Exception as e:
            logger.warning(f"[Redis锁]释放锁失败：{str(e)}")
            return False

    def wait_for_cache(self, query: str) -> str | None:
        if not self.enable:
            return None

        deadline = time.time() + self.lock_wait_timeout

        while time.time() < deadline:
            cached_answer = self.get(query)
            if cached_answer:
                return cached_answer

            time.sleep(self.lock_retry_interval)

        return None
