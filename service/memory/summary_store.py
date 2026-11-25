from functools import cache
from threading import RLock
from time import monotonic

from pydantic import BaseModel, ValidationError
from redis import Redis
from redis.exceptions import RedisError

from utils.config_handler import cache_conf, memory_conf

from .models import ConversationSummary


class _StoredSummary(BaseModel):
    summary: ConversationSummary
    batch: str | None = None


class SummaryStore:
    def __init__(self, client: Redis | None = None, *, ttl_seconds: int = 7200):
        self.client = client
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._fallback = {}
        self._pending = set()

    def _write(self, key, record):
        if record is None:
            self.client.delete(key)
        else:
            self.client.set(key, record.model_dump_json(), ex=self.ttl_seconds)

    def _get(self, key):
        now = monotonic()
        for expired in [k for k, (until, _) in self._fallback.items() if until <= now]:
            del self._fallback[expired]
            self._pending.discard(expired)
        local = self._fallback.get(key, (0, None))[1]
        if self.client is None:
            return local
        try:
            if key in self._pending:
                self._write(key, local)
                self._pending.discard(key)
            raw = self.client.get(key)
            current = _StoredSummary.model_validate_json(raw) if raw else None
            if current:
                self._fallback[key] = (now + self.ttl_seconds, current)
            else:
                self._fallback.pop(key, None)
            return current
        except (RedisError, ValidationError):
            return local

    def _save(self, key, record):
        self._fallback[key] = (monotonic() + self.ttl_seconds, record)
        if self.client is not None:
            try:
                self._write(key, record)
                self._pending.discard(key)
            except RedisError:
                self._pending.add(key)

    def get(self, thread_id: str) -> ConversationSummary | None:
        with self._lock:
            record = self._get(f"agent:summary:{thread_id}")
            return record.summary if record else None

    def save(self, thread_id: str, summary: ConversationSummary) -> None:
        with self._lock:
            self._save(f"agent:summary:{thread_id}", _StoredSummary(summary=summary))

    def advance(self, thread_id: str, batch: str, update) -> ConversationSummary:
        key = f"agent:summary:{thread_id}"
        with self._lock:
            previous = self._get(key)
            if previous and previous.batch == batch:
                return previous.summary
            summary = update(previous.summary if previous else None)
            self._save(key, _StoredSummary(summary=summary, batch=batch))
            return summary

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._save(f"agent:summary:{thread_id}", None)


@cache
def get_summary_store() -> SummaryStore:
    return SummaryStore(
        Redis(
            host=cache_conf.get("chat_checkpoint_redis_host", "127.0.0.1"),
            port=int(cache_conf.get("chat_checkpoint_redis_port", 6380)),
            db=int(cache_conf.get("chat_checkpoint_redis_db", 0)),
            password=cache_conf.get("chat_checkpoint_redis_password") or None,
            socket_timeout=0.25,
            socket_connect_timeout=0.25,
        ),
        ttl_seconds=memory_conf["summary_ttl_seconds"],
    )
