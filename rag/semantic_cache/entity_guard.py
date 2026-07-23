from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


@dataclass(frozen=True)
class ExtractedEntity:
    dimension: str
    dimension_label: str
    value: str
    value_label: str
    matched_alias: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dimension": self.dimension,
            "dimension_label": self.dimension_label,
            "value": self.value,
            "value_label": self.value_label,
            "matched_alias": self.matched_alias,
        }


@dataclass(frozen=True)
class EntityConflict:
    dimension: str
    dimension_label: str
    query_values: tuple[str, ...]
    query_labels: tuple[str, ...]
    cached_values: tuple[str, ...]
    cached_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "dimension_label": self.dimension_label,
            "query_values": list(self.query_values),
            "query_labels": list(self.query_labels),
            "cached_values": list(self.cached_values),
            "cached_labels": list(self.cached_labels),
        }


@dataclass(frozen=True)
class EntityGuardResult:
    compatible: bool
    status: str
    query_entities: tuple[ExtractedEntity, ...]
    cached_entities: tuple[ExtractedEntity, ...]
    conflicts: tuple[EntityConflict, ...]
    shared_dimensions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "status": self.status,
            "query_entities": [
                entity.to_dict()
                for entity in self.query_entities
            ],
            "cached_entities": [
                entity.to_dict()
                for entity in self.cached_entities
            ],
            "conflicts": [
                conflict.to_dict()
                for conflict in self.conflicts
            ],
            "shared_dimensions": list(
                self.shared_dimensions
            ),
        }


class SemanticCacheEntityGuard:
    """
    基于业务规则检查两个问题是否存在实体冲突。

    该组件只处理明确冲突，不负责判断完整语义是否等价。
    没有发现冲突的候选仍需交给Cross-Encoder验证。
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        configured_path = (
            Path(config_path)
            if config_path is not None
            else Path(
                get_abs_path(
                    "config/semantic_cache_entities.yml"
                )
            )
        )

        if not configured_path.exists():
            raise FileNotFoundError(
                f"找不到实体配置文件：{configured_path}"
            )

        with configured_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file) or {}

        self.enabled = bool(
            config.get("enabled", True)
        )

        self.mode = str(
            config.get(
                "mode",
                "conservative",
            )
        ).strip().lower()

        self.dimensions = config.get(
            "dimensions",
            {},
        )

        if self.mode not in {
            "conservative",
            "strict",
        }:
            raise ValueError(
                "实体校验mode只支持"
                "conservative或strict"
            )

        if not isinstance(self.dimensions, dict):
            raise ValueError(
                "dimensions必须是字典"
            )

        self._validate_config()
        self._alias_rules = self._build_alias_rules()

    def _validate_config(self) -> None:
        for dimension, dimension_config in (
            self.dimensions.items()
        ):
            if not isinstance(dimension_config, dict):
                raise ValueError(
                    f"{dimension}配置必须是字典"
                )

            values = dimension_config.get(
                "values",
                {},
            )

            if not isinstance(values, dict):
                raise ValueError(
                    f"{dimension}.values必须是字典"
                )

            for value, value_config in values.items():
                aliases = value_config.get(
                    "aliases",
                    [],
                )

                if not aliases:
                    raise ValueError(
                        f"{dimension}.{value}"
                        "必须至少配置一个alias"
                    )

    def _build_alias_rules(
        self,
    ) -> list[dict[str, str]]:
        rules = []

        for dimension, dimension_config in (
            self.dimensions.items()
        ):
            dimension_label = str(
                dimension_config.get(
                    "label",
                    dimension,
                )
            )

            values = dimension_config.get(
                "values",
                {},
            )

            for value, value_config in values.items():
                value_label = str(
                    value_config.get(
                        "label",
                        value,
                    )
                )

                aliases = value_config.get(
                    "aliases",
                    [],
                )

                for alias in aliases:
                    normalized_alias = str(
                        alias
                    ).strip()

                    if not normalized_alias:
                        continue

                    rules.append(
                        {
                            "dimension": dimension,
                            "dimension_label": (
                                dimension_label
                            ),
                            "value": value,
                            "value_label": value_label,
                            "alias": normalized_alias,
                        }
                    )

        # 优先匹配长词，减少“充电座”抢先匹配
        # “找不到充电座”等完整表达的情况。
        rules.sort(
            key=lambda item: len(item["alias"]),
            reverse=True,
        )

        return rules

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (
            text.strip()
            .replace(" ", "")
            .replace("\n", "")
            .lower()
        )

    def extract(
        self,
        text: str,
    ) -> list[ExtractedEntity]:
        if not self.enabled:
            return []

        if not text or not text.strip():
            return []

        normalized_text = self._normalize_text(text)
        extracted = []
        seen = set()

        for rule in self._alias_rules:
            alias = self._normalize_text(
                rule["alias"]
            )

            if alias not in normalized_text:
                continue

            key = (
                rule["dimension"],
                rule["value"],
            )

            if key in seen:
                continue

            seen.add(key)

            extracted.append(
                ExtractedEntity(
                    dimension=rule["dimension"],
                    dimension_label=(
                        rule["dimension_label"]
                    ),
                    value=rule["value"],
                    value_label=rule["value_label"],
                    matched_alias=rule["alias"],
                )
            )

        return extracted

    @staticmethod
    def _group_by_dimension(
        entities: list[ExtractedEntity],
    ) -> dict[str, dict[str, ExtractedEntity]]:
        grouped = {}

        for entity in entities:
            grouped.setdefault(
                entity.dimension,
                {},
            )[entity.value] = entity

        return grouped

    def check(
        self,
        query: str,
        cached_query: str,
    ) -> EntityGuardResult:
        if not self.enabled:
            return EntityGuardResult(
                compatible=True,
                status="disabled",
                query_entities=(),
                cached_entities=(),
                conflicts=(),
                shared_dimensions=(),
            )

        query_entities = self.extract(query)
        cached_entities = self.extract(
            cached_query
        )

        query_grouped = self._group_by_dimension(
            query_entities
        )
        cached_grouped = self._group_by_dimension(
            cached_entities
        )

        shared_dimensions = sorted(
            set(query_grouped)
            & set(cached_grouped)
        )

        conflicts = []

        for dimension in shared_dimensions:
            query_values = set(
                query_grouped[dimension]
            )
            cached_values = set(
                cached_grouped[dimension]
            )

            # 只要同一维度存在交集，就不认定为冲突。
            # 例如“主刷或边刷”和“边刷”存在交集。
            if query_values & cached_values:
                continue

            dimension_config = self.dimensions[
                dimension
            ]

            query_matches = (
                query_grouped[dimension]
            )
            cached_matches = (
                cached_grouped[dimension]
            )

            conflicts.append(
                EntityConflict(
                    dimension=dimension,
                    dimension_label=str(
                        dimension_config.get(
                            "label",
                            dimension,
                        )
                    ),
                    query_values=tuple(
                        sorted(query_values)
                    ),
                    query_labels=tuple(
                        query_matches[value].value_label
                        for value in sorted(query_values)
                    ),
                    cached_values=tuple(
                        sorted(cached_values)
                    ),
                    cached_labels=tuple(
                        cached_matches[value].value_label
                        for value in sorted(
                            cached_values
                        )
                    ),
                )
            )

        if conflicts:
            result = EntityGuardResult(
                compatible=False,
                status="conflict",
                query_entities=tuple(
                    query_entities
                ),
                cached_entities=tuple(
                    cached_entities
                ),
                conflicts=tuple(conflicts),
                shared_dimensions=tuple(
                    shared_dimensions
                ),
            )

            logger.info(
                "[语义缓存实体校验]发现实体冲突，"
                f"query={query}, "
                f"cached_query={cached_query}, "
                f"dimensions="
                f"{[item.dimension for item in conflicts]}"
            )

            return result

        has_shared_dimension = bool(
            shared_dimensions
        )

        if has_shared_dimension:
            status = "compatible"
        elif self.mode == "strict":
            status = "insufficient_entities"
        else:
            status = "unknown"

        compatible = not (
            self.mode == "strict"
            and not has_shared_dimension
        )

        return EntityGuardResult(
            compatible=compatible,
            status=status,
            query_entities=tuple(
                query_entities
            ),
            cached_entities=tuple(
                cached_entities
            ),
            conflicts=(),
            shared_dimensions=tuple(
                shared_dimensions
            ),
        )