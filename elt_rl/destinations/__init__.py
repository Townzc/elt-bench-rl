"""Destination adapters and their registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.destinations.duckdb_dest import DuckDBDestination

DestinationFactory = Callable[..., Destination]

_REGISTRY: dict[str, DestinationFactory] = {}


def register_destination(name: str, factory: DestinationFactory) -> None:
    """Register a warehouse adapter under ``name`` (the ``config.yaml`` section)."""
    _REGISTRY[name] = factory


def _snowflake_factory(**kwargs: Any) -> Destination:
    # Imported lazily so the Snowflake driver stays an optional dependency.
    from elt_rl.destinations.snowflake_dest import SnowflakeDestination

    return SnowflakeDestination(**kwargs)


register_destination("duckdb", DuckDBDestination)
register_destination("snowflake", _snowflake_factory)


def get_destination(name: str, **kwargs: Any) -> Destination:
    """Build the adapter registered as ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown destination {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    return factory(**kwargs)


def destination_names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Destination",
    "DuckDBDestination",
    "Namespace",
    "destination_names",
    "get_destination",
    "register_destination",
]
