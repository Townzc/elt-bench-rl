"""Destination (data warehouse) adapter interface.

Everything the environment needs from a warehouse goes through this class, so
supporting another warehouse means adding one subclass and registering it; the
environment, tools, grader and training code stay untouched.

An adapter has two faces:

* **Host side** (trusted, grader credentials): create / clone / drop the
  per-rollout namespace, and read tables for grading.
* **Agent side**: the ``config.yaml`` block the agent receives, which is what
  it uses to point Airbyte and dbt at its own namespace.
"""

from __future__ import annotations

import contextlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

# ``FROM schema.table`` / ``JOIN schema.table`` in warehouse-neutral evaluation SQL.
_RELATION_PATTERN = re.compile(
    r"\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Namespace:
    """Where one rollout's tables live.

    ``database`` is the host-side name (a database/catalog name, or a file
    path for file-based warehouses). ``agent_database`` is the same location
    as the agent's sandbox sees it, when that differs (e.g. a bind mount).
    """

    database: str
    schema: str
    agent_database: str | None = None

    @property
    def for_agent(self) -> str:
        return self.agent_database or self.database


class Destination(ABC):
    """A data warehouse the ELT pipeline loads into."""

    name: ClassVar[str]
    """Registry key and the destination section name in ``config.yaml``."""
    display_name: ClassVar[str]
    """Name used in prompts."""
    dbt_adapter: ClassVar[str]
    """dbt adapter type the agent should use."""
    supports_clone: ClassVar[bool] = False
    """Whether ``create_namespace(clone_from=...)`` is cheap (snapshot/zero-copy)."""
    grading_concurrency: ClassVar[int] = 1
    """Checks of one rollout that may run concurrently against the warehouse."""

    # -- namespace lifecycle (host side) ---------------------------------------
    @abstractmethod
    def new_namespace(
        self,
        task_name: str,
        rollout_id: str,
        host_workspace: str,
        agent_workspace: str,
    ) -> Namespace:
        """Choose an isolated namespace for one rollout (no side effects)."""

    @abstractmethod
    def create_namespace(self, ns: Namespace, clone_from: Namespace | None = None) -> None:
        """Drop and recreate ``ns``; optionally as a copy of ``clone_from``.

        Cloning backs the transformation-only start state, where the source
        tables are already loaded.
        """

    @abstractmethod
    def drop_namespace(self, ns: Namespace) -> None:
        """Remove the namespace and everything in it. Must be idempotent."""

    # -- introspection used by the grader (host side) --------------------------
    @abstractmethod
    def list_tables(self, ns: Namespace, include_views: bool = False) -> list[str]:
        """Relation names in the namespace, in the warehouse's native case.

        Source tables must be base tables; data models may be views (dbt's
        default materialization), so the grader asks for views there.
        """

    @abstractmethod
    def table_columns(self, ns: Namespace, table: str) -> list[str]:
        """Column names of a table, in ordinal order."""

    @abstractmethod
    def query(self, sql: str, ns: Namespace) -> pd.DataFrame:
        """Run a read query with the grader's credentials."""

    @abstractmethod
    def qualified(self, ns: Namespace, table: str) -> str:
        """Fully qualified, safely quoted reference to ``table`` in ``ns``."""

    @abstractmethod
    def norm(self, ident: str) -> str:
        """Fold an identifier to the warehouse's canonical case for matching."""

    @abstractmethod
    def fingerprint_sql(self, relation: str, columns: list[str]) -> str:
        """SQL returning one row ``(row_count, content_hash)`` for a relation.

        The hash must be order-independent and computed inside the warehouse,
        so comparing two loads costs one small result row each.
        """

    @abstractmethod
    def drop_table(self, ns: Namespace, table: str) -> None:
        """Drop one table or view if it exists (used by the replay check)."""

    # -- agent side --------------------------------------------------------------
    @abstractmethod
    def agent_config(self, ns: Namespace, base: dict[str, Any]) -> dict[str, Any]:
        """Destination block for the agent's ``config.yaml``.

        ``base`` is the block shipped with the task bundle (it may already
        carry account/credential fields); return it pointed at ``ns``.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, nothing to release by default
        """Release host-side connections."""

    @contextlib.contextmanager
    def session(self, ns: Namespace) -> Iterator[None]:
        """Scope of a batch of read queries on one namespace (one grading pass).

        Adapters whose connections are expensive to open may keep one open for
        the duration. The default does nothing.
        """
        del ns
        yield

    # -- shared helpers ------------------------------------------------------------
    def match_table(self, names: list[str], table: str) -> str | None:
        """The entry of ``names`` that is ``table`` under case folding, or None."""
        wanted = self.norm(table)
        return next((name for name in names if self.norm(name) == wanted), None)

    def resolve_table(
        self, ns: Namespace, table: str, include_views: bool = False
    ) -> str | None:
        """Actual name of ``table`` in ``ns`` (case-folded match), or None."""
        return self.match_table(self.list_tables(ns, include_views=include_views), table)

    def table_size(self, ns: Namespace, table: str) -> int:
        df = self.query(f"SELECT COUNT(*) FROM {self.qualified(ns, table)}", ns)
        return int(df.iloc[0, 0])

    def fetch_table(self, ns: Namespace, table: str) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {self.qualified(ns, table)}", ns)

    def table_fingerprint(self, ns: Namespace, table: str, columns: list[str]) -> tuple[int, str]:
        df = self.query(self.fingerprint_sql(self.qualified(ns, table), columns), ns)
        return int(df.iloc[0, 0]), str(df.iloc[0, 1])

    def query_eval_sql(self, sql: str, ns: Namespace) -> pd.DataFrame:
        """Run warehouse-neutral evaluation SQL against a rollout namespace.

        Benchmark queries name relations as ``<task>.<table>``; each reference
        is rewritten to the physical location of this rollout's namespace.
        """

        def replace(match: re.Match[str]) -> str:
            keyword, _logical_schema, table = match.groups()
            actual = self.resolve_table(ns, table, include_views=True) or table
            return f"{keyword} {self.qualified(ns, actual)}"

        return self.query(_RELATION_PATTERN.sub(replace, sql), ns)
