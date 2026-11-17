"""DuckDB destination: the credential-free local warehouse.

Each rollout owns one database file inside its workspace, so rollouts are
isolated by construction and a namespace can be cloned by copying a file.
DuckDB allows a single writing process, so the grader opens the file
read-only, briefly, and only after the agent's processes have finished.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from elt_rl.destinations.base import Destination, Namespace


def _quote(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


class DuckDBDestination(Destination):
    name: ClassVar[str] = "duckdb"
    display_name: ClassVar[str] = "DuckDB"
    dbt_adapter: ClassVar[str] = "duckdb"
    supports_clone: ClassVar[bool] = True

    def __init__(self, lock_retries: int = 20, lock_retry_delay: float = 0.25) -> None:
        self.lock_retries = lock_retries
        self.lock_retry_delay = lock_retry_delay
        # Read connections held open for the duration of a session(), by file.
        self._sessions: dict[str, Any] = {}
        self._sessions_lock = threading.Lock()

    # -- connections -------------------------------------------------------------
    def _connect(self, ns: Namespace, read_only: bool):
        import duckdb

        last_error: Exception | None = None
        for _ in range(self.lock_retries):
            try:
                return duckdb.connect(ns.database, read_only=read_only)
            except duckdb.IOException as e:  # another process still holds the lock
                last_error = e
                time.sleep(self.lock_retry_delay)
        raise RuntimeError(f"Could not open {ns.database}: {last_error}")

    # -- namespace lifecycle -------------------------------------------------------
    def new_namespace(
        self, task_name: str, rollout_id: str, host_workspace: str, agent_workspace: str
    ) -> Namespace:
        # The file stem is DuckDB's catalog name and must differ from the schema
        # name (the task name), otherwise "schema.table" is ambiguous.
        relative = os.path.join("warehouse", "warehouse.duckdb")
        return Namespace(
            database=os.path.join(host_workspace, relative),
            schema=task_name,
            agent_database=os.path.join(agent_workspace, relative),
        )

    def create_namespace(self, ns: Namespace, clone_from: Namespace | None = None) -> None:
        self.drop_namespace(ns)
        Path(ns.database).parent.mkdir(parents=True, exist_ok=True)
        if clone_from is not None:
            if clone_from.schema != ns.schema:
                raise ValueError("DuckDB clones keep the schema name of the source namespace")
            shutil.copyfile(clone_from.database, ns.database)
            return
        con = self._connect(ns, read_only=False)
        try:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(ns.schema)}")
        finally:
            con.close()

    def drop_namespace(self, ns: Namespace) -> None:
        for suffix in ("", ".wal"):
            Path(ns.database + suffix).unlink(missing_ok=True)

    # -- introspection ---------------------------------------------------------------
    @contextlib.contextmanager
    def session(self, ns: Namespace) -> Iterator[None]:
        # Opening the file is most of the cost of a small query, and a grading
        # pass runs a dozen or more of them.
        with self._sessions_lock:
            reuse = ns.database in self._sessions
        if reuse or not Path(ns.database).is_file():
            yield
            return
        con = self._connect(ns, read_only=True)
        with self._sessions_lock:
            self._sessions[ns.database] = con
        try:
            yield
        finally:
            with self._sessions_lock:
                self._sessions.pop(ns.database, None)
            con.close()

    def query(self, sql: str, ns: Namespace) -> pd.DataFrame:
        with self._sessions_lock:
            held = self._sessions.get(ns.database)
        if held is not None:
            return held.cursor().execute(sql).fetchdf()
        con = self._connect(ns, read_only=True)
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def list_tables(self, ns: Namespace, include_views: bool = False) -> list[str]:
        if not Path(ns.database).is_file():
            return []
        kinds = "('BASE TABLE', 'VIEW')" if include_views else "('BASE TABLE')"
        df = self.query(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE lower(table_schema) = lower('{ns.schema}') AND table_type IN {kinds}",
            ns,
        )
        return [str(x) for x in df.iloc[:, 0]]

    def table_columns(self, ns: Namespace, table: str) -> list[str]:
        df = self.query(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE lower(table_schema) = lower('{ns.schema}') "
            f"AND table_name = '{table}' ORDER BY ordinal_position",
            ns,
        )
        return [str(x) for x in df.iloc[:, 0]]

    def qualified(self, ns: Namespace, table: str) -> str:
        return f"{_quote(ns.schema)}.{_quote(table)}"

    def norm(self, ident: str) -> str:
        return ident.lower()

    def fingerprint_sql(self, relation: str, columns: list[str]) -> str:
        row = ", ".join(f"coalesce(cast({_quote(c)} AS VARCHAR), '<NULL>')" for c in columns)
        return (
            "SELECT count(*) AS n, "
            f"cast(coalesce(sum(hash(concat_ws(chr(31), {row})) % 1000000007), 0) AS VARCHAR) || ':' || "
            f"cast(coalesce(bit_xor(hash(concat_ws(chr(31), {row}))), 0) AS VARCHAR) AS h "
            f"FROM {relation}"
        )

    def drop_table(self, ns: Namespace, table: str) -> None:
        con = self._connect(ns, read_only=False)
        try:
            kinds = con.execute(
                "SELECT table_type FROM information_schema.tables "
                f"WHERE lower(table_schema) = lower('{ns.schema}') AND table_name = '{table}'"
            ).fetchall()
            for (kind,) in kinds:
                keyword = "VIEW" if kind == "VIEW" else "TABLE"
                con.execute(f"DROP {keyword} IF EXISTS {self.qualified(ns, table)}")
        finally:
            con.close()

    # -- agent side ------------------------------------------------------------------
    def agent_config(self, ns: Namespace, base: dict[str, Any]) -> dict[str, Any]:
        config = dict(base)
        config.update({"path": ns.for_agent, "schema": ns.schema})
        return config
