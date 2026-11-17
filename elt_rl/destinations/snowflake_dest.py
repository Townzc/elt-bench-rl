"""Snowflake destination: the warehouse the official ELT-Bench tasks target.

Layout follows the official harness: one shared database with one schema per
attempt. Here every *rollout* gets its own schema (``<TASK>__<ROLLOUT_ID>``)
so the rollouts of a group never see each other's tables by default name, and
evaluation SQL written as ``<task>.<table>`` is rewritten to that schema.

Two sets of credentials are involved and must stay separate:

* the **admin credential** (host only; ``setup/destination/snowflake_credential.json``
  in the ELT-Bench checkout) creates/clones/drops namespaces and reads tables
  for grading;
* the **agent credential** (the ``snowflake.config`` block of the task's
  ``config.yaml``) is what Airbyte and dbt use inside the sandbox.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from elt_rl.destinations.base import Destination, Namespace

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _quote(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


class SnowflakeDestination(Destination):
    name: ClassVar[str] = "snowflake"
    display_name: ClassVar[str] = "Snowflake"
    dbt_adapter: ClassVar[str] = "snowflake"
    supports_clone: ClassVar[bool] = True  # zero-copy CLONE
    grading_concurrency: ClassVar[int] = 8  # network round trips dominate

    def __init__(
        self,
        credential: dict[str, Any] | str | Path,
        database: str = "ELT_RL",
        agent_role: str | None = "AIRBYTE_ROLE",
        connect: Callable[[], Any] | None = None,
    ) -> None:
        """
        Args:
            credential: Admin connection kwargs for ``snowflake.connector.connect``
                (``account``, ``user``, ``password``, optional ``role`` /
                ``warehouse``), or a path to a JSON file holding them.
            database: Shared database holding one schema per rollout.
            agent_role: Role the agent's credential uses; it is granted
                ownership of each rollout schema. ``None`` skips the grants.
            connect: Connection factory override (tests).
        """
        if isinstance(credential, (str, Path)):
            with Path(credential).expanduser().open(encoding="utf-8") as f:
                credential = json.load(f)
        self._credential = dict(credential)
        if not _IDENT.match(database):
            raise ValueError(f"Invalid Snowflake database name: {database!r}")
        self.database = database.upper()
        self.agent_role = agent_role.upper() if agent_role else None
        self._connect_fn = connect
        self._conn: Any = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------------
    def _connection(self) -> Any:
        if self._conn is None:
            if self._connect_fn is not None:
                self._conn = self._connect_fn()
            else:
                import snowflake.connector

                self._conn = snowflake.connector.connect(login_timeout=60, **self._credential)
        return self._conn

    def _execute(self, sql: str, fetch: bool = False) -> tuple[list[str], list[tuple]]:
        # The connector allows threads to share a connection (each with its own
        # cursor), so only connection setup is serialized.
        with self._lock:
            connection = self._connection()
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            if not fetch:
                return [], []
            columns = [d[0] for d in cursor.description]
            return columns, list(cursor.fetchall())
        finally:
            cursor.close()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- namespace lifecycle ---------------------------------------------------------
    def new_namespace(
        self, task_name: str, rollout_id: str, host_workspace: str, agent_workspace: str
    ) -> Namespace:
        schema = f"{task_name}__{rollout_id}".upper()
        if not _IDENT.match(schema):
            raise ValueError(f"Invalid Snowflake schema name: {schema!r}")
        return Namespace(database=self.database, schema=schema)

    def _drop_airbyte_raw_tables(self, ns: Namespace) -> None:
        # Airbyte keeps raw records in AIRBYTE_INTERNAL as <schema>_raw__stream_<stream>;
        # leftovers would be re-materialized by a later append sync.
        _, rows = self._execute(f"SHOW TABLES IN DATABASE {_quote(ns.database)}", fetch=True)
        prefix = f"{ns.schema.upper()}_RAW__STREAM_"
        for row in rows:
            table_name, schema_name = str(row[1]), str(row[3])
            if schema_name.upper() == "AIRBYTE_INTERNAL" and table_name.upper().startswith(prefix):
                self._execute(
                    f"DROP TABLE IF EXISTS {_quote(ns.database)}.{_quote(schema_name)}."
                    f"{_quote(table_name)}"
                )

    def create_namespace(self, ns: Namespace, clone_from: Namespace | None = None) -> None:
        database, schema = _quote(ns.database), _quote(ns.schema)
        self._execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        self._execute(f"DROP SCHEMA IF EXISTS {database}.{schema} CASCADE")
        self._drop_airbyte_raw_tables(ns)
        if clone_from is not None:
            source = f"{_quote(clone_from.database.upper())}.{_quote(clone_from.schema.upper())}"
            self._execute(f"CREATE SCHEMA {database}.{schema} CLONE {source}")
        else:
            self._execute(f"CREATE SCHEMA {database}.{schema}")
        if self.agent_role:
            role = _quote(self.agent_role)
            # CREATE SCHEMA lets Airbyte create its raw-data schema in the database.
            self._execute(f"GRANT USAGE, CREATE SCHEMA ON DATABASE {database} TO ROLE {role}")
            copy_grants = " COPY CURRENT GRANTS" if clone_from is not None else ""
            self._execute(
                f"GRANT OWNERSHIP ON SCHEMA {database}.{schema} TO ROLE {role}{copy_grants}"
            )
            if clone_from is not None:
                self._execute(
                    f"GRANT OWNERSHIP ON ALL TABLES IN SCHEMA {database}.{schema} "
                    f"TO ROLE {role} COPY CURRENT GRANTS"
                )

    def drop_namespace(self, ns: Namespace) -> None:
        self._execute(f"DROP SCHEMA IF EXISTS {_quote(ns.database)}.{_quote(ns.schema)} CASCADE")
        self._drop_airbyte_raw_tables(ns)

    # -- introspection -------------------------------------------------------------------
    def query(self, sql: str, ns: Namespace) -> pd.DataFrame:
        columns, rows = self._execute(sql, fetch=True)
        return pd.DataFrame(rows, columns=columns)

    def list_tables(self, ns: Namespace, include_views: bool = False) -> list[str]:
        kinds = "('BASE TABLE', 'VIEW')" if include_views else "('BASE TABLE')"
        df = self.query(
            f"SELECT table_name FROM {_quote(ns.database.upper())}.information_schema.tables "
            f"WHERE table_schema = {_literal(ns.schema.upper())} AND table_type IN {kinds}",
            ns,
        )
        return [str(x) for x in df.iloc[:, 0]]

    def table_columns(self, ns: Namespace, table: str) -> list[str]:
        df = self.query(
            f"SELECT column_name FROM {_quote(ns.database.upper())}.information_schema.columns "
            f"WHERE table_schema = {_literal(ns.schema.upper())} AND table_name = {_literal(table)} "
            "ORDER BY ordinal_position",
            ns,
        )
        return [str(x) for x in df.iloc[:, 0]]

    def qualified(self, ns: Namespace, table: str) -> str:
        # Database and schema are created unquoted (upper case); the table name
        # is the exact name found in information_schema.
        return f"{_quote(ns.database.upper())}.{_quote(ns.schema.upper())}.{_quote(table)}"

    def norm(self, ident: str) -> str:
        return ident.upper()

    def fingerprint_sql(self, relation: str, columns: list[str]) -> str:
        row = ", ".join(f"TO_VARCHAR({_quote(c)})" for c in columns)
        return f"SELECT COUNT(*) AS n, TO_VARCHAR(HASH_AGG({row})) AS h FROM {relation}"

    def drop_table(self, ns: Namespace, table: str) -> None:
        relation = self.qualified(ns, table)
        df = self.query(
            f"SELECT table_type FROM {_quote(ns.database.upper())}.information_schema.tables "
            f"WHERE table_schema = {_literal(ns.schema.upper())} AND table_name = {_literal(table)}",
            ns,
        )
        for kind in df.iloc[:, 0]:
            keyword = "VIEW" if str(kind) == "VIEW" else "TABLE"
            self._execute(f"DROP {keyword} IF EXISTS {relation}")

    # -- agent side ------------------------------------------------------------------------
    def agent_config(self, ns: Namespace, base: dict[str, Any]) -> dict[str, Any]:
        config = dict(base)
        config.update({"database": ns.database, "schema": ns.schema})
        return config
