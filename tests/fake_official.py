"""Stand-ins for the parts of the official stack that need credentials.

* ``FakeSnowflake`` answers exactly the statements ``SnowflakeDestination``
  issues (and fails on anything else), on in-memory tables. Unquoted
  identifiers are upper case, as in Snowflake.
* ``FakeAgentMachine`` replaces the Docker container: it "runs" ``terraform
  apply`` and ``dbt run`` by reading the workspace's ``config.yaml`` and
  writing the corresponding tables into the fake warehouse.
* ``write_official_tree`` lays out a one-task benchmark the way an ELT-Bench
  checkout looks after ``setup/write_config.py --destination snowflake``.

They let the official code path (task loader, workspace preparation,
namespace lifecycle SQL, grading through a second destination adapter,
teardown) run in a test. They are not a substitute for a real rollout.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

TASK = "shop"
CUSTOMERS = pd.DataFrame({"CUSTOMER_ID": [1, 2, 3], "NAME": ["Ann", "Bo", "Cy"]})
ORDERS = pd.DataFrame(
    {"ORDER_ID": [10, 11, 12, 13], "CUSTOMER_ID": [1, 1, 2, 3], "AMOUNT": [5.5, 4.5, 20.0, 7.25]}
)
SOURCES = {"CUSTOMERS": CUSTOMERS, "ORDERS": ORDERS}


def customer_orders(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    grouped = orders.groupby("CUSTOMER_ID").agg(
        ORDER_COUNT=("ORDER_ID", "count"), TOTAL_AMOUNT=("AMOUNT", "sum")
    )
    return customers.merge(grouped, on="CUSTOMER_ID")[
        ["CUSTOMER_ID", "NAME", "ORDER_COUNT", "TOTAL_AMOUNT"]
    ]


def with_airbyte_columns(frame: pd.DataFrame, sync: str) -> pd.DataFrame:
    out = frame.copy()
    out["_AIRBYTE_RAW_ID"] = [f"{sync}-{i}" for i in range(len(out))]
    out["_AIRBYTE_EXTRACTED_AT"] = sync
    return out


# -- warehouse ---------------------------------------------------------------------
_Q = r'"([^"]+)"'
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("create_database", re.compile(rf"CREATE DATABASE IF NOT EXISTS {_Q}$")),
    ("drop_schema", re.compile(rf"DROP SCHEMA IF EXISTS {_Q}\.{_Q} CASCADE$")),
    ("show_tables", re.compile(rf"SHOW TABLES IN DATABASE {_Q}$")),
    ("clone_schema", re.compile(rf"CREATE SCHEMA {_Q}\.{_Q} CLONE {_Q}\.{_Q}$")),
    ("create_schema", re.compile(rf"CREATE SCHEMA {_Q}\.{_Q}$")),
    ("grant", re.compile(r"GRANT .+ TO ROLE \"[A-Z_]+\"( COPY CURRENT GRANTS)?$")),
    (
        "list_tables",
        re.compile(
            rf"SELECT table_name FROM {_Q}\.information_schema\.tables "
            r"WHERE table_schema = '([^']+)' AND table_type IN \(([^)]+)\)$"
        ),
    ),
    (
        "list_columns",
        re.compile(
            rf"SELECT column_name FROM {_Q}\.information_schema\.columns "
            r"WHERE table_schema = '([^']+)' AND table_name = '([^']+)' ORDER BY ordinal_position$"
        ),
    ),
    (
        "table_type",
        re.compile(
            rf"SELECT table_type FROM {_Q}\.information_schema\.tables "
            r"WHERE table_schema = '([^']+)' AND table_name = '([^']+)'$"
        ),
    ),
    ("drop_table", re.compile(rf"DROP (?:TABLE|VIEW) IF EXISTS {_Q}\.{_Q}\.{_Q}$")),
    (
        "fingerprint",
        re.compile(
            rf"SELECT COUNT\(\*\) AS n, TO_VARCHAR\(HASH_AGG\((.+)\)\) AS h FROM {_Q}\.{_Q}\.{_Q}$"
        ),
    ),
    ("count", re.compile(rf"SELECT COUNT\(\*\) FROM {_Q}\.{_Q}\.{_Q}$")),
    ("select_all", re.compile(rf"SELECT \* FROM {_Q}\.{_Q}\.{_Q}$")),
]


class FakeSnowflake:
    def __init__(self) -> None:
        self.schemas: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
        self.statements: list[str] = []
        self.connect_kwargs: list[dict[str, Any]] = []

    # -- what the agent's tools would do (not through the grader's connection) -----
    def write_table(self, database: str, schema: str, table: str, frame: pd.DataFrame) -> None:
        key = (database.upper(), schema.upper())
        if key not in self.schemas:
            raise RuntimeError(f"Schema {database}.{schema} does not exist or not authorized")
        self.schemas[key][table] = frame.reset_index(drop=True)

    def _table(self, database: str, schema: str, table: str) -> pd.DataFrame:
        try:
            return self.schemas[(database, schema)][table]
        except KeyError:
            raise RuntimeError(
                f"SQL compilation error: Object '{database}.{schema}.{table}' does not exist"
            ) from None

    # -- connection ------------------------------------------------------------------
    def connect(self, **kwargs: Any) -> _Connection:
        self.connect_kwargs.append(kwargs)
        return _Connection(self)

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]:
        self.statements.append(sql)
        for kind, pattern in _PATTERNS:
            match = pattern.match(sql)
            if match:
                return getattr(self, f"_{kind}")(*match.groups())
        raise AssertionError(f"SnowflakeDestination issued an unexpected statement: {sql}")

    def _create_database(self, database: str) -> tuple[list[str], list[tuple]]:
        return [], []

    def _grant(self, *_: Any) -> tuple[list[str], list[tuple]]:
        return [], []

    def _drop_schema(self, database: str, schema: str) -> tuple[list[str], list[tuple]]:
        self.schemas.pop((database, schema), None)
        return [], []

    def _create_schema(self, database: str, schema: str) -> tuple[list[str], list[tuple]]:
        assert (database, schema) not in self.schemas
        self.schemas[(database, schema)] = {}
        return [], []

    def _clone_schema(
        self, database: str, schema: str, src_database: str, src_schema: str
    ) -> tuple[list[str], list[tuple]]:
        self.schemas[(database, schema)] = copy.deepcopy(self.schemas[(src_database, src_schema)])
        return [], []

    def _show_tables(self, database: str) -> tuple[list[str], list[tuple]]:
        rows = [
            ("2026-01-01", table, db, schema)
            for (db, schema), tables in self.schemas.items()
            if db == database
            for table in tables
        ]
        return ["created_on", "name", "database_name", "schema_name"], rows

    def _list_tables(self, database: str, schema: str, kinds: str) -> tuple[list[str], list[tuple]]:
        del kinds  # the fake has base tables only
        tables = self.schemas.get((database, schema), {})
        return ["TABLE_NAME"], [(t,) for t in tables]

    def _list_columns(self, database: str, schema: str, table: str) -> tuple[list[str], list[tuple]]:
        return ["COLUMN_NAME"], [(c,) for c in self._table(database, schema, table).columns]

    def _table_type(self, database: str, schema: str, table: str) -> tuple[list[str], list[tuple]]:
        exists = table in self.schemas.get((database, schema), {})
        return ["TABLE_TYPE"], [("BASE TABLE",)] if exists else []

    def _drop_table(self, database: str, schema: str, table: str) -> tuple[list[str], list[tuple]]:
        self.schemas.get((database, schema), {}).pop(table, None)
        return [], []

    def _count(self, database: str, schema: str, table: str) -> tuple[list[str], list[tuple]]:
        return ["COUNT(*)"], [(len(self._table(database, schema, table)),)]

    def _select_all(self, database: str, schema: str, table: str) -> tuple[list[str], list[tuple]]:
        frame = self._table(database, schema, table)
        return list(frame.columns), [tuple(r) for r in frame.itertuples(index=False, name=None)]

    def _fingerprint(
        self, expression: str, database: str, schema: str, table: str
    ) -> tuple[list[str], list[tuple]]:
        columns = re.findall(r'TO_VARCHAR\("([^"]+)"\)', expression)
        frame = self._table(database, schema, table)[columns]
        rows = Counter(tuple(str(v) for v in r) for r in frame.itertuples(index=False, name=None))
        return ["N", "H"], [(len(frame), str(hash(frozenset(rows.items()))))]


class _Connection:
    def __init__(self, warehouse: FakeSnowflake) -> None:
        self.warehouse = warehouse
        self.closed = False

    def cursor(self) -> _Cursor:
        return _Cursor(self.warehouse)

    def close(self) -> None:
        self.closed = True


class _Cursor:
    def __init__(self, warehouse: FakeSnowflake) -> None:
        self.warehouse = warehouse
        self.description: list[tuple] = []
        self._rows: list[tuple] = []

    def execute(self, sql: str) -> None:
        columns, self._rows = self.warehouse.execute(sql)
        self.description = [(c,) for c in columns]

    def fetchall(self) -> list[tuple]:
        return self._rows

    def close(self) -> None:
        pass


# -- the agent's machine -----------------------------------------------------------------
class FakeAgentMachine:
    """Replaces the rollout container. Paths under /workspace map to the host workspace."""

    def __init__(
        self,
        warehouse: FakeSnowflake,
        host_workspace: Path,
        create_kwargs: dict[str, Any],
        corrupt_load: bool = False,
    ) -> None:
        self.warehouse = warehouse
        self.host_workspace = Path(host_workspace)
        self.create_kwargs = create_kwargs
        self.corrupt_load = corrupt_load
        self.commands: list[tuple[str, str | None]] = []
        self.cleaned_up = False

    def _host(self, path: str) -> Path:
        assert path.startswith("/workspace"), path
        return self.host_workspace / path.removeprefix("/workspace").lstrip("/")

    def _target(self) -> tuple[str, str]:
        config = yaml.safe_load((self.host_workspace / "config.yaml").read_text())
        block = config["snowflake"]["config"]
        return block["database"], block["schema"]

    async def run_command(
        self, command: str, workdir: str | None = None, timeout: int = 60, **_: Any
    ) -> SandboxResult:
        self.commands.append((command, workdir))
        stdout = ""
        if "dbt_project.yml" in command and command.startswith("find"):
            found = sorted(self.host_workspace.glob("elt/*/dbt_project.yml"))
            stdout = "\n".join(
                f"/workspace/{p.relative_to(self.host_workspace)}" for p in found[:1]
            )
        elif "terraform apply" in command:
            database, schema = self._target()
            for table, frame in SOURCES.items():
                loaded = with_airbyte_columns(frame, sync=schema)
                if self.corrupt_load and table == "ORDERS":
                    loaded.loc[0, "AMOUNT"] = 999.0  # right row count, wrong data
                self.warehouse.write_table(database, schema, table, loaded)
            stdout = "Apply complete! Resources: 6 added, 0 changed, 0 destroyed."
        elif command.strip() == "dbt run" or command.strip().endswith("&& dbt run"):
            database, schema = self._target()
            tables = self.warehouse.schemas[(database.upper(), schema.upper())]
            if "CUSTOMERS" not in tables or "ORDERS" not in tables:
                return SandboxResult(stdout="Database Error: object does not exist", stderr="", exit_code=1)
            self.warehouse.write_table(
                database, schema, "CUSTOMER_ORDERS", customer_orders(tables["CUSTOMERS"], tables["ORDERS"])
            )
            stdout = "Completed successfully\nDone. PASS=1 WARN=0 ERROR=0 SKIP=0 TOTAL=1"
        return SandboxResult(stdout=stdout, stderr="", exit_code=0)

    async def write_file(self, path: str, content: str | bytes, **_: Any) -> SandboxResult:
        target = self._host(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        target.write_bytes(data)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def read_file(self, path: str, **_: Any) -> SandboxResult:
        return SandboxResult(stdout=self._host(path).read_text(), stderr="", exit_code=0)

    async def cleanup(self) -> None:
        self.cleaned_up = True


# -- benchmark tree -----------------------------------------------------------------------
def write_official_tree(root: Path) -> Path:
    """One task, laid out like an ELT-Bench checkout prepared for Snowflake."""
    bundle = root / "inputs" / TASK
    (bundle / "schemas").mkdir(parents=True)
    (bundle / "documentation").mkdir()
    (bundle / "elt").mkdir()
    config = {
        "Airbyte": {"config": {"server_url": "http://airbyte/api/public/v1/", "username": "u",
                               "password": "airbyte-secret", "workspace_id": "w"}},
        "postgres": {"config": {"host": "elt-postgres", "database": TASK, "tables": ["customers", "orders"],
                                "sync_mode": "full_refresh_append"}},
        "snowflake": {"config": {"account": "acme-xy12345", "database": TASK.upper(), "schema": "AIRBYTE_SCHEMA",
                                 "role": "AIRBYTE_ROLE", "warehouse": "AIRBYTE_WAREHOUSE",
                                 "username": "AIRBYTE_USER"}},
    }  # fmt: skip
    (bundle / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # Like some official files, not strictly valid YAML: the loader must not need it.
    (bundle / "data_model.yaml").write_text(
        "models:\n  - name: customer_orders\n    description: Orders per customer: count, total\n"
    )
    (bundle / "schemas" / "customers.csv").write_text("column_name,type\ncustomer_id,int\nname,text\n")
    (bundle / "schemas" / "orders.csv").write_text(
        "column_name,type\norder_id,int\ncustomer_id,int\namount,float\n"
    )
    (bundle / "documentation" / "airbyte_Provider.md").write_text("# Airbyte provider\n")
    (bundle / "documentation" / "destination_snowflake.md").write_text("# Snowflake destination\n")
    (bundle / "documentation" / "source_postgres.md").write_text("# Postgres source\n")
    (bundle / "elt" / "main.tf").write_text('terraform { required_providers { airbyte = {} } }\n')
    (bundle / "check_job_status.py").write_text("print('succeeded')\n")
    (bundle / "snowflake_credential.json").write_text(
        json.dumps({"account": "acme-xy12345", "user": "ADMIN", "password": "admin-secret"})
    )

    evaluation = root / "evaluation"
    (evaluation / "sql" / TASK).mkdir(parents=True)
    # Metadata keys mix cases in the official files.
    (evaluation / "table.json").write_text(json.dumps({"Shop": {"customers": 3, "orders": 4}}))
    (evaluation / "sort_key.json").write_text(json.dumps({TASK: {"customer_orders": ["customer_id"]}}))
    (evaluation / "sql" / TASK / "customer_orders.sql").write_text(
        "select * from shop.customer_orders order by customer_id\n"
    )
    gt = root / "ground_truth" / "gt_snowflake" / TASK
    gt.mkdir(parents=True)
    gold = customer_orders(CUSTOMERS, ORDERS)
    gold.columns = [c.lower() for c in gold.columns]
    gold.to_csv(gt / "customer_orders.csv", index=False)

    admin = root / "setup" / "destination"
    admin.mkdir(parents=True)
    (admin / "snowflake_credential.json").write_text(
        json.dumps({"account": "acme-xy12345", "user": "ADMIN", "password": "admin-secret"})
    )
    return root
