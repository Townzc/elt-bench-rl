"""Destination adapters: the registry, DuckDB, and adding a new warehouse."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from elt_rl.destinations import (
    Destination,
    DuckDBDestination,
    Namespace,
    destination_names,
    get_destination,
    register_destination,
)
from elt_rl.destinations.snowflake_dest import SnowflakeDestination

from .fake_official import FakeSnowflake


def test_registry() -> None:
    assert {"duckdb", "snowflake"} <= set(destination_names())
    assert isinstance(get_destination("duckdb"), DuckDBDestination)
    with pytest.raises(ValueError, match="registered"):
        get_destination("oracle9i")

    class Mirror(DuckDBDestination):
        name = "mirror"

    register_destination("mirror", Mirror)
    assert isinstance(get_destination("mirror"), Mirror)
    # An adapter is complete once it implements the abstract interface.
    assert not getattr(Mirror, "__abstractmethods__", None)
    with pytest.raises(TypeError):
        Destination()  # type: ignore[abstract]


def test_duckdb_namespace_lifecycle(tmp_path: Path) -> None:
    destination = DuckDBDestination()
    ns = destination.new_namespace("shop", "abc", str(tmp_path / "host"), "/workspace")
    assert ns.schema == "shop" and ns.database.startswith(str(tmp_path / "host"))
    assert ns.for_agent == "/workspace/warehouse/warehouse.duckdb"
    assert destination.list_tables(ns) == []  # not created yet

    destination.create_namespace(ns)
    con = duckdb.connect(ns.database)
    con.execute('create table shop."Orders" as select 1 as "Id", 2.5 as amount')
    con.execute("create view shop.v as select * from shop.\"Orders\"")
    con.close()
    assert destination.list_tables(ns) == ["Orders"]
    assert sorted(destination.list_tables(ns, include_views=True)) == ["Orders", "v"]
    assert destination.resolve_table(ns, "orders") == "Orders"  # case-insensitive
    assert destination.resolve_table(ns, "v") is None
    assert destination.table_columns(ns, "Orders") == ["Id", "amount"]
    assert destination.table_size(ns, "Orders") == 1
    rows, digest = destination.table_fingerprint(ns, "Orders", ["Id", "amount"])
    assert rows == 1 and digest

    clone = destination.new_namespace("shop", "def", str(tmp_path / "other"), str(tmp_path / "other"))
    destination.create_namespace(clone, clone_from=ns)
    assert destination.table_fingerprint(clone, "Orders", ["Id", "amount"]) == (rows, digest)
    destination.drop_table(clone, "Orders")
    destination.drop_table(clone, "v")
    assert destination.list_tables(clone, include_views=True) == []
    assert destination.list_tables(ns) == ["Orders"]  # the original is untouched

    config = destination.agent_config(ns, {"path": "placeholder", "threads": 1})
    assert config == {"path": ns.for_agent, "threads": 1, "schema": "shop"}
    destination.drop_namespace(ns)
    destination.drop_namespace(ns)  # idempotent
    assert not Path(ns.database).exists()


def test_duckdb_fingerprint_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    destination = DuckDBDestination()

    def load(name: str, select: str) -> Namespace:
        ns = Namespace(database=str(tmp_path / f"{name}.duckdb"), schema="s")
        con = duckdb.connect(ns.database)
        con.execute("create schema s")
        con.execute(f"create table s.t as {select}")
        con.close()
        return ns

    def fingerprint(ns: Namespace) -> tuple[int, str]:
        return destination.table_fingerprint(ns, "t", ["a", "b"])

    base = fingerprint(load("a", "select i as a, i % 3 as b from range(100) t(i) order by i"))
    assert fingerprint(load("b", "select i as a, i % 3 as b from range(100) t(i) order by i desc")) == base
    assert fingerprint(load("c", "select i as a, i % 4 as b from range(100) t(i)")) != base
    # Same multiset of values, paired differently.
    assert fingerprint(load("d", "select i as a, (i + 1) % 3 as b from range(100) t(i)")) != base
    # A duplicated row changes the count and the hash.
    doubled = "select i % 50 as a, (i % 50) % 3 as b from range(100) t(i)"
    assert fingerprint(load("e", doubled)) != base


def test_eval_sql_is_rewritten_to_the_rollout_namespace(tmp_path: Path) -> None:
    destination = DuckDBDestination()
    ns = Namespace(database=str(tmp_path / "w.duckdb"), schema="rollout_schema")
    con = duckdb.connect(ns.database)
    con.execute("create schema rollout_schema")
    con.execute("create table rollout_schema.Sales as select 1 as id, 10 as amount")
    con.execute("create table rollout_schema.dim as select 1 as id, 'x' as label")
    con.close()
    df = destination.query_eval_sql(
        "select s.id, d.label from shop.sales s join shop.dim d on d.id = s.id order by s.id", ns
    )
    assert df.to_dict("records") == [{"id": 1, "label": "x"}]


def test_snowflake_adapter_statements() -> None:
    """Identifier handling of the Snowflake adapter, against the recording fake."""
    account = FakeSnowflake()
    destination = SnowflakeDestination(
        credential={"account": "a", "user": "u", "password": "p"},
        database="elt_rl",
        agent_role="airbyte_role",
        connect=account.connect,
    )
    ns = destination.new_namespace("books", "0a1b2c", "/host/ws", "/workspace")
    assert (ns.database, ns.schema) == ("ELT_RL", "BOOKS__0A1B2C")
    with pytest.raises(ValueError):
        destination.new_namespace('x"; drop database prod; --', "id", "", "")
    with pytest.raises(ValueError):
        SnowflakeDestination(credential={}, database="bad name")

    # Leftover Airbyte raw tables of this schema are removed with it; others are kept.
    account.schemas[("ELT_RL", "AIRBYTE_INTERNAL")] = {
        "BOOKS__0A1B2C_RAW__STREAM_AUTHOR": None,  # type: ignore[dict-item]
        "BOOKS__FFFFFF_RAW__STREAM_AUTHOR": None,  # type: ignore[dict-item]
    }
    destination.create_namespace(ns)
    assert list(account.schemas[("ELT_RL", "AIRBYTE_INTERNAL")]) == ["BOOKS__FFFFFF_RAW__STREAM_AUTHOR"]
    assert account.statements[:2] == [
        'CREATE DATABASE IF NOT EXISTS "ELT_RL"',
        'DROP SCHEMA IF EXISTS "ELT_RL"."BOOKS__0A1B2C" CASCADE',
    ]
    assert 'GRANT USAGE, CREATE SCHEMA ON DATABASE "ELT_RL" TO ROLE "AIRBYTE_ROLE"' in account.statements

    assert destination.qualified(ns, "Author") == '"ELT_RL"."BOOKS__0A1B2C"."Author"'
    assert destination.norm("author") == "AUTHOR"
    assert destination.fingerprint_sql("R", ["A", 'b"c']) == (
        'SELECT COUNT(*) AS n, TO_VARCHAR(HASH_AGG(TO_VARCHAR("A"), TO_VARCHAR("b""c"))) AS h FROM R'
    )
    agent = destination.agent_config(ns, {"account": "a", "database": "BOOKS", "schema": "AIRBYTE_SCHEMA"})
    assert agent == {"account": "a", "database": "ELT_RL", "schema": "BOOKS__0A1B2C"}
    destination.drop_namespace(ns)
    assert ("ELT_RL", "BOOKS__0A1B2C") not in account.schemas
    destination.close()
