#!/usr/bin/env python3
"""minielt: a small declarative extract-and-load tool for the local benchmark.

It stands in for Airbyte + Terraform in the credential-free tasks and keeps
the same shape of work: read the docs, declare sources, a destination and
connections in a file, apply it, trigger sync jobs, check job status.

    minielt apply -f elt/pipeline.yaml     validate and register connections
    minielt sync --all | <connection_id>   run sync jobs
    minielt status                         show connections and job results

Like Airbyte, loaded tables carry ``_airbyte_raw_id`` / ``_airbyte_extracted_at``
columns, and an *append* sync run twice loads the rows twice.

This file is intentionally self-contained (stdlib + duckdb + pandas + pyyaml)
so it can be copied into a sandbox image on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_TYPES = ("sqlite", "documents", "object_store", "file")
SYNC_MODES = ("full_refresh_append", "full_refresh_overwrite")
STATE_FILE = "minielt.state.json"


class ConfigError(Exception):
    """The pipeline file is invalid; the message says what to fix."""


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or mapping.get(key) in (None, ""):
        raise ConfigError(f"{where}: missing required field '{key}'")
    return mapping[key]


def _quote(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def load_pipeline(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise ConfigError(f"pipeline file not found: {path}")
    try:
        pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(pipeline, dict):
        raise ConfigError(f"{path} must contain a mapping with destination/sources/connections")
    return pipeline


def validate(pipeline: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Check the pipeline file and return its normalized form."""
    destination = _require(pipeline, "destination", "pipeline")
    if _require(destination, "type", "destination") != "duckdb":
        raise ConfigError("destination.type must be 'duckdb'")
    dest = {
        "type": "duckdb",
        "path": str(_require(destination, "path", "destination")),
        "schema": str(_require(destination, "schema", "destination")),
    }

    sources: dict[str, dict[str, Any]] = {}
    for i, source in enumerate(_require(pipeline, "sources", "pipeline")):
        where = f"sources[{i}]"
        name = str(_require(source, "name", where))
        kind = _require(source, "type", where)
        if kind not in SOURCE_TYPES:
            raise ConfigError(f"{where}: unknown type {kind!r}; expected one of {SOURCE_TYPES}")
        if name in sources:
            raise ConfigError(f"{where}: duplicate source name {name!r}")
        path = Path(str(_require(source, "path", where)))
        path = path if path.is_absolute() else base_dir / path
        if not path.exists():
            raise ConfigError(f"{where}: path does not exist: {path}")
        normalized = {"name": name, "type": kind, "path": str(path)}
        if kind == "file":
            fmt = _require(source, "format", where)
            if fmt not in ("csv", "jsonl"):
                raise ConfigError(f"{where}: format must be 'csv' or 'jsonl'")
            normalized["format"] = fmt
            normalized["dataset_name"] = str(_require(source, "dataset_name", where))
            if not path.is_file():
                raise ConfigError(f"{where}: a file source needs the path of a file: {path}")
        elif kind == "sqlite" and not path.is_file():
            raise ConfigError(f"{where}: sqlite path must be a database file: {path}")
        elif kind in ("documents", "object_store") and not path.is_dir():
            raise ConfigError(f"{where}: {kind} path must be a directory: {path}")
        sources[name] = normalized

    connections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, connection in enumerate(_require(pipeline, "connections", "pipeline")):
        where = f"connections[{i}]"
        name = str(_require(connection, "name", where))
        if name in seen:
            raise ConfigError(f"{where}: duplicate connection name {name!r}")
        seen.add(name)
        source_name = str(_require(connection, "source", where))
        if source_name not in sources:
            raise ConfigError(f"{where}: unknown source {source_name!r}")
        streams = []
        for j, stream in enumerate(_require(connection, "streams", where)):
            swhere = f"{where}.streams[{j}]"
            stream_name = str(_require(stream, "name", swhere))
            sync_mode = _require(stream, "sync_mode", swhere)
            if sync_mode not in SYNC_MODES:
                raise ConfigError(f"{swhere}: sync_mode must be one of {SYNC_MODES}")
            streams.append({"name": stream_name, "sync_mode": sync_mode})
        if not streams:
            raise ConfigError(f"{where}: at least one stream is required")
        digest = hashlib.sha1(f"{name}:{source_name}".encode()).hexdigest()
        connections.append(
            {
                "connection_id": f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}",
                "name": name,
                "source": source_name,
                "streams": streams,
            }
        )
    return {"destination": dest, "sources": sources, "connections": connections, "jobs": []}


# -- extraction ---------------------------------------------------------------------
def _available_streams(source: dict[str, Any]) -> list[str]:
    path = Path(source["path"])
    if source["type"] == "sqlite":
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            rows = con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return sorted(r[0] for r in rows)
    if source["type"] == "documents":
        return sorted(p.stem for p in path.glob("*.jsonl"))
    if source["type"] == "object_store":
        return sorted(p.stem for p in path.glob("*.csv"))
    return [source["dataset_name"]]


def _extract(con: Any, source: dict[str, Any], stream: str) -> Any:
    """Return a DuckDB relation or DataFrame holding the stream's records."""
    available = _available_streams(source)
    if stream not in available:
        raise RuntimeError(
            f"stream {stream!r} not found in source {source['name']!r}; available: {available}"
        )
    path = Path(source["path"])
    if source["type"] == "sqlite":
        import pandas as pd

        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as sq:
            return pd.read_sql_query(f"SELECT * FROM {_quote(stream)}", sq)
    if source["type"] == "documents":
        return con.read_json(str(path / f"{stream}.jsonl"), format="newline_delimited")
    if source["type"] == "object_store":
        return con.read_csv(str(path / f"{stream}.csv"), header=True)
    if source["format"] == "csv":
        return con.read_csv(str(path), header=True)
    return con.read_json(str(path), format="newline_delimited")


def _load(con: Any, schema: str, stream: str, records: Any, sync_mode: str) -> int:
    view = "_minielt_records"
    is_relation = hasattr(records, "create_view")
    if is_relation:
        records.create_view(view, replace=True)
    else:
        con.register(view, records)
    try:
        select = (
            "SELECT *, cast(uuid() AS VARCHAR) AS _airbyte_raw_id, "
            f"current_timestamp AS _airbyte_extracted_at FROM {view}"
        )
        target = f"{_quote(schema)}.{_quote(stream)}"
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(schema)}")
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, stream],
        ).fetchall()[0][0]
        if sync_mode == "full_refresh_overwrite" or not exists:
            con.execute(f"CREATE OR REPLACE TABLE {target} AS {select}")
        else:
            con.execute(f"INSERT INTO {target} BY NAME {select}")
        return int(con.execute(f"SELECT count(*) FROM {view}").fetchall()[0][0])
    finally:
        if is_relation:
            con.execute(f"DROP VIEW IF EXISTS {view}")
        else:
            con.unregister(view)


def run_connection(state: dict[str, Any], connection: dict[str, Any]) -> dict[str, Any]:
    import duckdb

    source = state["sources"][connection["source"]]
    destination = state["destination"]
    job = {
        "job_id": len(state["jobs"]) + 1,
        "connection_id": connection["connection_id"],
        "connection": connection["name"],
        "status": "running",
        "records": {},
        "error": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        Path(destination["path"]).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(destination["path"])
        try:
            for stream in connection["streams"]:
                records = _extract(con, source, stream["name"])
                job["records"][stream["name"]] = _load(
                    con, destination["schema"], stream["name"], records, stream["sync_mode"]
                )
        finally:
            con.close()
        job["status"] = "succeeded"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
    state["jobs"].append(job)
    return job


# -- commands -------------------------------------------------------------------------
def _state_path(state_dir: Path) -> Path:
    return state_dir / STATE_FILE


def _read_state(state_dir: Path) -> dict[str, Any]:
    path = _state_path(state_dir)
    if not path.is_file():
        raise ConfigError(f"no applied pipeline found ({path}); run `minielt apply -f <file>` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _state_path(state_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_apply(args: argparse.Namespace) -> int:
    pipeline_path = Path(args.file).resolve()
    state = validate(load_pipeline(pipeline_path), Path.cwd())
    state_dir = Path(args.state_dir) if args.state_dir else pipeline_path.parent
    _write_state(state_dir, state)
    print(f"Applied {pipeline_path}: {len(state['connections'])} connection(s) registered.")
    for connection in state["connections"]:
        streams = ", ".join(s["name"] for s in connection["streams"])
        print(f"  connection_id={connection['connection_id']} name={connection['name']} streams=[{streams}]")
    print(f"State written to {_state_path(state_dir)}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    state = _read_state(state_dir)
    if args.all:
        selected = state["connections"]
    else:
        selected = [c for c in state["connections"] if c["connection_id"] in args.connection_ids]
        unknown = set(args.connection_ids) - {c["connection_id"] for c in selected}
        if unknown:
            raise ConfigError(f"unknown connection id(s): {sorted(unknown)}")
    if not selected:
        raise ConfigError("nothing to sync: pass connection ids or --all")
    failed = 0
    for connection in selected:
        job = run_connection(state, connection)
        _write_state(state_dir, state)
        line = f"job {job['job_id']} connection={job['connection']} status={job['status']}"
        if job["status"] == "succeeded":
            line += " records=" + json.dumps(job["records"])
        else:
            line += f" error={job['error']}"
            failed += 1
        print(line)
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    state = _read_state(Path(args.state_dir))
    print(f"destination: {state['destination']['path']} schema={state['destination']['schema']}")
    for connection in state["connections"]:
        jobs = [j for j in state["jobs"] if j["connection_id"] == connection["connection_id"]]
        last = jobs[-1]["status"] if jobs else "never run"
        print(
            f"connection_id={connection['connection_id']} name={connection['name']} "
            f"jobs={len(jobs)} last_status={last}"
        )
        for job in jobs:
            detail = json.dumps(job["records"]) if job["status"] == "succeeded" else job["error"]
            print(f"    job {job['job_id']}: {job['status']} {detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minielt", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    apply = sub.add_parser("apply", help="validate a pipeline file and register its connections")
    apply.add_argument("-f", "--file", required=True, help="pipeline YAML file")
    apply.add_argument("--state-dir", help="where to keep state (default: next to the file)")
    apply.set_defaults(func=cmd_apply)

    sync = sub.add_parser("sync", help="run sync jobs")
    sync.add_argument("connection_ids", nargs="*", help="connection ids to sync")
    sync.add_argument("--all", action="store_true", help="sync every connection")
    sync.add_argument("--state-dir", default="elt", help="state directory (default: elt)")
    sync.set_defaults(func=cmd_sync)

    status = sub.add_parser("status", help="show connections and their jobs")
    status.add_argument("--state-dir", default="elt", help="state directory (default: elt)")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as e:
        print(f"minielt: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
