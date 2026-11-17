"""Build the credential-free local benchmark on disk.

The output uses the same directory layout as an ELT-Bench checkout (see
``elt_rl.tasks``), so the loader, environment and grader are exercised exactly
as they are for the official tasks. Everything is generated deterministically
from a seed: source data, the reference load, and the ground truth.

    python -m elt_rl.local_bench.build --root ~/.cache/elt-rl/local_bench
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from elt_rl.local_bench import oracle
from elt_rl.local_bench.specs import SPECS, Rows, SourceTable, TaskSpec
from elt_rl.minielt import cli as minielt

BENCH_VERSION = 1
DESTINATION = "duckdb"
DEFAULT_ROOT = Path.home() / ".cache" / "elt-rl" / "local_bench"

MINIELT_DOC = """\
# minielt: extract and load

`minielt` moves data from source systems into the destination warehouse. You
describe the pipeline in one YAML file, apply it, then trigger sync jobs.

## 1. Write the pipeline file

Create `elt/pipeline.yaml` with three sections.

```yaml
destination:
  type: duckdb
  path: <path of the DuckDB database file>
  schema: <schema to load into>

sources:
  - name: <any unique name>
    type: <source type>
    path: <path of the source>
    # plus type-specific fields, see below

connections:
  - name: <any unique name>
    source: <name of a source>
    streams:
      - name: <stream name>
        sync_mode: <sync mode>
```

Each stream is loaded into the table `<schema>.<stream name>`. Relative paths
are resolved against the directory you run `minielt` from.

### Source types

| type | `path` points to | streams |
| --- | --- | --- |
| `sqlite` | a SQLite database file | one per table, named after the table |
| `documents` | a directory of `<collection>.jsonl` files | one per collection |
| `object_store` | a bucket directory of `<object>.csv` files | one per object, named after the file without `.csv` |
| `file` | a single CSV or JSON-lines file | exactly one stream |

A `file` source needs two more fields: `format` (`csv` or `jsonl`) and
`dataset_name`, which is the name of its single stream and therefore of the
table it is loaded into.

### Sync modes

- `full_refresh_append`: every sync appends all source records to the table.
  Running the same sync twice loads the records twice.
- `full_refresh_overwrite`: every sync replaces the table.

## 2. Apply

```bash
minielt apply -f elt/pipeline.yaml
```

Validates the file and registers the connections. It prints one
`connection_id` per connection and writes them to `elt/minielt.state.json`.
Nothing is loaded yet.

## 3. Trigger sync jobs

```bash
minielt sync <connection_id> [<connection_id> ...]
minielt sync --all
```

## 4. Check job status

```bash
minielt status
```

Lists every connection with its jobs. A job is `succeeded` or `failed`; a
failed job shows the error. Fix the pipeline file, apply again, and re-run
the failed connection.
"""

DUCKDB_DOC = """\
# DuckDB destination

The warehouse is a single DuckDB database file. Its path and the schema to
use are in `config.yaml` under `duckdb.config`.

## dbt

Use the `duckdb` adapter (`dbt-duckdb` is installed). A minimal profile:

```yaml
my_project:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: <duckdb.config.path>
      schema: <duckdb.config.schema>
      threads: 1
```

Put `profiles.yml` in the project directory or in `~/.dbt/`.

## Inspecting the warehouse

```bash
python - <<'EOF'
import duckdb
con = duckdb.connect("<duckdb.config.path>", read_only=True)
print(con.sql("select table_schema, table_name from information_schema.tables").fetchall())
EOF
```

Only one process can write to the database file at a time. Close connections
(or let scripts exit) before running `minielt` or `dbt`.
"""


def reference_db_path(root: Path, task_name: str) -> Path:
    # The file stem becomes DuckDB's catalog name; it must differ from the
    # schema name (the task name) or two-part names become ambiguous.
    return root / "reference" / f"{task_name}__reference.duckdb"


def _write_csv(path: Path, rows: Rows, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row[c] is None else row[c]) for c in columns})


def _write_jsonl(path: Path, rows: Rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _sqlite_type(value: Any) -> str:
    if isinstance(value, (bool, int)):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _write_sqlite(path: Path, tables: dict[str, tuple[SourceTable, Rows]]) -> None:
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    try:
        for name, (table, rows) in tables.items():
            columns = [c.name for c in table.columns]
            types = []
            for column in columns:
                sample = next((r[column] for r in rows if r[column] is not None), "")
                types.append(_sqlite_type(sample))
            ddl = ", ".join(f'"{c}" {t}' for c, t in zip(columns, types, strict=True))
            con.execute(f'CREATE TABLE "{name}" ({ddl})')
            placeholders = ", ".join("?" for _ in columns)
            con.executemany(
                f'INSERT INTO "{name}" VALUES ({placeholders})',
                [[row[c] for c in columns] for row in rows],
            )
        con.commit()
    finally:
        con.close()


def write_sources(spec: TaskSpec, data: dict[str, Rows], sources_dir: Path) -> None:
    """Materialize the task's source systems."""
    if sources_dir.exists():
        shutil.rmtree(sources_dir)
    for sub in (oracle.DOCUMENTS_DIR, oracle.BUCKET_DIR, oracle.FILES_DIR):
        (sources_dir / sub).mkdir(parents=True)
    sqlite_tables: dict[str, tuple[SourceTable, Rows]] = {}
    for table in spec.tables:
        rows = data[table.name]
        columns = [c.name for c in table.columns]
        if table.kind == "sqlite":
            sqlite_tables[table.name] = (table, rows)
        elif table.kind == "documents":
            _write_jsonl(sources_dir / oracle.DOCUMENTS_DIR / f"{table.name}.jsonl", rows)
        elif table.kind == "object_store":
            _write_csv(sources_dir / oracle.BUCKET_DIR / f"{table.name}.csv", rows, columns)
        elif table.fmt == "csv":
            _write_csv(sources_dir / oracle.FILES_DIR / f"{table.name}.csv", rows, columns)
        else:
            _write_jsonl(sources_dir / oracle.FILES_DIR / f"{table.name}.jsonl", rows)
    if sqlite_tables:
        _write_sqlite(sources_dir / oracle.SQLITE_FILE, sqlite_tables)
    for sub in (oracle.DOCUMENTS_DIR, oracle.BUCKET_DIR, oracle.FILES_DIR):
        if not any((sources_dir / sub).iterdir()):
            (sources_dir / sub).rmdir()


def agent_config(spec: TaskSpec, task_name: str) -> dict[str, Any]:
    """The ``config.yaml`` shipped to the agent (paths relative to the workspace)."""
    config: dict[str, Any] = {"minielt": {"config": {"pipeline_file": "elt/pipeline.yaml"}}}
    sync_mode = "full_refresh_append"
    by_kind: dict[str, list[SourceTable]] = {}
    for table in spec.tables:
        by_kind.setdefault(table.kind, []).append(table)
    if "sqlite" in by_kind:
        config["relational_db"] = {
            "config": {
                "engine": "sqlite",
                "path": f"sources/{oracle.SQLITE_FILE}",
                "sync_mode": sync_mode,
                "tables": [t.name for t in by_kind["sqlite"]],
            }
        }
    if "documents" in by_kind:
        config["document_store"] = {
            "config": {
                "path": f"sources/{oracle.DOCUMENTS_DIR}",
                "sync_mode": sync_mode,
                "tables": [t.name for t in by_kind["documents"]],
            }
        }
    if "object_store" in by_kind:
        config["object_store"] = {
            "bucket": f"sources/{oracle.BUCKET_DIR}",
            "data": [
                {
                    "path": f"sources/{oracle.BUCKET_DIR}/{t.name}.csv",
                    "sync_mode": sync_mode,
                    "table": t.name,
                }
                for t in by_kind["object_store"]
            ],
        }
    if "file" in by_kind:
        config["flat_files"] = [
            {
                "format": t.fmt,
                "path": f"sources/{oracle.FILES_DIR}/{t.name}.{t.fmt}",
                "sync_mode": sync_mode,
                "table": t.name,
            }
            for t in by_kind["file"]
        ]
    # Path and schema are filled in per rollout by the destination adapter.
    config[DESTINATION] = {"config": {"path": "<set per rollout>", "schema": task_name}}
    return config


def data_model_yaml(spec: TaskSpec) -> str:
    return yaml.safe_dump(
        {
            "models": [
                {
                    "name": m.name,
                    "description": m.description,
                    "columns": [{"name": c.name, "description": c.description} for c in m.columns],
                }
                for m in spec.models
            ]
        },
        sort_keys=False,
        width=1000,
    )


def build_task(root: Path, spec: TaskSpec, task_name: str, seed: int) -> dict[str, Any]:
    """Generate one task under ``root`` and return its evaluation metadata."""
    import duckdb

    data = spec.generate(random.Random(f"{spec.name}:{seed}"))
    sources_dir = root / "sources" / task_name
    write_sources(spec, data, sources_dir)

    # Agent-visible bundle.
    bundle = root / f"inputs_{DESTINATION}" / task_name
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "schemas").mkdir(parents=True)
    (bundle / "documentation").mkdir()
    (bundle / "elt").mkdir()
    (bundle / "config.yaml").write_text(
        yaml.safe_dump(agent_config(spec, task_name), sort_keys=False), encoding="utf-8"
    )
    (bundle / "data_model.yaml").write_text(data_model_yaml(spec), encoding="utf-8")
    for table in spec.tables:
        with (bundle / "schemas" / f"{table.name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["column_name", "column_description"])
            writer.writerows([c.name, c.description] for c in table.columns)
    (bundle / "documentation" / "minielt.md").write_text(MINIELT_DOC, encoding="utf-8")
    (bundle / "documentation" / f"destination_{DESTINATION}.md").write_text(
        DUCKDB_DOC, encoding="utf-8"
    )

    # Reference load: the reference pipeline run through the same loader the
    # agent uses. It is both the stage-1 content reference and the snapshot a
    # transformation-only rollout starts from.
    reference = reference_db_path(root, task_name)
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.unlink(missing_ok=True)
    state = minielt.validate(
        oracle.pipeline_config(spec, str(sources_dir), str(reference), task_name), sources_dir
    )
    for connection in state["connections"]:
        job = minielt.run_connection(state, connection)
        if job["status"] != "succeeded":
            raise RuntimeError(f"reference load failed for {task_name}: {job['error']}")

    # Ground truth: reference transformations over the reference load.
    gt_dir = root / "ground_truth" / f"gt_{DESTINATION}" / task_name
    if gt_dir.exists():
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True)
    sql_dir = root / "evaluation" / "sql" / task_name
    if sql_dir.exists():
        shutil.rmtree(sql_dir)
    sql_dir.mkdir(parents=True)
    con = duckdb.connect(str(reference), read_only=True)
    try:
        for model in spec.models:
            df = con.execute(oracle.reference_sql(spec, model.name, task_name)).fetchdf()
            df.to_csv(gt_dir / f"{model.name}.csv", index=False)
            (sql_dir / f"{model.name}.sql").write_text(
                f"select * from {task_name}.{model.name} order by {', '.join(model.key)}",
                encoding="utf-8",
            )
    finally:
        con.close()

    return {
        "tables": {t.name: len(data[t.name]) for t in spec.tables},
        "sort_keys": {m.name: list(m.key) for m in spec.models},
    }


def build_local_bench(
    root: str | Path = DEFAULT_ROOT,
    seed: int = 0,
    variants: int = 1,
    only: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Build (or reuse) the local benchmark and return its root.

    Args:
        root: Output directory.
        seed: Seed for the generated source data.
        variants: Data variants per task specification. Variant ``k > 1`` is a
            separate task named ``<spec>_v<k>`` with different data, which
            gives more distinct training tasks at no authoring cost.
        only: Restrict to these task specifications.
        force: Rebuild even if a matching benchmark already exists.
    """
    root = Path(root).expanduser().resolve()
    names = sorted(only) if only else sorted(SPECS)
    manifest = {"version": BENCH_VERSION, "seed": seed, "variants": variants, "specs": names}
    manifest_path = root / "manifest.json"
    if (
        not force
        and manifest_path.is_file()
        and json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    ):
        return root
    root.mkdir(parents=True, exist_ok=True)
    for stale in ("evaluation", "ground_truth", f"inputs_{DESTINATION}", "reference", "sources"):
        shutil.rmtree(root / stale, ignore_errors=True)
    manifest_path.unlink(missing_ok=True)
    (root / "evaluation").mkdir()

    table_json: dict[str, Any] = {}
    sort_key_json: dict[str, Any] = {}
    for name in names:
        spec = SPECS[name]
        for variant in range(1, variants + 1):
            task_name = name if variant == 1 else f"{name}_v{variant}"
            meta = build_task(root, spec, task_name, seed + variant - 1)
            table_json[task_name] = meta["tables"]
            sort_key_json[task_name] = meta["sort_keys"]
    (root / "evaluation" / "table.json").write_text(json.dumps(table_json, indent=1), encoding="utf-8")
    (root / "evaluation" / "sort_key.json").write_text(
        json.dumps(sort_key_json, indent=1), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return root


def spec_for_task(task_name: str) -> TaskSpec:
    """Specification behind a local task name (``bookshop_v2`` -> ``bookshop``)."""
    base = task_name.rsplit("_v", 1)[0] if "_v" in task_name else task_name
    return SPECS[base if base in SPECS else task_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the credential-free local benchmark.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = build_local_bench(args.root, args.seed, args.variants, force=args.force)
    print(f"Local benchmark ready at {root}")


if __name__ == "__main__":
    main()
