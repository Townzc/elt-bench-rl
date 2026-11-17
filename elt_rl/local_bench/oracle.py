"""Reference solutions for the local benchmark.

Used to (a) build the reference load and ground truth, and (b) drive scripted
rollouts in tests and smoke runs. Nothing here is ever copied into an agent's
workspace.
"""

from __future__ import annotations

import yaml

from elt_rl.local_bench.specs import TaskSpec

SQLITE_FILE = "app.sqlite"
DOCUMENTS_DIR = "documents"
BUCKET_DIR = "bucket"
FILES_DIR = "files"
DBT_PROJECT = "elt_project"


def pipeline_config(
    spec: TaskSpec,
    sources_dir: str,
    warehouse_path: str,
    schema: str,
    sync_mode: str = "full_refresh_append",
) -> dict:
    """A correct minielt pipeline for ``spec``."""
    sources: list[dict] = []
    connections: list[dict] = []
    by_kind: dict[str, list[str]] = {}
    for table in spec.tables:
        by_kind.setdefault(table.kind, []).append(table.name)

    def streams(names: list[str]) -> list[dict]:
        return [{"name": n, "sync_mode": sync_mode} for n in names]

    if "sqlite" in by_kind:
        sources.append({"name": "relational", "type": "sqlite", "path": f"{sources_dir}/{SQLITE_FILE}"})
        connections.append(
            {"name": "relational_to_warehouse", "source": "relational", "streams": streams(by_kind["sqlite"])}
        )
    if "documents" in by_kind:
        sources.append({"name": "documents", "type": "documents", "path": f"{sources_dir}/{DOCUMENTS_DIR}"})
        connections.append(
            {"name": "documents_to_warehouse", "source": "documents", "streams": streams(by_kind["documents"])}
        )
    if "object_store" in by_kind:
        sources.append({"name": "bucket", "type": "object_store", "path": f"{sources_dir}/{BUCKET_DIR}"})
        connections.append(
            {"name": "bucket_to_warehouse", "source": "bucket", "streams": streams(by_kind["object_store"])}
        )
    for table in spec.tables:
        if table.kind != "file":
            continue
        sources.append(
            {
                "name": f"file_{table.name}",
                "type": "file",
                "path": f"{sources_dir}/{FILES_DIR}/{table.name}.{table.fmt}",
                "format": table.fmt,
                "dataset_name": table.name,
            }
        )
        connections.append(
            {
                "name": f"file_{table.name}_to_warehouse",
                "source": f"file_{table.name}",
                "streams": streams([table.name]),
            }
        )
    return {
        "destination": {"type": "duckdb", "path": warehouse_path, "schema": schema},
        "sources": sources,
        "connections": connections,
    }


def pipeline_yaml(spec: TaskSpec, sources_dir: str, warehouse_path: str, schema: str) -> str:
    return yaml.safe_dump(
        pipeline_config(spec, sources_dir, warehouse_path, schema), sort_keys=False
    )


def reference_sql(spec: TaskSpec, model: str, schema: str) -> str:
    for m in spec.models:
        if m.name == model:
            return m.sql.format(s=f'"{schema}"').strip()
    raise KeyError(model)


def dbt_project_files(spec: TaskSpec, warehouse_path: str, schema: str) -> dict[str, str]:
    """A correct dbt project, as ``{relative path: content}``."""
    files = {
        "dbt_project.yml": yaml.safe_dump(
            {
                "name": DBT_PROJECT,
                "version": "1.0.0",
                "profile": DBT_PROJECT,
                "model-paths": ["models"],
                "models": {DBT_PROJECT: {"+materialized": "table"}},
            },
            sort_keys=False,
        ),
        "profiles.yml": yaml.safe_dump(
            {
                DBT_PROJECT: {
                    "target": "dev",
                    "outputs": {
                        "dev": {
                            "type": "duckdb",
                            "path": warehouse_path,
                            "schema": schema,
                            "threads": 1,
                        }
                    },
                }
            },
            sort_keys=False,
        ),
    }
    for model in spec.models:
        files[f"models/{model.name}.sql"] = reference_sql(spec, model.name, schema) + "\n"
    return files
