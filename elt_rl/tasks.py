"""ELT task definition and loaders.

A task is one ELT-Bench database: a bundle of agent-visible inputs
(``config.yaml``, ``data_model.yaml``, ``schemas/``, ``documentation/``) plus
host-only grading assets (expected source row counts, ground-truth CSVs, unique
keys, evaluation SQL).

Both the official benchmark and the credential-free local benchmark are stored
in the same on-disk layout, so one loader serves both::

    <root>/
      inputs[_<destination>]/<task>/      agent-visible bundle (write_config.py output)
      evaluation/table.json               {task: {source_table: row_count}}
      evaluation/sort_key.json            {task: {model: [unique key columns]}}
      evaluation/sql/<task>/<model>.sql   optional evaluation query per model
      ground_truth/gt_<destination>/<task>/<model>.csv

Grading assets are never copied into the agent's workspace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ELTTask:
    """One ELT-Bench task (a source database and its target data models)."""

    name: str
    """Logical schema name used by the benchmark, e.g. ``books``."""
    destination: str
    """Destination the bundle was generated for (``duckdb``, ``snowflake``, ...)."""
    input_dir: Path
    """Agent-visible bundle. Copied into the rollout workspace."""
    source_tables: dict[str, int]
    """Expected row count of every source table once extract/load is correct."""
    models: tuple[str, ...]
    """Target data models declared in ``data_model.yaml``."""
    gt_dir: Path
    """Host-only ground-truth directory holding ``<model>.csv``."""
    sort_keys: dict[str, list[str]] = field(default_factory=dict)
    """Unique-key columns per model, used to align rows before comparison."""
    eval_sql: dict[str, str] = field(default_factory=dict)
    """Optional evaluation query per model, in ``schema.table`` form."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Runtime-specific paths (e.g. local source data, reference warehouse)."""

    def gt_path(self, model: str) -> Path:
        return self.gt_dir / f"{model}.csv"


def default_inputs_dir(root: Path, destination: str) -> Path:
    """Where ``setup/write_config.py`` puts the bundles for a destination."""
    return root / ("inputs" if destination == "snowflake" else f"inputs_{destination}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _model_names(
    data_model_path: Path, task_gt: Path, sort_keys: dict[str, Any], eval_sql: dict[str, str]
) -> tuple[str, ...]:
    """Names of the data models to grade.

    The ground-truth directory is authoritative (the official evaluator also
    drives stage 2 from the ground-truth CSVs). Without it, fall back to the
    benchmark metadata, then to ``data_model.yaml``; that file is written for
    agents and is not always strictly valid YAML, so it comes last.
    """
    if task_gt.is_dir():
        names = sorted(p.stem for p in task_gt.glob("*.csv"))
        if names:
            return tuple(names)
    if sort_keys:
        return tuple(sorted(sort_keys))
    if eval_sql:
        return tuple(sorted(eval_sql))
    try:
        with data_model_path.open(encoding="utf-8") as f:
            data_model = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return ()
    return tuple(str(m["name"]).strip() for m in data_model.get("models", []))


def _lookup_ci(mapping: dict[str, Any], key: str) -> Any:
    """Case-insensitive dict lookup (benchmark metadata mixes cases)."""
    lowered = {k.lower(): v for k, v in mapping.items()}
    return lowered.get(key.lower())


def load_tasks(
    root: str | Path,
    destination: str,
    *,
    inputs_dir: str | Path | None = None,
    gt_dir: str | Path | None = None,
    only: list[str] | None = None,
    require_ground_truth: bool = True,
) -> list[ELTTask]:
    """Load tasks from an ELT-Bench style directory tree.

    Args:
        root: Benchmark root (the ELT-Bench checkout, or a local benchmark
            built by ``elt_rl.local_bench``).
        destination: Destination name the bundles were generated for.
        inputs_dir: Agent input bundles; defaults to the ``write_config.py``
            output location for ``destination``.
        gt_dir: Ground-truth root holding ``<task>/<model>.csv``; defaults to
            ``<root>/ground_truth/gt_<destination>``.
        only: Restrict to these task names.
        require_ground_truth: Skip tasks with no ground truth (they cannot
            produce a transformation reward).
    """
    root = Path(root).expanduser().resolve()
    inputs = Path(inputs_dir).expanduser().resolve() if inputs_dir else default_inputs_dir(
        root, destination
    )
    gt_root = (
        Path(gt_dir).expanduser().resolve()
        if gt_dir
        else root / "ground_truth" / f"gt_{destination}"
    )
    if not inputs.is_dir():
        raise FileNotFoundError(
            f"Agent input bundles not found: {inputs}. For the official benchmark run "
            f"`python setup/write_config.py --destination {destination}` first."
        )

    table_json = _read_json(root / "evaluation" / "table.json")
    sort_key_json = _read_json(root / "evaluation" / "sort_key.json")
    sql_root = root / "evaluation" / "sql"

    wanted = {name.lower() for name in only} if only else None
    tasks: list[ELTTask] = []
    for task_dir in sorted(p for p in inputs.iterdir() if p.is_dir()):
        name = task_dir.name
        if wanted is not None and name.lower() not in wanted:
            continue
        data_model = task_dir / "data_model.yaml"
        if not (task_dir / "config.yaml").is_file() or not data_model.is_file():
            continue
        source_tables = _lookup_ci(table_json, name)
        if not source_tables:
            raise ValueError(f"No expected source tables for task {name!r} in table.json")
        task_gt = gt_root / name
        if require_ground_truth and not task_gt.is_dir():
            continue

        eval_sql: dict[str, str] = {}
        task_sql = sql_root / name
        if task_sql.is_dir():
            for sql_file in task_sql.glob("*.sql"):
                eval_sql[sql_file.stem] = sql_file.read_text(encoding="utf-8")
        sort_keys = {
            str(k): list(v) for k, v in (_lookup_ci(sort_key_json, name) or {}).items()
        }
        models = _model_names(data_model, task_gt, sort_keys, eval_sql)

        extras: dict[str, Any] = {}
        sources_dir = root / "sources" / name
        if sources_dir.is_dir():
            extras["sources_dir"] = sources_dir
        reference_db = root / "reference" / f"{name}__reference.duckdb"
        if reference_db.is_file():
            extras["reference_db"] = reference_db
        tasks.append(
            ELTTask(
                name=name,
                destination=destination,
                input_dir=task_dir,
                source_tables={str(k): int(v) for k, v in source_tables.items()},
                models=models,
                gt_dir=task_gt,
                sort_keys=sort_keys,
                eval_sql=eval_sql,
                extras=extras,
            )
        )
    if wanted is not None:
        missing = wanted - {t.name.lower() for t in tasks}
        if missing:
            raise ValueError(f"Tasks not found under {inputs}: {sorted(missing)}")
    return tasks
