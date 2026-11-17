"""Execution-based grading of the warehouse state a rollout leaves behind.

The grader never looks at what the model *said*. It connects to the rollout's
namespace with host-side credentials and checks two things:

* **Stage 1 (extract & load)**: every expected source table exists as a base
  table with the expected row count and, when a reference load is available,
  the same content.
* **Stage 2 (transform)**: every target data model matches its ground-truth
  CSV column by column (same comparator as the official evaluator).

From those checks it derives a bounded reward in ``[0, 1]`` plus the official
success metrics, reported separately so shaping never hides the real metric.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chz
import pandas as pd

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.grading.compare import ColumnComparison, as_string_frame, compare_tables
from elt_rl.tasks import ELTTask

StartStage = Literal["full", "transform"]

# Columns Airbyte (or the local loader) adds to every loaded table. They differ
# per sync and are not part of any task schema.
_METADATA_COLUMN = re.compile(r"^(_airbyte_|_ab_)", flags=re.IGNORECASE)
_TRIVIAL_EVAL_SQL = re.compile(
    r"^\s*select\s+\*\s+from\s+[A-Za-z_][\w]*\.[A-Za-z_][\w]*"
    r"(\s+order\s+by\s+[\w\s,.\"`]+)?\s*;?\s*$",
    flags=re.IGNORECASE,
)


@chz.chz
class RewardConfig:
    """How verification results turn into a scalar reward.

    The reward is a weighted sum of three verified fractions, each in [0, 1]:

    ``reward = w_el * el_frac + w_model * model_frac + w_column * column_frac``

    All three are properties of the final warehouse state, so no term can be
    collected twice and none can be earned without loading or computing data.
    """

    w_el: float = 0.3
    """Weight of the fraction of source tables loaded correctly."""
    w_model: float = 0.5
    """Weight of the fraction of data models that match ground truth exactly."""
    w_column: float = 0.2
    """Weight of the mean fraction of correct non-key columns per model.

    Dense shaping for the common case where a model is almost right. Key
    columns are excluded: they are pass-through columns that would otherwise
    hand out credit for a table with nothing computed in it.
    """
    content_check: Literal["auto", "rowcount", "reference"] = "auto"
    """Stage-1 strictness. ``rowcount`` is the official check. ``reference``
    also requires the loaded content to equal a reference load. ``auto`` uses
    ``reference`` whenever a reference namespace is available."""
    content_fallback_max_rows: int = 200_000
    """When a content fingerprint differs, tables up to this size are
    re-compared value by value (tolerating ``1`` vs ``1.0``); larger ones fail."""
    gate_models_on_el: bool = False
    """Official protocol: stage 2 only counts when stage 1 fully passed. Off by
    default for training, because it removes almost all learning signal; the
    gated numbers are always reported under ``official/``."""
    comparison_rule: Literal["typed", "rtol"] = "typed"


@dataclass
class SourceTableCheck:
    name: str
    exists: bool = False
    rows: int | None = None
    expected_rows: int = 0
    content_ok: bool | None = None
    """None when no content check ran."""
    error: str | None = None

    @property
    def count_ok(self) -> bool:
        return self.exists and self.rows == self.expected_rows

    @property
    def ok(self) -> bool:
        return self.count_ok and self.content_ok is not False


@dataclass
class ModelCheck:
    name: str
    exists: bool = False
    comparison: ColumnComparison = field(default_factory=ColumnComparison)
    key_columns: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def match(self) -> bool:
        return self.exists and self.error is None and self.comparison.match

    @property
    def column_frac(self) -> float:
        """Fraction of non-key ground-truth columns that match."""
        keys = {k.lower() for k in self.key_columns}
        c = self.comparison
        matched = [x for x in c.matched if x.lower() not in keys]
        total = [x for x in c.matched + c.unmatched + c.missed if x.lower() not in keys]
        if not total:
            return 1.0 if self.match else 0.0
        return len(matched) / len(total)


@dataclass
class GradeResult:
    task: str
    start_stage: StartStage
    sources: list[SourceTableCheck]
    models: list[ModelCheck]
    reward: float
    metrics: dict[str, float]
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"task={self.task} stage={self.start_stage} reward={self.reward:.3f}"]
        for s in self.sources:
            status = "ok" if s.ok else "FAIL"
            detail = f"rows={s.rows} expected={s.expected_rows}"
            if s.content_ok is not None:
                detail += f" content={'ok' if s.content_ok else 'DIFF'}"
            if s.error:
                detail += f" error={s.error}"
            lines.append(f"  source {s.name}: {status} ({detail})")
        for m in self.models:
            status = "ok" if m.match else "FAIL"
            detail = (
                f"matched={len(m.comparison.matched)} unmatched={m.comparison.unmatched} "
                f"missed={m.comparison.missed}"
            )
            if m.error:
                detail += f" error={m.error}"
            lines.append(f"  model {m.name}: {status} ({detail})")
        return "\n".join(lines)


@lru_cache(maxsize=512)
def _load_ground_truth(path: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns  # cache key only
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Ground-truth CSV as strings, parsed once per file and cached."""
    return _load_ground_truth(str(path), path.stat().st_mtime_ns)


class _Listing:
    """The relations of a namespace, fetched once per grading pass.

    Every check needs to know whether its table exists; asking the warehouse
    once instead of once per check removes most of the round trips.
    """

    def __init__(self, destination: Destination, ns: Namespace) -> None:
        self._destination = destination
        self._ns = ns
        self._lock = threading.Lock()
        self._cache: dict[bool, list[str]] = {}

    def _get(self, include_views: bool) -> list[str]:
        with self._lock:
            if include_views not in self._cache:
                self._cache[include_views] = self._destination.list_tables(
                    self._ns, include_views=include_views
                )
            return self._cache[include_views]

    def base_tables(self) -> list[str]:
        return self._get(False)

    def relations(self) -> list[str]:
        return self._get(True)


class Grader:
    """Grades rollout namespaces of one destination."""

    def __init__(
        self,
        destination: Destination,
        config: RewardConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        self.destination = destination
        self.config = config or RewardConfig()
        self.max_workers = max(1, max_workers)
        # Fingerprints of the immutable reference load, computed once.
        self._reference_fingerprints: dict[tuple[str, str, str], tuple[int, str, list[str]]] = {}

    # -- stage 1 ------------------------------------------------------------------
    def _data_columns(self, ns: Namespace, table: str) -> list[str]:
        return [
            c for c in self.destination.table_columns(ns, table) if not _METADATA_COLUMN.match(c)
        ]

    def _reference_fingerprint(
        self, reference: Namespace, table: str
    ) -> tuple[int, str, list[str]] | None:
        key = (reference.database, reference.schema, table)
        if key not in self._reference_fingerprints:
            actual = self.destination.resolve_table(reference, table)
            if actual is None:
                return None
            columns = self._data_columns(reference, actual)
            rows, digest = self.destination.table_fingerprint(reference, actual, columns)
            self._reference_fingerprints[key] = (rows, digest, columns)
        return self._reference_fingerprints[key]

    def _check_content(
        self, ns: Namespace, actual: str, reference: Namespace, table: str
    ) -> bool | None:
        ref = self._reference_fingerprint(reference, table)
        if ref is None:
            return None
        ref_rows, ref_digest, ref_columns = ref
        actual_columns = {
            self.destination.norm(c): c for c in self.destination.table_columns(ns, actual)
        }
        mapped = [actual_columns.get(self.destination.norm(c)) for c in ref_columns]
        if any(c is None for c in mapped):
            return False  # a documented source column was not loaded
        columns = [c for c in mapped if c is not None]
        rows, digest = self.destination.table_fingerprint(ns, actual, columns)
        if rows != ref_rows:
            return False
        if digest == ref_digest:
            return True
        if rows > self.config.content_fallback_max_rows:
            return False
        # Same rows but different rendering (e.g. 1 vs 1.0, text vs integer):
        # compare value by value with the tolerant comparator.
        ref_actual = self.destination.resolve_table(reference, table)
        assert ref_actual is not None
        gold = as_string_frame(self.destination.fetch_table(reference, ref_actual)[ref_columns])
        candidate = as_string_frame(self.destination.fetch_table(ns, actual)[columns])
        candidate.columns = list(gold.columns)
        return compare_tables(gold, candidate, list(gold.columns), self.config.comparison_rule).match

    def _check_source(
        self,
        ns: Namespace,
        table: str,
        expected_rows: int,
        reference: Namespace | None,
        listing: _Listing,
    ) -> SourceTableCheck:
        check = SourceTableCheck(name=table, expected_rows=expected_rows)
        try:
            # A source table must be a base table: a view is not loaded data.
            actual = self.destination.match_table(listing.base_tables(), table)
            if actual is None:
                return check
            check.exists = True
            check.rows = self.destination.table_size(ns, actual)
            use_reference = reference is not None and self.config.content_check != "rowcount"
            if check.count_ok and use_reference:
                assert reference is not None
                check.content_ok = self._check_content(ns, actual, reference, table)
        except Exception as e:  # a broken table is a failed check, not a crash
            check.error = f"{type(e).__name__}: {e}"[:300]
        return check

    # -- stage 2 ------------------------------------------------------------------
    def _check_model(
        self, task: ELTTask, ns: Namespace, model: str, listing: _Listing
    ) -> ModelCheck:
        keys = list(task.sort_keys.get(model, []))
        check = ModelCheck(name=model, key_columns=keys)
        gt_path = task.gt_path(model)
        try:
            gold = load_ground_truth(gt_path)
            check.comparison.missed = list(gold.columns)
            # A data model may be a view (dbt's default materialization).
            actual = self.destination.match_table(listing.relations(), model)
            if actual is None:
                return check
            check.exists = True
            # Cheap guard: a wrong row count fails every column, so skip the fetch.
            if self.destination.table_size(ns, actual) != len(gold):
                check.comparison = ColumnComparison(unmatched=list(gold.columns))
                return check
            sql = task.eval_sql.get(model)
            if sql and not _TRIVIAL_EVAL_SQL.match(sql):
                fetched = self.destination.query_eval_sql(sql, ns)
            else:
                fetched = self.destination.fetch_table(ns, actual)
            candidate = as_string_frame(fetched)
            check.comparison = compare_tables(gold, candidate, keys, self.config.comparison_rule)
        except Exception as e:
            check.error = f"{type(e).__name__}: {e}"[:300]
        return check

    # -- entry point ----------------------------------------------------------------
    def grade(
        self,
        task: ELTTask,
        ns: Namespace,
        start_stage: StartStage = "full",
        reference: Namespace | None = None,
    ) -> GradeResult:
        """Grade one rollout's namespace. Synchronous; run it in a thread."""
        started = time.monotonic()
        if self.config.content_check == "reference" and reference is None:
            raise ValueError("content_check='reference' needs a reference namespace")

        listing = _Listing(self.destination, ns)

        def run_all() -> tuple[list[SourceTableCheck], list[ModelCheck]]:
            source_jobs = list(task.source_tables.items())
            if self.max_workers == 1:
                sources = [
                    self._check_source(ns, t, n, reference, listing) for t, n in source_jobs
                ]
                models = [self._check_model(task, ns, m, listing) for m in task.models]
                return sources, models
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                source_futures = [
                    pool.submit(self._check_source, ns, t, n, reference, listing)
                    for t, n in source_jobs
                ]
                model_futures = [
                    pool.submit(self._check_model, task, ns, m, listing) for m in task.models
                ]
                return [f.result() for f in source_futures], [f.result() for f in model_futures]

        with self.destination.session(ns):
            sources, models = run_all()
        reward, metrics = self.score(sources, models, start_stage)
        seconds = time.monotonic() - started
        metrics["elt/grade_seconds"] = seconds
        return GradeResult(
            task=task.name,
            start_stage=start_stage,
            sources=sources,
            models=models,
            reward=reward,
            metrics=metrics,
            seconds=seconds,
        )

    def score(
        self, sources: list[SourceTableCheck], models: list[ModelCheck], start_stage: StartStage
    ) -> tuple[float, dict[str, float]]:
        """Turn check results into ``(reward, metrics)``."""
        cfg = self.config
        el_frac = sum(s.ok for s in sources) / len(sources) if sources else 0.0
        el_success = float(bool(sources) and all(s.ok for s in sources))
        model_frac = sum(m.match for m in models) / len(models) if models else 0.0
        column_frac = sum(m.column_frac for m in models) / len(models) if models else 0.0

        gate = el_success if cfg.gate_models_on_el else 1.0
        if start_stage == "transform":
            # Sources were pre-loaded: they earn nothing, but tampering with them
            # (dropping or reloading tables) voids the episode.
            denom = cfg.w_model + cfg.w_column
            transform = (cfg.w_model * model_frac + cfg.w_column * column_frac) / denom if denom else 0.0
            reward = transform * el_success
        else:
            total = cfg.w_el + cfg.w_model + cfg.w_column
            reward = (
                cfg.w_el * el_frac + gate * (cfg.w_model * model_frac + cfg.w_column * column_frac)
            ) / (total if total else 1.0)

        # Row-count-only stage 1, i.e. exactly what the official evaluator checks.
        official_stage1 = float(bool(sources) and all(s.count_ok for s in sources))
        metrics: dict[str, float] = {
            "elt/el_frac": el_frac,
            "elt/el_success": el_success,
            "elt/model_frac": model_frac,
            "elt/column_frac": column_frac,
            "elt/task_success": float(el_success == 1.0 and model_frac == 1.0),
            "elt/models_correct": float(sum(m.match for m in models)),
            "elt/models_total": float(len(models)),
            "elt/content_mismatch": float(sum(s.content_ok is False for s in sources)),
            "elt/check_errors": float(
                sum(s.error is not None for s in sources) + sum(m.error is not None for m in models)
            ),
            # Official ELT-Bench metrics: SRDEL (stage 1 passed) and SRDT (models
            # correct, counted only when stage 1 passed).
            "official/stage1_success": official_stage1,
            "official/stage2_frac": model_frac * official_stage1,
        }
        return float(reward), metrics
