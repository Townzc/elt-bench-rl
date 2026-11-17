"""Unit tests for the comparator, the scoring rule and the grader's checks."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from elt_rl.destinations import DuckDBDestination, Namespace
from elt_rl.grading.compare import ColumnComparison, compare_tables
from elt_rl.grading.grader import Grader, ModelCheck, RewardConfig, SourceTableCheck
from elt_rl.tasks import ELTTask


def frame(**columns: list) -> pd.DataFrame:
    return pd.DataFrame({k: [None if v is None else str(v) for v in vs] for k, vs in columns.items()})


# -- comparator -------------------------------------------------------------------------
def test_comparison_ignores_row_order_column_case_and_extra_columns() -> None:
    gold = frame(id=[1, 2, 3], name=["a", "b", "c"])
    candidate = frame(NAME=["c", "a", "b"], ID=[3, 1, 2], extra=["x", "y", "z"])
    result = compare_tables(gold, candidate, ["id"])
    assert result.match and result.matched == ["id", "name"]


def test_comparison_reports_missing_and_wrong_columns() -> None:
    gold = frame(id=[1, 2], total=[10, 20], label=["a", "b"])
    candidate = frame(id=[1, 2], total=[10, 21])
    result = compare_tables(gold, candidate, ["id"])
    assert (result.matched, result.unmatched, result.missed) == (["id"], ["total"], ["label"])
    assert not result.match and result.total == 3


def test_typed_rule_is_exact_for_integers_and_tolerant_for_decimals() -> None:
    gold = frame(id=[1, 2], count=[1000, 2000], avg=["3.14", "2.50"])
    # 1000 vs 1001 is within 1% but an integer column must be equal.
    off_by_one = frame(id=[1, 2], count=[1001, 2000], avg=["3.14", "2.50"])
    assert compare_tables(gold, off_by_one, ["id"]).unmatched == ["count"]
    assert compare_tables(gold, off_by_one, ["id"], rule="rtol").match
    # Integers rendered as floats, and decimals rounded differently, are fine.
    rendered = frame(id=["1.0", "2.0"], count=["1000.0", "2000.0"], avg=["3.141", "2.5"])
    assert compare_tables(gold, rendered, ["id"]).match
    wrong = frame(id=[1, 2], count=[1000, 2000], avg=["3.16", "2.50"])
    assert compare_tables(gold, wrong, ["id"]).unmatched == ["avg"]


def test_nulls_and_text_normalization() -> None:
    gold = frame(id=[1, 2], city=["Paris", None])
    assert compare_tables(gold, frame(id=[1, 2], city=[" paris ", None]), ["id"]).match
    assert not compare_tables(gold, frame(id=[1, 2], city=["Paris", "x"]), ["id"]).match
    assert not compare_tables(gold, frame(id=[1], city=["Paris"]), ["id"]).match  # row count


# -- scoring ----------------------------------------------------------------------------
def source(ok: bool, content_ok: bool | None = None) -> SourceTableCheck:
    return SourceTableCheck(
        name="t", exists=True, rows=5 if ok else 4, expected_rows=5, content_ok=content_ok
    )


def model(matched: list[str], unmatched: list[str], keys: list[str]) -> ModelCheck:
    return ModelCheck(
        name="m",
        exists=True,
        comparison=ColumnComparison(matched=matched, unmatched=unmatched),
        key_columns=keys,
    )


def test_reward_is_a_weighted_sum_of_verified_fractions() -> None:
    grader = Grader(DuckDBDestination(), RewardConfig())
    sources = [source(True), source(False)]
    models = [model(["id", "a", "b"], [], ["id"]), model(["id", "a"], ["b"], ["id"])]
    reward, metrics = grader.score(sources, models, "full")
    assert metrics["elt/el_frac"] == 0.5 and metrics["elt/el_success"] == 0.0
    assert metrics["elt/model_frac"] == 0.5
    assert metrics["elt/column_frac"] == pytest.approx(0.75)  # (2/2 + 1/2) / 2, keys excluded
    assert reward == pytest.approx(0.3 * 0.5 + 0.5 * 0.5 + 0.2 * 0.75)
    assert metrics["elt/task_success"] == 0.0
    assert metrics["official/stage2_frac"] == 0.0  # official: stage 2 gated on stage 1

    perfect, metrics = grader.score([source(True)], [models[0]], "full")
    assert perfect == pytest.approx(1.0) and metrics["elt/task_success"] == 1.0
    empty, _ = grader.score([source(False)], [ModelCheck(name="m")], "full")
    assert empty == 0.0


def test_transform_stage_reward_ignores_el_credit_but_requires_intact_sources() -> None:
    grader = Grader(DuckDBDestination(), RewardConfig())
    models = [model(["id", "a"], ["b"], ["id"])]
    reward, _ = grader.score([source(True)], models, "transform")
    assert reward == pytest.approx((0.5 * 0.0 + 0.2 * 0.5) / 0.7)
    reward, _ = grader.score([source(True), source(False)], models, "transform")
    assert reward == 0.0


def test_content_mismatch_fails_a_source_but_not_the_official_metric() -> None:
    grader = Grader(DuckDBDestination(), RewardConfig())
    reward, metrics = grader.score([source(True, content_ok=False)], [], "full")
    assert reward == 0.0 and metrics["elt/content_mismatch"] == 1.0
    assert metrics["official/stage1_success"] == 1.0


def test_key_only_model_gets_no_column_credit() -> None:
    assert model(["id"], ["a", "b"], ["id"]).column_frac == 0.0
    assert model(["ID", "a"], ["b"], ["id"]).column_frac == 0.5


# -- grader against a real (DuckDB) warehouse ---------------------------------------------
@pytest.fixture
def warehouse(tmp_path: Path):
    """A reference load and a helper to create rollout namespaces from SQL."""
    destination = DuckDBDestination()
    reference = Namespace(database=str(tmp_path / "ref.duckdb"), schema="shop")
    con = duckdb.connect(reference.database)
    con.execute("create schema shop")
    con.execute("create table shop.orders as select i as id, i * 2 as qty from range(5) t(i)")
    con.close()

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "summary.csv").write_text("bucket,n,total\n0,3,12\n1,2,8\n")
    task = ELTTask(
        name="shop",
        destination="duckdb",
        input_dir=tmp_path,
        source_tables={"orders": 5},
        models=("summary",),
        gt_dir=gt_dir,
        sort_keys={"summary": ["bucket"]},
        # Not a plain "select *": runs through the destination's SQL rewriting.
        eval_sql={"summary": "select s.* from shop.summary s join shop.orders o on o.id = 0"},
    )

    def make(name: str, statements: list[str]) -> Namespace:
        ns = Namespace(database=str(tmp_path / f"{name}.duckdb"), schema="shop")
        con = duckdb.connect(ns.database)
        con.execute("create schema shop")
        for statement in statements:
            con.execute(statement)
        con.close()
        return ns

    return destination, reference, task, make


GOOD_SUMMARY = (
    "create table shop.summary as select id % 2 as bucket, count(*) as n, sum(qty) as total "
    "from shop.orders group by 1"
)


def test_grader_end_to_end(warehouse) -> None:
    destination, reference, task, make = warehouse
    grader = Grader(destination, RewardConfig())
    ns = make(
        "good",
        [
            # Loaded as text with a loader column, as an EL tool might do.
            "create table shop.orders as select cast(i as varchar) as id, "
            "cast(i * 2 as double) as qty, 'x' as _airbyte_raw_id from range(5) t(i)",
            GOOD_SUMMARY.replace("sum(qty)", "sum(cast(qty as bigint))").replace(
                "id % 2", "cast(id as integer) % 2"
            ),
        ],
    )
    result = grader.grade(task, ns, "full", reference)
    assert result.reward == pytest.approx(1.0), result.summary()
    # The fingerprint differs (text/double vs integers); the value comparison accepts it.
    assert result.sources[0].content_ok is True
    assert "task=shop" in result.summary()


def test_grader_rejects_wrong_content_views_as_sources_and_missing_models(warehouse) -> None:
    destination, reference, task, make = warehouse
    grader = Grader(destination, RewardConfig())

    wrong = make("wrong", ["create table shop.orders as select i as id, 7 as qty from range(5) t(i)"])
    result = grader.grade(task, wrong, "full", reference)
    assert result.sources[0].count_ok and result.sources[0].content_ok is False
    assert result.reward == 0.0 and not result.models[0].exists

    # A source "table" that is really a view was not loaded.
    view = make("view", ["create view shop.orders as select i as id, i * 2 as qty from range(5) t(i)"])
    assert grader.grade(task, view, "full", reference).sources[0].exists is False

    # A data model may be a view (dbt's default materialization).
    as_view = make(
        "model_view",
        [
            "create table shop.orders as select i as id, i * 2 as qty from range(5) t(i)",
            GOOD_SUMMARY.replace("create table", "create view"),
        ],
    )
    assert grader.grade(task, as_view, "full", reference).reward == pytest.approx(1.0)

    # Missing column in the source: content check fails even though the count is right.
    narrow = make("narrow", ["create table shop.orders as select i as id from range(5) t(i)"])
    assert grader.grade(task, narrow, "full", reference).sources[0].content_ok is False

    # A namespace that does not exist at all is a clean zero, not an error.
    missing = Namespace(database=str(Path(reference.database).parent / "nope.duckdb"), schema="shop")
    result = grader.grade(task, missing, "full", reference)
    assert result.reward == 0.0 and result.metrics["elt/check_errors"] == 0.0


def test_row_count_guard_and_reference_requirement(warehouse) -> None:
    destination, reference, task, make = warehouse
    ns = make(
        "short",
        [
            "create table shop.orders as select i as id, i * 2 as qty from range(5) t(i)",
            "create table shop.summary as select 0 as bucket, 3 as n, 12 as total",
        ],
    )
    result = Grader(destination, RewardConfig()).grade(task, ns, "full", reference)
    assert result.models[0].comparison.unmatched == ["bucket", "n", "total"]
    assert result.reward == pytest.approx(0.3)
    with pytest.raises(ValueError, match="reference"):
        Grader(destination, RewardConfig(content_check="reference")).grade(task, ns, "full", None)
    # Without a reference the check falls back to the official row count.
    assert Grader(destination, RewardConfig()).grade(task, ns, "full", None).sources[0].ok
