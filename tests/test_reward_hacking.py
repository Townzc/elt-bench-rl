"""Regression tests for reward hacking.

Each ``cheat_*`` script in ``elt_rl.scripted`` is a shortcut an RL policy could
discover. Every test runs the shortcut through the real environment and
asserts that it earns nothing for the part it faked.
"""

from __future__ import annotations

import chz
import pytest

from elt_rl.grading.grader import RewardConfig

from .conftest import run_policy

W = RewardConfig()


async def test_fake_row_counts_fail_the_content_check(tasks, runtime_config) -> None:
    """Tables with the right row count and no real data.

    The official stage-1 check (row counts only) accepts them, which is why
    the reward uses the content check against a reference load instead.
    """
    (grade,) = await run_policy(tasks["bookshop"], runtime_config, "cheat_fake_row_counts")
    assert grade.metrics["official/stage1_success"] == 1.0  # the loophole
    assert grade.metrics["elt/el_frac"] == 0.0  # closed
    assert grade.metrics["elt/content_mismatch"] == len(grade.sources)
    assert grade.reward == 0.0

    # The same rollout under the official row-count-only check collects EL credit.
    rowcount_only = chz.replace(W, content_check="rowcount")
    (grade,) = await run_policy(
        tasks["bookshop"], runtime_config, "cheat_fake_row_counts", reward_config=rowcount_only
    )
    assert grade.reward == pytest.approx(W.w_el)


async def test_duplicated_loads_fail(tasks, runtime_config) -> None:
    """Syncing an append stream twice doubles its rows: not a correct load."""
    (grade,) = await run_policy(tasks["airfare"], runtime_config, "cheat_double_sync")
    assert grade.metrics["elt/el_success"] == 0.0
    assert grade.metrics["elt/el_frac"] < 1.0
    assert any(s.exists and not s.count_ok for s in grade.sources)
    assert grade.metrics["elt/task_success"] == 0.0


async def test_handmade_tables_do_not_survive_replay(tasks, runtime_config) -> None:
    """Correct final tables written by a one-off script, with no dbt project.

    The replay check drops the data models and rebuilds them from the
    submitted project, so only tables the pipeline produces are graded.
    """
    (grade,) = await run_policy(tasks["clinic"], runtime_config, "cheat_handmade_tables")
    assert grade.metrics["elt/replay_ok"] == 0.0
    assert grade.metrics["elt/model_frac"] == 0.0
    assert grade.metrics["elt/column_frac"] == 0.0
    assert grade.reward == pytest.approx(W.w_el)  # the extract/load part was real

    # Without the replay check the same shortcut gets full reward.
    no_replay = chz.replace(runtime_config, verify_replay=False)
    (grade,) = await run_policy(tasks["clinic"], no_replay, "cheat_handmade_tables")
    assert grade.reward == pytest.approx(1.0)


async def test_models_reading_handmade_copies_do_not_survive_replay(tasks, runtime_config) -> None:
    """dbt models that are thin wrappers around hand-materialized tables.

    The replay starts from the source tables only, so the hand-made copies
    are gone when the project is run again.
    """
    (grade,) = await run_policy(tasks["airfare"], runtime_config, "cheat_passthrough_models")
    assert grade.metrics["elt/replay_ok"] == 0.0
    assert grade.metrics["elt/model_frac"] == 0.0
    assert grade.metrics["elt/el_success"] == 1.0  # source tables are left alone by the replay
    assert grade.reward == pytest.approx(W.w_el)


async def test_key_columns_alone_earn_no_column_credit(tasks, runtime_config) -> None:
    """Models with the right keys and NULL everywhere else.

    Key columns are pass-through values; counting them would pay for a table
    in which nothing was computed.
    """
    (grade,) = await run_policy(tasks["bookshop"], runtime_config, "cheat_keys_only")
    assert all(m.exists for m in grade.models)
    assert grade.metrics["elt/replay_ok"] == 1.0
    assert grade.metrics["elt/model_frac"] == 0.0
    assert grade.metrics["elt/column_frac"] == 0.0
    assert grade.reward == pytest.approx(W.w_el)


async def test_tampering_with_preloaded_sources_voids_the_episode(tasks, runtime_config) -> None:
    """Transformation-only start: the pre-loaded source tables must stay intact."""
    (grade,) = await run_policy(
        tasks["airfare"], runtime_config, "cheat_tamper_sources", start_stage="transform"
    )
    assert grade.metrics["elt/el_success"] == 0.0
    assert grade.reward == 0.0


async def test_official_gating_option(tasks, runtime_config) -> None:
    """``gate_models_on_el`` reproduces the official protocol in the reward."""
    gated = chz.replace(W, gate_models_on_el=True)
    (grade,) = await run_policy(
        tasks["airfare"], runtime_config, "cheat_double_sync", reward_config=gated
    )
    # Whatever the models look like, they count for nothing once stage 1 failed.
    assert grade.reward == pytest.approx(W.w_el * grade.metrics["elt/el_frac"])
    assert grade.metrics["official/stage2_frac"] == 0.0
