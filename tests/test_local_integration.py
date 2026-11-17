"""Credential-free end-to-end test.

Real environment, real sandboxed commands (``minielt`` and ``dbt`` against
DuckDB), real grader; only the policy is scripted. No API key, no Docker, no
network.
"""

from __future__ import annotations

import os
from pathlib import Path

import chz
import pytest
import yaml

from elt_rl.env import EnvConfig
from elt_rl.grading.grader import RewardConfig
from elt_rl.runtime import get_runtime, new_rollout_id
from elt_rl.scripted import SUBMIT, assistant_message, oracle_script, run_script

from .conftest import make_builder, run_policy

TASK_NAMES = ["airfare", "bookshop", "clinic"]


def test_local_benchmark_layout(bench_root: Path, tasks) -> None:
    assert sorted(tasks) == TASK_NAMES
    for task in tasks.values():
        assert task.destination == "duckdb"
        assert len(task.source_tables) >= 3
        assert len(task.models) == 2
        for model in task.models:
            assert task.gt_path(model).is_file()
            assert task.sort_keys[model]
        # Grading assets live outside the bundle that is copied to the agent.
        bundled = {p.name for p in task.input_dir.rglob("*")}
        assert not bundled & {f"{m}.csv" for m in task.models}
        assert task.gt_dir is not None and task.input_dir not in task.gt_dir.parents


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("start_stage", ["full", "transform"])
async def test_oracle_solves_task(tasks, runtime_config, task_name: str, start_stage) -> None:
    (grade,) = await run_policy(tasks[task_name], runtime_config, "oracle", start_stage=start_stage)
    assert grade.reward == pytest.approx(1.0), grade.summary()
    assert grade.metrics["elt/task_success"] == 1.0
    assert grade.metrics["elt/el_success"] == 1.0
    assert grade.metrics["elt/model_frac"] == 1.0
    assert grade.metrics["elt/submitted"] == 1.0
    assert grade.metrics["elt/replay_ok"] == 1.0
    assert grade.metrics["elt/check_errors"] == 0.0
    assert grade.metrics["official/stage1_success"] == 1.0
    assert grade.metrics["official/stage2_frac"] == 1.0
    # Stage 1 was verified against the reference load, not just by row count.
    assert all(s.content_ok is True for s in grade.sources)


@pytest.mark.parametrize("start_stage", ["full", "transform"])
async def test_doing_nothing_earns_nothing(tasks, runtime_config, start_stage) -> None:
    (grade,) = await run_policy(tasks["bookshop"], runtime_config, "noop", start_stage=start_stage)
    assert grade.reward == 0.0
    assert grade.metrics["elt/task_success"] == 0.0
    assert grade.metrics["elt/model_frac"] == 0.0


async def test_partial_pipeline_is_credited_without_submit(tasks, runtime_config) -> None:
    """A rollout that stops after extract/load keeps the stage-1 credit."""
    (grade,) = await run_policy(tasks["clinic"], runtime_config, "extract_load_only")
    assert grade.metrics["elt/submitted"] == 0.0
    assert grade.metrics["elt/el_success"] == 1.0
    assert grade.reward == pytest.approx(RewardConfig().w_el)

    strict = EnvConfig(require_submit=True)
    (grade,) = await run_policy(
        tasks["clinic"], runtime_config, "extract_load_only", env_config=strict
    )
    assert grade.reward == 0.0


async def test_lingering_process_cannot_block_grading(tasks, runtime_config) -> None:
    """A background process holding the warehouse open is stopped before grading."""
    (grade,) = await run_policy(tasks["airfare"], runtime_config, "leave_background_process")
    assert grade.metrics["elt/check_errors"] == 0.0
    assert grade.reward == pytest.approx(1.0), grade.summary()


async def test_group_rollouts_are_isolated_and_cleaned_up(tasks, runtime_config) -> None:
    builder = make_builder(tasks["airfare"], runtime_config, group_size=3)
    rollouts = await builder.start_rollouts()
    try:
        workspaces = [r.ctx.host_workspace for r in rollouts]
        assert len({str(w) for w in workspaces}) == 3
        assert len({r.ctx.namespace.database for r in rollouts}) == 3
        assert len({r.ctx.rollout_id for r in rollouts}) == 3

        # Only the first member does the work; the others must not benefit.
        env = builder.build_message_env(rollouts[0])
        await run_script(env, oracle_script(rollouts[0].ctx))
        grades = [await builder.grade_rollout(r) for r in rollouts]
        assert grades[0].reward == pytest.approx(1.0)
        assert grades[1].reward == 0.0 and grades[2].reward == 0.0
    finally:
        await builder.cleanup()
    assert not any(w.exists() for w in workspaces)


async def test_concurrent_group(tasks, runtime_config) -> None:
    grades = await run_policy(tasks["bookshop"], runtime_config, "oracle", group_size=4)
    assert [g.reward for g in grades] == pytest.approx([1.0] * 4)


@pytest.mark.parametrize("start_stage", ["full", "transform"])
async def test_workspace_exposes_no_grading_assets(tasks, runtime_config, start_stage) -> None:
    task = tasks["bookshop"]
    runtime = get_runtime(runtime_config)
    ctx = await runtime.start(task, new_rollout_id(), start_stage)
    try:
        files = [p for p in ctx.host_workspace.rglob("*") if p.is_file()]
        names = {p.name for p in files}
        assert "config.yaml" in names and "data_model.yaml" in names
        assert not any(n.endswith("_credential.json") for n in names)
        assert not names & {f"{m}.csv" for m in task.models}, "ground truth leaked"
        assert not names & {"table.json", "sort_key.json", "manifest.json"}
        assert not any("reference" in p.name for p in files)
        if start_stage == "transform":
            # Sources are already in the warehouse; the source systems are absent.
            assert not (ctx.host_workspace / "sources").exists()
            config = yaml.safe_load((ctx.host_workspace / "config.yaml").read_text())
            assert list(config) == ["duckdb"]
        else:
            assert (ctx.host_workspace / "sources").is_dir()
        # The task config points the agent at this rollout's own warehouse.
        config = yaml.safe_load((ctx.host_workspace / "config.yaml").read_text())
        assert config["duckdb"]["config"]["path"] == ctx.namespace.for_agent
        assert str(ctx.host_workspace) in ctx.namespace.for_agent
    finally:
        await runtime.stop(ctx)


async def test_tools_enforce_their_contract(tasks, runtime_config) -> None:
    builder = make_builder(tasks["clinic"], runtime_config)
    (rollout,) = await builder.start_rollouts()
    try:
        env = builder.build_message_env(rollout)
        messages = await env.initial_observation()
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "submit" in messages[0]["content"]

        # write_file is confined to elt/.
        config_before = (rollout.ctx.host_workspace / "config.yaml").read_text()
        for path in ["config.yaml", "../escape.txt", "elt/../data_model.yaml", "/etc/elt_rl_x"]:
            step = await env.step(assistant_message([("write_file", {"path": path, "content": "x"})], 0))
            assert "error:" in step.next_messages[-1]["content"], path
        assert (rollout.ctx.host_workspace / "config.yaml").read_text() == config_before
        step = await env.step(
            assistant_message([("write_file", {"path": "elt/notes/a.txt", "content": "hello"})], 1)
        )
        assert (rollout.ctx.host_workspace / "elt" / "notes" / "a.txt").read_text() == "hello"

        # A command that overruns its budget is killed and reported, not hung.
        step = await env.step(
            assistant_message([("bash", {"command": "sleep 30", "timeout": 1})], 2)
        )
        assert "exit_code: 124" in step.next_messages[-1]["content"]

        # Long output is cut in the middle, keeping head and tail.
        step = await env.step(
            assistant_message([("bash", {"command": "seq 1 20000"})], 3)
        )
        output = step.next_messages[-1]["content"]
        assert len(output) < 2 * builder.env_config.max_output_chars
        assert "\n1\n2\n" in output and output.rstrip().endswith("20000")

        # submit ends the episode and says nothing about correctness.
        assert not step.episode_done
        step = await env.step(assistant_message(SUBMIT, 4))
        assert step.episode_done
        assert step.next_messages[-1]["content"] == "Submission received."
        assert step.reward == 0.0
    finally:
        await builder.cleanup()


async def test_turn_budget_ends_the_episode(tasks, runtime_config) -> None:
    builder = make_builder(tasks["clinic"], runtime_config, env_config=EnvConfig(max_turns=2))
    (rollout,) = await builder.start_rollouts()
    try:
        env = builder.build_message_env(rollout)
        await env.initial_observation()
        first = await env.step(assistant_message([("bash", {"command": "ls"})], 0))
        second = await env.step(assistant_message([("bash", {"command": "ls"})], 1))
        assert not first.episode_done and second.episode_done
        grade = await builder.grade_rollout(rollout)
        assert grade.reward == 0.0
    finally:
        await builder.cleanup()


def test_dataset_schedules_and_fresh_builders(tasks, runtime_config) -> None:
    import asyncio

    from elt_rl.env import ELTDatasetBuilder

    builder = ELTDatasetBuilder(
        tasks=list(tasks.values()),
        eval_tasks=[tasks["clinic"]],
        runtime_config=runtime_config,
        model_name="test/byte-level",
        renderer_name="qwen3_instruct",
        batch_size=3,
        group_size=2,
        start_stage="mixed",
        n_batches=4,
    )
    train, evaluation = asyncio.run(builder())
    assert len(train) == 4 and evaluation is not None and len(evaluation) == 1
    seen: dict[str, set[str]] = {}
    objects = []
    for index in range(len(train)):
        batch = train.get_batch(index)
        assert len(batch) == 3
        for group in batch:
            objects.append(group)
            seen.setdefault(group.task.name, set()).add(group.start_stage)
            assert f"elt_{group.start_stage}" in group.logging_tags()
    # Every group is its own builder object, and each task sees both stages.
    assert len({id(o) for o in objects}) == len(objects)
    assert all(stages == {"full", "transform"} for stages in seen.values())
    assert all(g.start_stage == "full" and g.group_size == 1 for g in evaluation.get_batch(0))
    # chz configs are frozen and hashable/picklable for multi-process rollouts.
    import pickle

    clone = pickle.loads(pickle.dumps(objects[0]))
    assert clone.task.name == objects[0].task.name and clone._rollouts == []
    assert chz.replace(runtime_config, verify_replay=False).verify_replay is False
    assert os.path.isdir(objects[0].task.input_dir)
