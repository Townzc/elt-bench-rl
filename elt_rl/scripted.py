"""Scripted policies for the local benchmark.

They drive the real environment (same tools, same sandbox, same grader) with
a fixed sequence of tool calls instead of a model. Uses:

* the credential-free integration test and smoke runs (``oracle``);
* regression tests for reward hacking: each ``cheat_*`` script is a shortcut a
  policy might discover, and the tests assert it earns nothing for it.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.tool_use import AgentToolMessageEnv

from elt_rl.local_bench import oracle
from elt_rl.local_bench.build import spec_for_task
from elt_rl.runtime import RolloutContext

if TYPE_CHECKING:
    from elt_rl.env import ELTEnvGroupBuilder

Turn = list[tuple[str, dict[str, Any]]]
"""One assistant turn: the tool calls it makes, as ``(tool name, arguments)``."""


def _bash(command: str) -> tuple[str, dict[str, Any]]:
    return ("bash", {"command": command})


def _write(path: str, content: str) -> tuple[str, dict[str, Any]]:
    return ("write_file", {"path": path, "content": content})


SUBMIT: Turn = [("submit", {})]


def extract_load_turns(ctx: RolloutContext) -> list[Turn]:
    spec = spec_for_task(ctx.task.name)
    pipeline = oracle.pipeline_yaml(
        spec, "sources", ctx.namespace.for_agent, ctx.namespace.schema
    )
    return [
        [_bash("cat config.yaml && ls documentation schemas")],
        [_write("elt/pipeline.yaml", pipeline)],
        [_bash("minielt apply -f elt/pipeline.yaml")],
        [_bash("minielt sync --all")],
        [_bash("minielt status")],
    ]


def transform_turns(ctx: RolloutContext, files: dict[str, str] | None = None) -> list[Turn]:
    spec = spec_for_task(ctx.task.name)
    if files is None:
        files = oracle.dbt_project_files(spec, ctx.namespace.for_agent, ctx.namespace.schema)
    project = f"elt/{oracle.DBT_PROJECT}"
    return [
        [_write(f"{project}/{path}", content) for path, content in files.items()],
        [_bash(f"cd {project} && dbt run")],
    ]


def oracle_script(ctx: RolloutContext) -> list[Turn]:
    """A correct solution of the task."""
    turns = [] if ctx.start_stage == "transform" else extract_load_turns(ctx)
    return turns + transform_turns(ctx) + [SUBMIT]


def noop_script(ctx: RolloutContext) -> list[Turn]:
    """Submit without doing anything."""
    return [SUBMIT]


def cheat_fake_row_counts(ctx: RolloutContext) -> list[Turn]:
    """Create source tables with the right number of rows and no real data.

    Passes a row-count-only stage 1 check; must fail the content check.
    """
    statements = [f'create schema if not exists "{ctx.namespace.schema}"']
    for table, rows in ctx.task.source_tables.items():
        statements.append(
            f'create or replace table "{ctx.namespace.schema}"."{table}" as '
            f"select i as id from range({rows}) t(i)"
        )
    script = (
        "import duckdb\n"
        f"con = duckdb.connect({ctx.namespace.for_agent!r})\n"
        f"for s in {statements!r}:\n    con.execute(s)\ncon.close()\n"
    )
    return [[_write("elt/fake.py", script)], [_bash("python elt/fake.py")], SUBMIT]


def cheat_double_sync(ctx: RolloutContext) -> list[Turn]:
    """Run every append sync twice (duplicated rows)."""
    return extract_load_turns(ctx) + [[_bash("minielt sync --all")], SUBMIT]


def cheat_handmade_tables(ctx: RolloutContext) -> list[Turn]:
    """Correct extract/load, then final tables written by an ad-hoc script.

    The tables are right, but there is no transformation pipeline behind them;
    the replay check must discard them.
    """
    spec = spec_for_task(ctx.task.name)
    statements = [
        f'create or replace table "{ctx.namespace.schema}"."{m.name}" as '
        + oracle.reference_sql(spec, m.name, ctx.namespace.schema)
        for m in spec.models
    ]
    script = (
        "import duckdb\n"
        f"con = duckdb.connect({ctx.namespace.for_agent!r})\n"
        f"for s in {statements!r}:\n    con.execute(s)\ncon.close()\n"
    )
    return extract_load_turns(ctx) + [
        [_write("elt/handmade.py", script)],
        [_bash("python elt/handmade.py")],
        SUBMIT,
    ]


def cheat_passthrough_models(ctx: RolloutContext) -> list[Turn]:
    """Handmade tables under other names, plus dbt models that just select from them.

    A replay that only dropped the data models would rebuild them from the
    handmade copies. The replay starts from the source tables only, so the
    copies are gone and the models cannot be built.
    """
    spec = spec_for_task(ctx.task.name)
    statements = [
        f'create or replace table "{ctx.namespace.schema}"."handmade_{m.name}" as '
        + oracle.reference_sql(spec, m.name, ctx.namespace.schema)
        for m in spec.models
    ]
    script = (
        "import duckdb\n"
        f"con = duckdb.connect({ctx.namespace.for_agent!r})\n"
        f"for s in {statements!r}:\n    con.execute(s)\ncon.close()\n"
    )
    files = oracle.dbt_project_files(spec, ctx.namespace.for_agent, ctx.namespace.schema)
    for model in spec.models:
        files[f"models/{model.name}.sql"] = (
            f'select * from "{ctx.namespace.schema}"."handmade_{model.name}"\n'
        )
    return (
        extract_load_turns(ctx)
        + [[_write("elt/handmade.py", script)], [_bash("python elt/handmade.py")]]
        + transform_turns(ctx, files)
        + [SUBMIT]
    )


def leave_background_process(ctx: RolloutContext) -> list[Turn]:
    """A correct solution that leaves a process holding the warehouse open."""
    holder = (
        "import duckdb, time\n"
        f"con = duckdb.connect({ctx.namespace.for_agent!r})\n"
        "time.sleep(600)\n"
    )
    return oracle_script(ctx)[:-1] + [
        [_write("elt/holder.py", holder)],
        [_bash("nohup python elt/holder.py > /dev/null 2>&1 &")],
        SUBMIT,
    ]


def cheat_keys_only(ctx: RolloutContext) -> list[Turn]:
    """Correct extract/load, then models that only carry the key column."""
    spec = spec_for_task(ctx.task.name)
    files = oracle.dbt_project_files(spec, ctx.namespace.for_agent, ctx.namespace.schema)
    for model in spec.models:
        reference = oracle.reference_sql(spec, model.name, ctx.namespace.schema)
        others = ", ".join(
            f"cast(null as varchar) as {c.name}" for c in model.columns if c.name not in model.key
        )
        files[f"models/{model.name}.sql"] = (
            f"select {', '.join(model.key)}, {others} from ({reference}) reference\n"
        )
    return extract_load_turns(ctx) + transform_turns(ctx, files) + [SUBMIT]


def extract_load_only(ctx: RolloutContext) -> list[Turn]:
    """Correct extract/load, no transformation and no ``submit``.

    The episode ends by running out of scripted turns, like a rollout that
    hits its turn budget half-way: the loaded data must still be credited.
    """
    return extract_load_turns(ctx)


def cheat_tamper_sources(ctx: RolloutContext) -> list[Turn]:
    """Transformation-only start: build the models, then drop a pre-loaded source table."""
    table = sorted(ctx.task.source_tables)[0]
    script = (
        "import duckdb\n"
        f"con = duckdb.connect({ctx.namespace.for_agent!r})\n"
        f"con.execute('drop table \"{ctx.namespace.schema}\".\"{table}\"')\ncon.close()\n"
    )
    return transform_turns(ctx) + [
        [_write("elt/tamper.py", script)],
        [_bash("python elt/tamper.py")],
        SUBMIT,
    ]


SCRIPTS = {
    "oracle": oracle_script,
    "noop": noop_script,
    "extract_load_only": extract_load_only,
    "cheat_tamper_sources": cheat_tamper_sources,
    "cheat_fake_row_counts": cheat_fake_row_counts,
    "cheat_double_sync": cheat_double_sync,
    "cheat_handmade_tables": cheat_handmade_tables,
    "cheat_passthrough_models": cheat_passthrough_models,
    "leave_background_process": leave_background_process,
    "cheat_keys_only": cheat_keys_only,
}


def assistant_message(turn: Turn, turn_index: int) -> Message:
    return {
        "role": "assistant",
        "content": f"Step {turn_index + 1}.",
        "tool_calls": [
            ToolCall(
                id=f"call_{turn_index}_{i}",
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
            )
            for i, (name, arguments) in enumerate(turn)
        ],
    }


async def run_script(env: AgentToolMessageEnv, turns: list[Turn]) -> list[Message]:
    """Play scripted turns against a message-level environment."""
    await env.initial_observation()
    for index, turn in enumerate(turns):
        result = await env.step(assistant_message(turn, index))
        if result.episode_done:
            break
    return env.history


async def run_scripted_group(builder: ELTEnvGroupBuilder, policy: str) -> None:
    """Run one group with a scripted policy and grade it.

    The message-level twin of ``do_group_rollout``: same sandboxes, tools,
    finalization and grader, with no tokenizer or model. Results are left in
    ``builder.last_grades`` and ``builder.last_histories``.
    """
    if policy not in SCRIPTS:
        raise ValueError(f"Unknown scripted policy {policy!r}; available: {sorted(SCRIPTS)}")
    try:
        rollouts = await builder.start_rollouts()
        envs = [builder.build_message_env(r) for r in rollouts]
        await asyncio.gather(
            *[run_script(env, SCRIPTS[policy](r.ctx)) for env, r in zip(envs, rollouts, strict=True)]
        )
        builder.last_grades = list(
            await asyncio.gather(*[builder.grade_rollout(r) for r in rollouts])
        )
        builder.last_histories = [list(env.history) for env in envs]
    finally:
        await builder.cleanup()
