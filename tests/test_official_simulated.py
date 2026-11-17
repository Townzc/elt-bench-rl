"""The official code path, with the credentialed services replaced by fakes.

Covers what can be checked without Airbyte, Docker and Snowflake: loading an
ELT-Bench style checkout, preparing a credential-free workspace pointed at a
per-rollout schema, the namespace lifecycle statements, grading through the
Snowflake adapter (upper-case identifiers, Airbyte metadata columns), and
teardown. See ``fake_official.py`` for what is simulated.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from elt_rl.env import ELTEnvGroupBuilder
from elt_rl.runtime import OfficialRuntimeConfig
from elt_rl.sandbox.docker import DockerSandbox
from elt_rl.scripted import SUBMIT, Turn, run_script
from elt_rl.scripts.common import BackendConfig, resolve_backend

from .fake_official import (
    SOURCES,
    TASK,
    FakeAgentMachine,
    FakeSnowflake,
    with_airbyte_columns,
    write_official_tree,
)

FULL_TURNS: list[Turn] = [
    [("bash", {"command": "cat config.yaml"})],
    [("write_file", {"path": "elt/main.tf", "content": "# sources, destination, connections\n"})],
    [("bash", {"command": "cd elt && terraform init && terraform apply -auto-approve"})],
    [("write_file", {"path": "elt/shop_dbt/dbt_project.yml", "content": "name: shop_dbt\n"})],
    [("bash", {"command": "cd elt/shop_dbt && dbt run"})],
    SUBMIT,
]
TRANSFORM_TURNS = FULL_TURNS[3:]


class Stack:
    """A fake ELT-Bench installation and the handles a test needs."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = write_official_tree(root / "ELT-Bench")
        self.runs_dir = root / "runs"
        self.warehouse = FakeSnowflake()
        self.machines: list[FakeAgentMachine] = []
        self.corrupt_load = False

        # `import snowflake.connector` resolves to the fake account.
        package = types.ModuleType("snowflake")
        connector = types.ModuleType("snowflake.connector")
        connector.connect = self.warehouse.connect  # type: ignore[attr-defined]
        package.connector = connector  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "snowflake", package)
        monkeypatch.setitem(sys.modules, "snowflake.connector", connector)

        async def create(**kwargs) -> FakeAgentMachine:
            machine = FakeAgentMachine(
                self.warehouse, kwargs["host_workspace"], kwargs, corrupt_load=self.corrupt_load
            )
            self.machines.append(machine)
            return machine

        monkeypatch.setattr(DockerSandbox, "create", staticmethod(create))

    def load_reference(self) -> None:
        """A known-good load of the source tables, as a reference database."""
        self.warehouse.schemas[("ELT_REFERENCE", TASK.upper())] = {
            table: with_airbyte_columns(frame, sync="reference") for table, frame in SOURCES.items()
        }

    def builder(self, start_stage="full", group_size=1, **backend) -> ELTEnvGroupBuilder:
        config = BackendConfig(
            backend="official",
            elt_bench_root=str(self.root),
            runs_dir=str(self.runs_dir),
            **backend,
        )
        tasks, runtime_config = resolve_backend(config)
        assert isinstance(runtime_config, OfficialRuntimeConfig)
        assert [t.name for t in tasks] == [TASK]
        return ELTEnvGroupBuilder(
            task=tasks[0],
            runtime_config=runtime_config,
            model_name="test/byte-level",
            group_size=group_size,
            start_stage=start_stage,
        )


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stack:
    return Stack(tmp_path, monkeypatch)


async def run(builder: ELTEnvGroupBuilder, turns: list[Turn]):
    rollouts = await builder.start_rollouts()
    try:
        for rollout in rollouts:
            await run_script(builder.build_message_env(rollout), turns)
        return rollouts, [await builder.grade_rollout(r) for r in rollouts]
    finally:
        await builder.cleanup()


def test_loader_reads_an_official_checkout(stack: Stack) -> None:
    task = stack.builder().task
    assert task.destination == "snowflake"
    assert task.input_dir == stack.root / "inputs" / TASK
    assert task.source_tables == {"customers": 3, "orders": 4}
    assert task.models == ("customer_orders",)  # from the ground truth, not the YAML
    assert task.sort_keys == {"customer_orders": ["customer_id"]}
    assert task.gt_dir == stack.root / "ground_truth" / "gt_snowflake" / TASK


async def test_full_rollout_on_the_official_path(stack: Stack) -> None:
    stack.load_reference()
    builder = stack.builder(reference_database="ELT_REFERENCE", group_size=2)
    rollouts = await builder.start_rollouts()
    try:
        schemas = [r.ctx.namespace.schema for r in rollouts]
        assert len(set(schemas)) == 2 and all(s.startswith("SHOP__") for s in schemas)
        for rollout, machine in zip(rollouts, stack.machines, strict=True):
            ctx = rollout.ctx
            # Container: benchmark image and network, only elt/ writable.
            assert machine.create_kwargs["image"] == "elt-swe"
            assert machine.create_kwargs["network"] == "elt-docker_elt_network"
            assert machine.create_kwargs["writable_subdirs"] == ("elt",)
            assert ctx.workspace == "/workspace"
            # Workspace: the bundle minus credential files, config pointed at this rollout.
            names = {p.name for p in ctx.host_workspace.rglob("*")}
            assert {"config.yaml", "data_model.yaml", "main.tf", "check_job_status.py"} <= names
            assert "snowflake_credential.json" not in names
            assert "customer_orders.csv" not in names
            assert "admin-secret" not in "".join(
                p.read_text() for p in ctx.host_workspace.rglob("*") if p.is_file()
            )
            block = yaml.safe_load((ctx.host_workspace / "config.yaml").read_text())["snowflake"]
            assert block["config"]["database"] == "ELT_RL"
            assert block["config"]["schema"] == ctx.namespace.schema
            assert block["config"]["account"] == "acme-xy12345"  # the rest is untouched
            prompt = builder.initial_messages(rollout)
            assert "Airbyte" in prompt[0]["content"] and "terraform apply" in prompt[1]["content"]
            assert "/workspace/elt/main.tf" in prompt[1]["content"]

        # Only the first rollout does the task.
        await run_script(builder.build_message_env(rollouts[0]), FULL_TURNS)
        grades = [await builder.grade_rollout(r) for r in rollouts]
    finally:
        await builder.cleanup()

    assert grades[0].reward == pytest.approx(1.0), grades[0].summary()
    assert grades[0].metrics["elt/task_success"] == 1.0
    assert all(s.content_ok is True for s in grades[0].sources)  # Airbyte columns ignored
    assert grades[1].reward == 0.0  # same task, same database, separate schema

    statements = stack.warehouse.statements
    for schema in schemas:
        assert f'CREATE SCHEMA "ELT_RL"."{schema}"' in statements
        assert f'GRANT OWNERSHIP ON SCHEMA "ELT_RL"."{schema}" TO ROLE "AIRBYTE_ROLE"' in statements
        # Teardown: schema dropped again, Airbyte resources destroyed, workspace gone.
        assert statements.count(f'DROP SCHEMA IF EXISTS "ELT_RL"."{schema}" CASCADE') == 2
        assert ("ELT_RL", schema) not in stack.warehouse.schemas
    for machine in stack.machines:
        assert machine.cleaned_up
        assert "terraform destroy" in machine.commands[-1][0]
        assert machine.commands[-1][1] == "/workspace/elt"
        assert not machine.host_workspace.exists()
    # The admin credential came from the checkout's setup directory.
    assert stack.warehouse.connect_kwargs[0]["user"] == "ADMIN"


async def test_transform_only_rollout_clones_the_reference(stack: Stack) -> None:
    stack.load_reference()
    builder = stack.builder(start_stage="transform", reference_database="ELT_REFERENCE")
    rollouts, (grade,) = await run(builder, TRANSFORM_TURNS)
    schema = rollouts[0].ctx.namespace.schema
    assert f'CREATE SCHEMA "ELT_RL"."{schema}" CLONE "ELT_REFERENCE"."SHOP"' in (
        stack.warehouse.statements
    )
    assert grade.reward == pytest.approx(1.0), grade.summary()
    (machine,) = stack.machines
    # No source systems or Airbyte settings in a transformation-only workspace...
    assert not any("terraform" in command for command, _ in machine.commands)
    # ...and the reference load itself was never modified.
    assert set(stack.warehouse.schemas[("ELT_REFERENCE", "SHOP")]) == {"CUSTOMERS", "ORDERS"}


async def test_egress_proxy_is_passed_to_the_container(stack: Stack) -> None:
    builder = stack.builder(network="elt-rl-internal", egress_proxy="http://elt-egress:3128")
    await builder.start_rollouts()
    try:
        (machine,) = stack.machines
        assert machine.create_kwargs["network"] == "elt-rl-internal"
        env = machine.create_kwargs["env"]
        assert env["HTTPS_PROXY"] == env["https_proxy"] == "http://elt-egress:3128"
        assert "elt-postgres" in env["NO_PROXY"] and "airbyte-abctl-control-plane" in env["no_proxy"]
    finally:
        await builder.cleanup()
    # Without a proxy the container environment is left alone.
    plain = stack.builder()
    await plain.start_rollouts()
    try:
        assert stack.machines[-1].create_kwargs["env"] == {}
    finally:
        await plain.cleanup()


async def test_transform_only_needs_a_reference(stack: Stack) -> None:
    builder = stack.builder(start_stage="transform")
    with pytest.raises(ValueError, match="reference_database"):
        await builder.start_rollouts()


async def test_content_check_on_the_official_path(stack: Stack) -> None:
    """A load with the right row counts and wrong data passes the official stage 1 only."""
    stack.load_reference()
    stack.corrupt_load = True
    _, (grade,) = await run(stack.builder(reference_database="ELT_REFERENCE"), FULL_TURNS)
    assert grade.metrics["official/stage1_success"] == 1.0
    assert grade.metrics["elt/el_frac"] == 0.5
    assert grade.metrics["elt/content_mismatch"] == 1.0
    assert grade.metrics["elt/model_frac"] == 0.0  # the model was computed from bad data


async def test_without_a_reference_stage1_is_the_official_row_count_check(stack: Stack) -> None:
    _, (grade,) = await run(stack.builder(), FULL_TURNS)
    assert grade.reward == pytest.approx(1.0)
    assert all(s.content_ok is None for s in grade.sources)


async def test_replay_check_on_the_official_path(stack: Stack) -> None:
    builder = stack.builder(verify_replay=True)
    _, (grade,) = await run(builder, FULL_TURNS)
    assert grade.metrics["elt/replay_ok"] == 1.0 and grade.reward == pytest.approx(1.0)
    (machine,) = stack.machines
    replays = [c for c in machine.commands if c == ("dbt run", "/workspace/elt/shop_dbt")]
    assert len(replays) == 1

    # Extract/load only, then nothing: no dbt project to replay, no model credit.
    _, (grade,) = await run(stack.builder(verify_replay=True), FULL_TURNS[:3] + [SUBMIT])
    assert grade.metrics["elt/replay_ok"] == 0.0
    assert grade.reward == pytest.approx(0.3)


async def test_unreachable_warehouse_fails_the_group_instead_of_scoring_zero(stack: Stack) -> None:
    builder = stack.builder()
    (rollout,) = await builder.start_rollouts()
    try:
        def down(sql: str):
            raise ConnectionError("warehouse unreachable")

        stack.warehouse.execute = down  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="Grading infrastructure failure"):
            await builder.grade_rollout(rollout)
    finally:
        del stack.warehouse.execute
        await builder.cleanup()
