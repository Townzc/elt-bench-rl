"""Shared fixtures: a locally built benchmark and helpers to run scripted rollouts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from elt_rl.env import ELTEnvGroupBuilder, EnvConfig
from elt_rl.grading.grader import GradeResult, RewardConfig, StartStage
from elt_rl.local_bench.build import build_local_bench
from elt_rl.runtime import LocalRuntimeConfig
from elt_rl.scripted import run_scripted_group
from elt_rl.tasks import ELTTask, load_tasks

MODEL_NAME = "test/byte-level"


@pytest.fixture(scope="session")
def bench_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The credential-free benchmark, generated once per test session."""
    return build_local_bench(tmp_path_factory.mktemp("local_bench"), seed=0)


@pytest.fixture(scope="session")
def tasks(bench_root: Path) -> dict[str, ELTTask]:
    return {t.name: t for t in load_tasks(bench_root, "duckdb")}


@pytest.fixture
def runtime_config(tmp_path: Path) -> LocalRuntimeConfig:
    return LocalRuntimeConfig(runs_dir=str(tmp_path / "workspaces"))


def make_builder(
    task: ELTTask,
    runtime_config: LocalRuntimeConfig,
    *,
    start_stage: StartStage = "full",
    group_size: int = 1,
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> ELTEnvGroupBuilder:
    return ELTEnvGroupBuilder(
        task=task,
        runtime_config=runtime_config,
        model_name=MODEL_NAME,
        renderer_name="qwen3_instruct",
        group_size=group_size,
        start_stage=start_stage,
        env_config=env_config,
        reward_config=reward_config,
    )


async def run_policy(
    task: ELTTask,
    runtime_config: LocalRuntimeConfig,
    policy: str,
    **builder_kwargs,
) -> list[GradeResult]:
    """Run one group with a scripted policy; return the verified grades."""
    builder = make_builder(task, runtime_config, **builder_kwargs)
    await run_scripted_group(builder, policy)
    return builder.last_grades


def have(tool: str) -> bool:
    return shutil.which(tool) is not None
