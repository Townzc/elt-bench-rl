"""Checks that need credentials or a Docker daemon. Skipped when they are absent.

    pytest -m tinker     # TINKER_API_KEY: sample from a model, one optimization step
    pytest -m docker     # Docker daemon and the elt-rl-local image
    pytest -m snowflake  # the ELT-Bench stack, see README ("Official ELT-Bench tasks")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from elt_rl.env import EnvConfig
from elt_rl.runtime import LocalRuntimeConfig

from .conftest import run_policy

needs_tinker = pytest.mark.skipif(
    not os.environ.get("TINKER_API_KEY"), reason="TINKER_API_KEY is not set"
)


def _docker_image_available(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@pytest.mark.tinker
@needs_tinker
async def test_one_rl_optimization_step_on_execution_rewards(bench_root: Path, tmp_path: Path) -> None:
    """Sample rollouts from a Tinker model, grade them by execution, take one optimizer step."""
    from elt_rl.scripts.train import CLIConfig, cli_main

    log_path = tmp_path / "log"
    config = CLIConfig(
        backend="local",
        local_bench_root=str(bench_root),
        runs_dir=str(tmp_path / "workspaces"),
        tasks="bookshop,clinic",
        start_stage="transform",  # short episodes: only the dbt project has to be written
        group_size=2,
        groups_per_batch=2,
        max_steps=1,
        eval_every=0,
        save_every=0,
        max_tokens=2048,
        env=EnvConfig(max_turns=12),
        log_path=str(log_path),
        behavior_if_log_dir_exists="delete",
    )
    await cli_main(config)
    rows = [json.loads(line) for line in (log_path / "metrics.jsonl").read_text().splitlines()]
    keys = {key for row in rows for key in row}
    assert any(key.endswith("elt/task_success") for key in keys), sorted(keys)
    assert any(key.endswith("reward/total") for key in keys), sorted(keys)
    assert any(key.startswith("optim/") for key in keys), sorted(keys)


@pytest.mark.tinker
@needs_tinker
async def test_tinker_model_rollout_is_graded(bench_root: Path, tmp_path: Path) -> None:
    from elt_rl.scripts.rollout import RolloutCLIConfig, run

    records = await run(
        RolloutCLIConfig(
            backend="local",
            local_bench_root=str(bench_root),
            runs_dir=str(tmp_path / "workspaces"),
            output_dir=str(tmp_path / "out"),
            tasks="bookshop",
            policy="tinker",
            start_stage="transform",
            max_tokens=2048,
            env=EnvConfig(max_turns=12),
        )
    )
    assert len(records) == 1 and 0.0 <= records[0]["reward"] <= 1.0
    assert (tmp_path / "out" / "bookshop_0_transcript.json").is_file()


@pytest.mark.docker
@pytest.mark.skipif(
    not _docker_image_available("elt-rl-local"),
    reason="needs Docker and the elt-rl-local image (docker/local.Dockerfile)",
)
@pytest.mark.parametrize("start_stage", ["full", "transform"])
async def test_oracle_in_the_docker_sandbox(tasks, tmp_path: Path, start_stage) -> None:
    """The local tasks in a container with no network and a read-only task directory."""
    config = LocalRuntimeConfig(runs_dir=str(tmp_path / "workspaces"), sandbox="docker")
    (grade,) = await run_policy(tasks["bookshop"], config, "oracle", start_stage=start_stage)
    assert grade.reward == pytest.approx(1.0), grade.summary()
    assert config.sandbox == "docker"
