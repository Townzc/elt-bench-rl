"""Shared CLI plumbing: turn backend options into tasks and a runtime config."""

from __future__ import annotations

from typing import Literal

import chz

from elt_rl.local_bench.build import DEFAULT_ROOT, build_local_bench
from elt_rl.runtime import LocalRuntimeConfig, OfficialRuntimeConfig, RuntimeConfig
from elt_rl.tasks import ELTTask, load_tasks

Backend = Literal["local", "official"]


@chz.chz
class BackendConfig:
    """Which tasks to run and on what infrastructure."""

    backend: Backend = "local"
    """``local``: credential-free DuckDB tasks. ``official``: ELT-Bench tasks
    on Airbyte + a cloud warehouse (needs the ELT-Bench setup and credentials)."""
    tasks: str | None = None
    """Comma-separated task names; default is every task found."""
    runs_dir: str | None = None
    """Where rollout workspaces are created."""
    keep_workspaces: bool = False

    # -- local backend ---------------------------------------------------------
    local_bench_root: str = str(DEFAULT_ROOT)
    local_seed: int = 0
    local_variants: int = 1
    """Data variants per local task specification (more distinct tasks)."""
    local_sandbox: Literal["process", "docker"] = "process"
    local_docker_image: str = "elt-rl-local"
    verify_replay: bool | None = None
    """Re-run the submitted dbt project before grading. Default: on for the
    local backend, off for the official one."""

    # -- official backend ------------------------------------------------------
    elt_bench_root: str | None = None
    """Path of the ELT-Bench checkout, after its setup has been run."""
    destination: str = "snowflake"
    credential_path: str | None = None
    """Admin credential JSON; default ``<elt_bench_root>/setup/destination/<destination>_credential.json``."""
    database: str = "ELT_RL"
    agent_role: str | None = "AIRBYTE_ROLE"
    reference_database: str | None = None
    """Database with a known-good load of the source tables (one schema per
    task). Needed for transformation-only rollouts and the content check."""
    gt_dir: str | None = None
    """Ground-truth CSV root; default ``<elt_bench_root>/ground_truth/gt_<destination>``."""
    image: str = "elt-swe"
    network: str = "elt-docker_elt_network"
    egress_proxy: str | None = None
    """Allow-listing forward proxy for rollout containers (see README, "Network egress")."""
    command_timeout: int | None = None


def resolve_backend(config: BackendConfig) -> tuple[list[ELTTask], RuntimeConfig]:
    """Load the tasks and build the runtime configuration for a backend."""
    only = [t.strip() for t in config.tasks.split(",") if t.strip()] if config.tasks else None
    if config.backend == "local":
        root = build_local_bench(
            config.local_bench_root, seed=config.local_seed, variants=config.local_variants
        )
        tasks = load_tasks(root, "duckdb", only=only)
        runtime: RuntimeConfig = LocalRuntimeConfig(
            runs_dir=config.runs_dir,
            sandbox=config.local_sandbox,
            docker_image=config.local_docker_image,
            command_timeout=config.command_timeout or 120,
            verify_replay=True if config.verify_replay is None else config.verify_replay,
            keep_workspaces=config.keep_workspaces,
        )
        return tasks, runtime

    if not config.elt_bench_root:
        raise ValueError("backend=official needs elt_bench_root=<path of the ELT-Bench checkout>")
    tasks = load_tasks(config.elt_bench_root, config.destination, gt_dir=config.gt_dir, only=only)
    credential = config.credential_path or (
        f"{config.elt_bench_root}/setup/destination/{config.destination}_credential.json"
    )
    runtime = OfficialRuntimeConfig(
        destination=config.destination,
        credential_path=credential,
        database=config.database,
        agent_role=config.agent_role,
        reference_database=config.reference_database,
        image=config.image,
        network=config.network,
        egress_proxy=config.egress_proxy,
        runs_dir=config.runs_dir,
        command_timeout=config.command_timeout or 600,
        verify_replay=bool(config.verify_replay),
        keep_workspaces=config.keep_workspaces,
    )
    return tasks, runtime
