"""Local runtime: hermetic, credential-free rollouts on DuckDB.

Source systems are files copied into each rollout's workspace, extract/load
is done with ``minielt``, and the warehouse is a DuckDB file. Nothing outside
the workspace is touched, so rollouts are fully isolated from each other and
can run with no network at all.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Literal

import chz
from tinker_cookbook import cli_utils

from elt_rl import prompts
from elt_rl.destinations import DuckDBDestination, Namespace
from elt_rl.grading.grader import StartStage
from elt_rl.runtime.base import ELTRuntime, RolloutContext
from elt_rl.sandbox.docker import CONTAINER_WORKDIR, DockerSandbox
from elt_rl.sandbox.local import LocalProcessSandbox
from elt_rl.tasks import ELTTask

logger = logging.getLogger(__name__)

MINIELT_CLI = Path(__file__).resolve().parents[1] / "minielt" / "cli.py"


@chz.chz
class LocalRuntimeConfig:
    """Configuration of the local runtime (picklable; build with ``build()``)."""

    runs_dir: str | None = None
    """Where rollout workspaces are created. Defaults under the tinker runs dir."""
    sandbox: Literal["process", "docker"] = "process"
    """``process`` runs commands as host subprocesses (tests, trusted use).
    ``docker`` runs them in a container with no network (use this to train)."""
    docker_image: str = "elt-rl-local"
    command_timeout: int = 120
    verify_replay: bool = True
    """Re-run the submitted dbt project before grading (see ``replay_transform``)."""
    keep_workspaces: bool = False
    """Keep workspaces after the rollout, for debugging."""

    def build(self) -> LocalRuntime:
        return LocalRuntime(self)


class LocalRuntime(ELTRuntime):
    def __init__(self, config: LocalRuntimeConfig) -> None:
        self.config = config
        self.destination = DuckDBDestination()
        self.command_timeout = config.command_timeout
        self.verify_replay = config.verify_replay
        self.runs_dir = Path(
            config.runs_dir or cli_utils.runs_path("elt_rl", "workspaces")
        ).expanduser()

    # -- prompts -------------------------------------------------------------------
    def system_prompt(self, ctx: RolloutContext) -> str:
        return prompts.render_system_prompt(
            workspace=ctx.workspace,
            destination=self.destination.display_name,
            transform_only=ctx.start_stage == "transform",
            source_kinds="a relational database, a document store, an object store, and flat files",
            el_tool="minielt",
            tooling="minielt, dbt, python",
        )

    def instructions(self, ctx: RolloutContext) -> str:
        template = (
            prompts.LOCAL_TRANSFORM_INSTRUCTION
            if ctx.start_stage == "transform"
            else prompts.LOCAL_FULL_INSTRUCTION
        )
        text = template.format(
            workspace=ctx.workspace,
            destination=self.destination.display_name,
            destination_key=self.destination.name,
        )
        return text + (prompts.REPLAY_NOTE if self.verify_replay else "")

    # -- lifecycle -------------------------------------------------------------------
    def reference_namespace(self, task: ELTTask) -> Namespace | None:
        reference_db = task.extras.get("reference_db")
        if reference_db is None:
            return None
        return Namespace(database=str(reference_db), schema=task.name)

    def _prepare(self, task: ELTTask, rollout_id: str, start_stage: StartStage) -> tuple:
        host_workspace = self.runs_dir / f"{task.name}-{rollout_id}"
        if host_workspace.exists():
            shutil.rmtree(host_workspace)
        agent_workspace = (
            CONTAINER_WORKDIR if self.config.sandbox == "docker" else str(host_workspace)
        )
        ns = self.destination.new_namespace(
            task.name, rollout_id, str(host_workspace), agent_workspace
        )
        reference = self.reference_namespace(task)
        self.prepare_workspace(task, host_workspace, ns, start_stage)
        if start_stage == "transform":
            if reference is None:
                raise ValueError(f"Task {task.name} has no reference load to start from")
            self.destination.create_namespace(ns, clone_from=reference)
        else:
            self.destination.create_namespace(ns)
            # Each rollout reads its own copy of the source systems, so one
            # rollout cannot disturb another by modifying them.
            shutil.copytree(task.extras["sources_dir"], host_workspace / "sources")

        # `minielt` on PATH without requiring this package to be installed.
        bin_dir = host_workspace / ".bin"
        bin_dir.mkdir(exist_ok=True)
        if self.config.sandbox == "process":
            shim = bin_dir / "minielt"
            shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{MINIELT_CLI}" "$@"\n')
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return host_workspace, agent_workspace, ns, reference, bin_dir

    async def start(
        self, task: ELTTask, rollout_id: str, start_stage: StartStage
    ) -> RolloutContext:
        host_workspace, agent_workspace, ns, reference, bin_dir = await asyncio.to_thread(
            self._prepare, task, rollout_id, start_stage
        )
        if self.config.sandbox == "docker":
            sandbox = await DockerSandbox.create(
                image=self.config.docker_image,
                host_workspace=host_workspace,
                network=None,  # hermetic: nothing to download, nothing to exfiltrate
                writable_subdirs=("elt", "warehouse"),
            )
        else:
            sandbox = LocalProcessSandbox(
                host_workspace,
                env={"PATH": os.pathsep.join([str(bin_dir), _base_path()])},
            )
        return RolloutContext(
            task=task,
            rollout_id=rollout_id,
            start_stage=start_stage,
            sandbox=sandbox,
            namespace=ns,
            workspace=agent_workspace,
            host_workspace=host_workspace,
            reference=reference,
        )

    async def stop(self, ctx: RolloutContext) -> None:
        try:
            await ctx.sandbox.cleanup()
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", ctx.rollout_id, e)
        if not self.config.keep_workspaces:
            await asyncio.to_thread(shutil.rmtree, ctx.host_workspace, True)


def _base_path() -> str:
    return os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")])
