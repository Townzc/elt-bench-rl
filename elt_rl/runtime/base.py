"""Runtime: everything stateful behind one rollout.

An ELT episode is stateful in three places: the workspace filesystem
(Terraform state, the dbt project), the extract/load control plane (Airbyte
sources, connections and jobs) and the warehouse namespace (raw and final
tables). A runtime owns that state: it creates a fresh, isolated copy for each
rollout, and destroys it afterwards.

A runtime is destination-agnostic. It talks to the warehouse only through the
:class:`~elt_rl.destinations.base.Destination` adapter.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tinker_cookbook.sandbox import SandboxInterface

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.grading.grader import StartStage
from elt_rl.tasks import ELTTask

logger = logging.getLogger(__name__)


@dataclass
class RolloutContext:
    """The live state of one rollout."""

    task: ELTTask
    rollout_id: str
    start_stage: StartStage
    sandbox: SandboxInterface
    namespace: Namespace
    workspace: str
    """Workspace path as the agent sees it."""
    host_workspace: Path
    reference: Namespace | None = None
    """Reference load of the source tables (content check, clone source)."""
    replay_ok: bool | None = None
    """Outcome of the replay check; None when it did not run."""


def new_rollout_id() -> str:
    return uuid.uuid4().hex[:10]


class ELTRuntime(ABC):
    """Creates, finalizes and destroys rollout state for one destination."""

    destination: Destination
    command_timeout: int = 120
    verify_replay: bool = False
    replay_timeout: int = 600

    # -- prompts -----------------------------------------------------------------
    @abstractmethod
    def system_prompt(self, ctx: RolloutContext) -> str: ...

    @abstractmethod
    def instructions(self, ctx: RolloutContext) -> str: ...

    # -- lifecycle ----------------------------------------------------------------
    @abstractmethod
    async def start(
        self, task: ELTTask, rollout_id: str, start_stage: StartStage
    ) -> RolloutContext:
        """Create an isolated workspace, sandbox and warehouse namespace."""

    @abstractmethod
    async def stop(self, ctx: RolloutContext) -> None:
        """Release everything ``start`` created. Must not raise."""

    @abstractmethod
    def reference_namespace(self, task: ELTTask) -> Namespace | None:
        """Where a known-good load of the task's source tables lives, if any."""

    # -- shared helpers -----------------------------------------------------------
    def prepare_workspace(
        self, task: ELTTask, host_workspace: Path, ns: Namespace, start_stage: StartStage
    ) -> None:
        """Copy the agent-visible bundle and point ``config.yaml`` at ``ns``.

        Credential files and grading assets are never copied. A
        transformation-only workspace keeps only what that stage needs, as in
        the official transformation-only setting: the data model, the source
        schemas, the destination section of the config and its documentation.
        """
        destination = self.destination.name
        host_workspace.mkdir(parents=True, exist_ok=True)
        transform_only = start_stage == "transform"
        for item in task.input_dir.iterdir():
            if item.name.endswith("_credential.json"):
                continue
            if transform_only and item.name not in ("data_model.yaml", "schemas", "config.yaml"):
                continue
            target = host_workspace / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        if transform_only:
            docs = task.input_dir / "documentation"
            for doc in sorted(docs.glob(f"destination_{destination}*")) if docs.is_dir() else []:
                (host_workspace / "documentation").mkdir(exist_ok=True)
                shutil.copy2(doc, host_workspace / "documentation" / doc.name)
        (host_workspace / "elt").mkdir(exist_ok=True)

        config_path = host_workspace / "config.yaml"
        with config_path.open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        try:
            base = config[destination]["config"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"{task.input_dir}/config.yaml has no {destination}.config") from e
        config[destination]["config"] = self.destination.agent_config(ns, base)
        if transform_only:
            config = {destination: config[destination]}
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    async def finalize(self, ctx: RolloutContext) -> None:
        """Run post-episode checks that need the sandbox, before grading."""
        # Nothing the agent started may outlive the episode: a lingering process
        # could hold a warehouse lock, or keep changing tables while they are graded.
        kill_background = getattr(ctx.sandbox, "kill_background", None)
        if kill_background is not None:
            try:
                await kill_background()
            except Exception as e:
                logger.warning("Could not stop background processes of %s: %s", ctx.rollout_id, e)
        if self.verify_replay:
            ctx.replay_ok = await self.replay_transform(ctx)

    async def replay_transform(self, ctx: RolloutContext) -> bool:
        """Rebuild the data models from the submitted dbt project.

        Every relation in the rollout's namespace except the source tables is
        dropped, and ``dbt run`` is executed once more in the project the
        agent left behind. What is graded is therefore what the submitted
        *pipeline* produces from the loaded sources: tables materialized by
        hand, helper tables a model secretly reads from, and tables that no
        longer correspond to the submitted code do not survive.
        Returns whether a project was found and the run succeeded.
        """
        elt_dir = f"{ctx.workspace}/elt"
        found = await ctx.sandbox.run_command(
            f"find {elt_dir} -maxdepth 4 -name dbt_project.yml "
            "-not -path '*/dbt_packages/*' -not -path '*/target/*' | sort | head -n 1",
            workdir=ctx.workspace,
            timeout=60,
        )
        project_file = found.stdout.strip().splitlines()[0] if found.stdout.strip() else ""

        def drop_derived() -> None:
            sources = {self.destination.norm(t) for t in ctx.task.source_tables}
            for relation in self.destination.list_tables(ctx.namespace, include_views=True):
                if self.destination.norm(relation) not in sources:
                    self.destination.drop_table(ctx.namespace, relation)

        try:
            await asyncio.to_thread(drop_derived)
        except Exception as e:
            logger.warning("Replay check could not reset the namespace: %s", e)
            return False
        if not project_file:
            return False
        project_dir = project_file.rsplit("/", 1)[0]
        result = await ctx.sandbox.run_command(
            "dbt run", workdir=project_dir, timeout=self.replay_timeout
        )
        if result.exit_code != 0:
            logger.info("Replay `dbt run` failed for %s: %s", ctx.rollout_id, result.stdout[-500:])
        return result.exit_code == 0
