"""Official runtime: ELT-Bench tasks on the real stack.

Requires the ELT-Bench setup to be in place on the host (see README):
Airbyte (abctl), the source containers from ``elt-docker``, the generated
input bundles (``setup/write_config.py``), the agent image (``elt-swe``) and
warehouse credentials.

Per rollout it follows the official runners (``agents/codex_agent/run.py``):
copy the bundle without credential files, point ``config.yaml`` at a fresh
namespace, reset that namespace, and start a container on the benchmark's
Docker network with only ``/workspace/elt`` writable. Unlike the official
runners it gives every rollout its own namespace, so the rollouts of one
training group can run concurrently on the same task.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook import cli_utils

from elt_rl import prompts
from elt_rl.destinations import Destination, Namespace, get_destination
from elt_rl.grading.grader import StartStage
from elt_rl.runtime.base import ELTRuntime, RolloutContext
from elt_rl.sandbox.docker import CONTAINER_WORKDIR, DockerSandbox
from elt_rl.tasks import ELTTask

logger = logging.getLogger(__name__)


@chz.chz
class OfficialRuntimeConfig:
    """Configuration of the official runtime (picklable; build with ``build()``)."""

    destination: str = "snowflake"
    credential_path: str = ""
    """Host-side admin credential JSON (never copied into a workspace), e.g.
    ``<ELT-Bench>/setup/destination/snowflake_credential.json``."""
    database: str = "ELT_RL"
    """Shared database/catalog that holds one schema per rollout."""
    agent_role: str | None = "AIRBYTE_ROLE"
    """Role used by the agent's credential; granted ownership of its schema."""
    reference_database: str | None = None
    """Database holding a known-good load of every task's source tables, one
    schema per task. Enables transformation-only rollouts (zero-copy clone)
    and the stage-1 content check."""
    image: str = "elt-swe"
    network: str = "elt-docker_elt_network"
    """Docker network of the rollout containers. It must reach Airbyte and the
    source systems. The benchmark's network also reaches the internet; see
    ``egress_proxy`` to close that."""
    egress_proxy: str | None = None
    """URL of an allow-listing forward proxy, e.g. ``http://elt-egress:3128``.
    Rollout containers get ``HTTP(S)_PROXY`` pointing at it. Use it with an
    internal Docker network (no route out) as ``network``, so the proxy is
    the only way to the internet and only the warehouse is reachable: the
    ground truth of the official tasks is public, and nothing else stops a
    policy from downloading it."""
    no_proxy: str = (
        "localhost,127.0.0.1,airbyte-abctl-control-plane,elt-postgres,elt-mongodb,"
        "elt-localstack,elt-api"
    )
    """Hosts reached directly (Airbyte and the source systems)."""
    runs_dir: str | None = None
    command_timeout: int = 600
    verify_replay: bool = False
    terraform_destroy_on_stop: bool = True
    """Delete the Airbyte sources/destinations/connections a rollout created."""
    drop_namespace_on_stop: bool = True
    keep_workspaces: bool = False

    def build(self) -> OfficialRuntime:
        return OfficialRuntime(self)


class OfficialRuntime(ELTRuntime):
    def __init__(
        self, config: OfficialRuntimeConfig, destination: Destination | None = None
    ) -> None:
        self.config = config
        if destination is None:
            kwargs: dict[str, Any] = {}
            if config.destination == "snowflake":
                kwargs = {
                    "credential": config.credential_path,
                    "database": config.database,
                    "agent_role": config.agent_role,
                }
            destination = get_destination(config.destination, **kwargs)
        self.destination = destination
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
            source_kinds="custom APIs, PostgreSQL, MongoDB, flat files, and the cloud service S3",
            el_tool="Airbyte's Terraform provider",
            tooling="terraform, dbt, python, psql",
        )

    def instructions(self, ctx: RolloutContext) -> str:
        template = (
            prompts.OFFICIAL_TRANSFORM_INSTRUCTION
            if ctx.start_stage == "transform"
            else prompts.OFFICIAL_FULL_INSTRUCTION
        )
        text = template.format(workspace=ctx.workspace, destination=self.destination.display_name)
        return text + (prompts.REPLAY_NOTE if self.verify_replay else "")

    # -- lifecycle -------------------------------------------------------------------
    def reference_namespace(self, task: ELTTask) -> Namespace | None:
        if not self.config.reference_database:
            return None
        return Namespace(database=self.config.reference_database, schema=task.name)

    def _prepare(
        self, task: ELTTask, rollout_id: str, start_stage: StartStage
    ) -> tuple[Path, Namespace, Namespace | None]:
        host_workspace = self.runs_dir / f"{task.name}-{rollout_id}"
        if host_workspace.exists():
            shutil.rmtree(host_workspace)
        ns = self.destination.new_namespace(
            task.name, rollout_id, str(host_workspace), CONTAINER_WORKDIR
        )
        reference = self.reference_namespace(task)
        self.prepare_workspace(task, host_workspace, ns, start_stage)
        if start_stage == "transform":
            if reference is None:
                raise ValueError(
                    "Transformation-only rollouts need reference_database (a loaded copy "
                    "of the source tables to clone)"
                )
            self.destination.create_namespace(ns, clone_from=reference)
        else:
            # A failed reset means stale warehouse state: let it raise so the
            # rollout is aborted rather than graded on someone else's tables.
            self.destination.create_namespace(ns)
        return host_workspace, ns, reference

    async def start(
        self, task: ELTTask, rollout_id: str, start_stage: StartStage
    ) -> RolloutContext:
        host_workspace, ns, reference = await asyncio.to_thread(
            self._prepare, task, rollout_id, start_stage
        )
        env: dict[str, str] = {}
        if self.config.egress_proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env[key] = self.config.egress_proxy
            env["NO_PROXY"] = env["no_proxy"] = self.config.no_proxy
        sandbox = await DockerSandbox.create(
            image=self.config.image,
            host_workspace=host_workspace,
            network=self.config.network,
            writable_subdirs=("elt",),
            env=env,
            name=f"elt-rl-{task.name.replace('_', '-')}-{rollout_id}",
        )
        return RolloutContext(
            task=task,
            rollout_id=rollout_id,
            start_stage=start_stage,
            sandbox=sandbox,
            namespace=ns,
            workspace=CONTAINER_WORKDIR,
            host_workspace=host_workspace,
            reference=reference,
        )

    async def stop(self, ctx: RolloutContext) -> None:
        # Order matters: Airbyte resources are destroyed from inside the
        # sandbox (it holds the Terraform state), then the container goes.
        try:
            if self.config.terraform_destroy_on_stop and ctx.start_stage == "full":
                await ctx.sandbox.run_command(
                    "test -f terraform.tfstate && terraform destroy -auto-approve -input=false "
                    "|| true",
                    workdir=f"{ctx.workspace}/elt",
                    timeout=300,
                )
        except Exception as e:
            logger.warning("terraform destroy failed for %s: %s", ctx.rollout_id, e)
        try:
            await ctx.sandbox.cleanup()
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", ctx.rollout_id, e)
        if self.config.drop_namespace_on_stop:
            try:
                await asyncio.to_thread(self.destination.drop_namespace, ctx.namespace)
            except Exception as e:
                logger.warning("Dropping namespace failed for %s: %s", ctx.rollout_id, e)
        if not self.config.keep_workspaces:
            await asyncio.to_thread(shutil.rmtree, ctx.host_workspace, True)
