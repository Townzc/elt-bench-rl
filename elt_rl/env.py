"""ELT-Bench RL environment for tinker-cookbook.

One :class:`ELTEnvGroupBuilder` is one task; its group is ``group_size``
independent rollouts of that task, each with its own sandbox and warehouse
namespace. The episode is a tool-use conversation (``bash`` / ``write_file`` /
``submit``) driven by tinker-cookbook's ``AgentToolMessageEnv``.

Reward comes from ``compute_group_rewards``, not from the conversation: after
every rollout of the group has ended, for whatever reason, the warehouse
state each one left behind is verified by execution and scored. A rollout
that ran out of turns after loading all the data is therefore credited for
it, and nothing the model says about its own success is used.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tool_use import AgentToolMessageEnv, build_agent_tool_env
from tinker_cookbook.utils import logtree

from elt_rl.grading.grader import Grader, GradeResult, RewardConfig, StartStage
from elt_rl.runtime import RolloutContext, RuntimeConfig, get_runtime, new_rollout_id
from elt_rl.tasks import ELTTask
from elt_rl.tools import ELTTools

logger = logging.getLogger(__name__)

StageSchedule = Literal["full", "transform", "mixed"]


@chz.chz
class EnvConfig:
    """Per-rollout budgets and conversation handling."""

    max_turns: int = 40
    """Assistant turns per rollout. A full ELT task needs a few dozen."""
    max_trajectory_tokens: int = 64 * 1024
    """Context budget; the episode ends before a prompt would overflow it."""
    max_generation_tokens: int | None = None
    """Reserve for the next generation when checking the context budget."""
    context_overflow_reward: float = -0.1
    failed_parse_reward: float = -0.1
    parse_error_retries: int = 2
    """Consecutive malformed tool calls answered with an error before stopping."""
    rollout_timeout_seconds: float | None = 3600.0
    """Wall-clock budget of one rollout."""
    max_output_chars: int = 8000
    """Tool output kept per stream (head and tail); the rest is elided."""
    require_submit: bool = False
    """Give no reward unless the agent called ``submit``. Off by default: the
    warehouse is graded however the episode ended."""


@dataclass
class Rollout:
    """One rollout of a group: its runtime state and its tools."""

    ctx: RolloutContext
    tools: ELTTools
    grade: GradeResult | None = None


async def _no_step_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    """Episode-end hook of ``AgentToolMessageEnv``; grading happens at group level."""
    del history
    return 0.0, {}


_GRADERS: dict[tuple[str, str], Grader] = {}


def _get_grader(runtime_config: RuntimeConfig, reward_config: RewardConfig) -> Grader:
    key = (repr(runtime_config), repr(reward_config))
    if key not in _GRADERS:
        destination = get_runtime(runtime_config).destination
        _GRADERS[key] = Grader(
            destination, reward_config, max_workers=destination.grading_concurrency
        )
    return _GRADERS[key]


class ELTEnvGroupBuilder(EnvGroupBuilder):
    """Builds ``group_size`` rollouts of one ELT task and grades them."""

    def __init__(
        self,
        task: ELTTask,
        runtime_config: RuntimeConfig,
        model_name: str,
        renderer_name: str | None = None,
        group_size: int = 4,
        start_stage: StartStage = "full",
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.task = task
        self.runtime_config = runtime_config
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.start_stage: StartStage = start_stage
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        # Live state, created in make_envs(); never pickled with content.
        self._rollouts: list[Rollout] = []
        self._by_env: dict[int, Rollout] = {}
        # Kept after cleanup, for reporting (evaluation scripts, tests).
        self.last_grades: list[GradeResult] = []
        self.last_histories: list[list[Message]] = []

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.update(_rollouts=[], _by_env={}, last_grades=[], last_histories=[])
        return state

    def fresh(self, start_stage: StartStage | None = None) -> ELTEnvGroupBuilder:
        """A new builder for the same task with no live state.

        A builder owns the sandboxes of one group, so every group that runs
        (including the same task appearing twice in a batch) needs its own.
        """
        return ELTEnvGroupBuilder(
            task=self.task,
            runtime_config=self.runtime_config,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            group_size=self.group_size,
            start_stage=start_stage or self.start_stage,
            env_config=self.env_config,
            reward_config=self.reward_config,
        )

    # -- rollout construction -----------------------------------------------------
    async def start_rollouts(self) -> list[Rollout]:
        """Create the sandbox, workspace and namespace of every group member."""
        runtime = get_runtime(self.runtime_config)
        started = await asyncio.gather(
            *[
                runtime.start(self.task, new_rollout_id(), self.start_stage)
                for _ in range(self.group_size)
            ],
            return_exceptions=True,
        )
        contexts = [c for c in started if isinstance(c, RolloutContext)]
        errors = [c for c in started if isinstance(c, BaseException)]
        if errors:
            await asyncio.gather(*[runtime.stop(c) for c in contexts])
            raise errors[0]
        self._rollouts = [
            Rollout(
                ctx=ctx,
                tools=ELTTools(
                    ctx.sandbox,
                    ctx.workspace,
                    command_timeout=runtime.command_timeout,
                    max_output_chars=self.env_config.max_output_chars,
                ),
            )
            for ctx in contexts
        ]
        return self._rollouts

    def initial_messages(self, rollout: Rollout, renderer: Renderer | None = None) -> list[Message]:
        """System prompt with tool declarations, then the task instruction."""
        runtime = get_runtime(self.runtime_config)
        system_prompt = runtime.system_prompt(rollout.ctx)
        user: Message = {"role": "user", "content": runtime.instructions(rollout.ctx)}
        if renderer is None:
            return [{"role": "system", "content": system_prompt}, user]
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in rollout.tools.tools], system_prompt=system_prompt
        )
        return prefix + [user]

    def _rollout_config(self) -> RolloutConfig:
        cfg = self.env_config
        return RolloutConfig(
            limits=RolloutLimits(rollout_timeout_seconds=cfg.rollout_timeout_seconds),
            parse_errors=(
                ParseErrorPolicy(max_consecutive=cfg.parse_error_retries)
                if cfg.parse_error_retries > 0
                else None
            ),
            # The tools share a filesystem and a warehouse: the model's call
            # order is the order of side effects.
            tool_execution="sequential",
        )

    def build_message_env(self, rollout: Rollout) -> AgentToolMessageEnv:
        """Message-level environment, for scripted policies and tests (no tokenizer)."""
        rollout_config = self._rollout_config()
        return AgentToolMessageEnv(
            tools=rollout.tools.tools,
            initial_messages=self.initial_messages(rollout),
            max_turns=self.env_config.max_turns,
            reward_fn=_no_step_reward,
            failed_parse_reward=self.env_config.failed_parse_reward,
            parse_error_policy=rollout_config.parse_errors,
            tool_execution=rollout_config.tool_execution,
        )

    async def make_envs(self) -> Sequence[Env]:
        rollouts = await self.start_rollouts()
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)
        cfg = self.env_config
        envs: list[Env] = []
        self._by_env = {}
        for rollout in rollouts:
            env = build_agent_tool_env(
                renderer=renderer,
                tools=rollout.tools.tools,
                initial_messages=self.initial_messages(rollout, renderer),
                reward_fn=_no_step_reward,
                rollout_config=self._rollout_config(),
                max_turns=cfg.max_turns,
                failed_parse_reward=cfg.failed_parse_reward,
                max_trajectory_tokens=cfg.max_trajectory_tokens,
                max_generation_tokens=cfg.max_generation_tokens,
                context_overflow_reward=cfg.context_overflow_reward,
            )
            self._by_env[id(env)] = rollout
            envs.append(env)
        return envs

    # -- grading -------------------------------------------------------------------
    async def grade_rollout(self, rollout: Rollout) -> GradeResult:
        """Verify one rollout's warehouse state and score it."""
        runtime = get_runtime(self.runtime_config)
        grader = _get_grader(self.runtime_config, self.reward_config)
        ctx = rollout.ctx
        await runtime.finalize(ctx)
        result = await asyncio.to_thread(
            grader.grade, ctx.task, ctx.namespace, ctx.start_stage, ctx.reference
        )
        checks = len(result.sources) + len(result.models)
        if checks and result.metrics["elt/check_errors"] >= checks:
            # Every check errored: the warehouse was unreachable, not wrong. Fail
            # the group instead of training on a fabricated zero.
            first = next(
                (c.error for c in [*result.sources, *result.models] if c.error), "unknown"
            )
            raise RuntimeError(f"Grading infrastructure failure for {ctx.task.name}: {first}")

        result.metrics["elt/submitted"] = float(rollout.tools.submitted)
        result.metrics["elt/commands_run"] = float(rollout.tools.commands_run)
        if ctx.replay_ok is not None:
            result.metrics["elt/replay_ok"] = float(ctx.replay_ok)
        if self.env_config.require_submit and not rollout.tools.submitted:
            result.reward = 0.0
        rollout.grade = result
        return result

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        rollouts = [self._by_env[id(env)] for env in env_group]
        results = await asyncio.gather(*[self.grade_rollout(r) for r in rollouts])
        self.last_grades = list(results)
        self.last_histories = [
            list(getattr(getattr(env, "message_env", None), "history", [])) for env in env_group
        ]
        with logtree.scope_header(f"ELT verification: {self.task.name}"):
            for result in results:
                logtree.details(result.summary(), summary=f"reward={result.reward:.3f}")
        return [(r.reward, dict(r.metrics)) for r in results]

    async def cleanup(self) -> None:
        runtime = get_runtime(self.runtime_config)
        rollouts, self._rollouts, self._by_env = self._rollouts, [], {}
        for rollout in rollouts:
            try:
                await runtime.stop(rollout.ctx)
            except Exception as e:
                logger.warning("Cleanup failed for rollout %s: %s", rollout.ctx.rollout_id, e)

    def logging_tags(self) -> list[str]:
        return ["elt", f"elt_{self.task.destination}", f"elt_{self.start_stage}"]


class ELTDataset(RLDataset):
    """Batches of task groups, cycling through the task list.

    With the ``mixed`` schedule a task alternates between the two start
    stages on successive passes. A group is always one stage, because
    advantages are centered within the group.
    """

    def __init__(
        self,
        builders: list[ELTEnvGroupBuilder],
        batch_size: int,
        n_batches: int | None = None,
        schedule: StageSchedule = "full",
    ) -> None:
        if not builders:
            raise ValueError("ELTDataset needs at least one task")
        self.builders = builders
        self.batch_size = batch_size
        self.schedule: StageSchedule = schedule
        self.n_batches = n_batches or math.ceil(len(builders) / batch_size)

    def _stage(self, position: int) -> StartStage:
        if self.schedule != "mixed":
            return self.schedule
        task_index, pass_index = position % len(self.builders), position // len(self.builders)
        return "transform" if (task_index + pass_index) % 2 == 0 else "full"

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return [
            self.builders[position % len(self.builders)].fresh(self._stage(position))
            for position in range(start, start + self.batch_size)
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    """Builds the RL dataset over ELT tasks."""

    tasks: list[ELTTask]
    runtime_config: RuntimeConfig
    model_name: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    start_stage: StageSchedule = "full"
    """``full`` (extract/load + transform), ``transform`` (sources pre-loaded
    from a reference snapshot), or ``mixed`` (alternate per task)."""
    eval_tasks: list[ELTTask] | None = None
    """Held-out tasks for periodic evaluation (group size 1)."""
    n_batches: int | None = None
    """Training batches; defaults to one pass over the tasks."""
    env_config: EnvConfig = chz.field(default_factory=EnvConfig)
    reward_config: RewardConfig = chz.field(default_factory=RewardConfig)

    def _builders(self, tasks: list[ELTTask], group_size: int) -> list[ELTEnvGroupBuilder]:
        return [
            ELTEnvGroupBuilder(
                task=task,
                runtime_config=self.runtime_config,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                group_size=group_size,
                env_config=self.env_config,
                reward_config=self.reward_config,
            )
            for task in tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train = ELTDataset(
            self._builders(self.tasks, self.group_size),
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            schedule=self.start_stage,
        )
        if not self.eval_tasks:
            return train, None
        # Evaluation always uses the full task: that is the benchmark.
        eval_stage: StageSchedule = "full" if self.start_stage == "mixed" else self.start_stage
        evaluation = ELTDataset(
            self._builders(self.eval_tasks, 1),
            batch_size=min(self.batch_size, len(self.eval_tasks)),
            schedule=eval_stage,
        )
        return train, evaluation
