"""RL training on ELT tasks with Tinker.

Thin wrapper around ``tinker_cookbook.rl.train``: it builds the ELT dataset
and hands it to the cookbook's training loop, so sampling, advantage
computation, the optimizer step, checkpointing and logging are all the
cookbook's.

Credential-free tasks (needs only TINKER_API_KEY)::

    python -m elt_rl.scripts.train backend=local start_stage=mixed \\
        group_size=4 groups_per_batch=3 max_steps=1

Longer runs should overlap sampling with training (async RL), e.g.
``num_epochs=20 sampling_steps_ahead=2``.

Official ELT-Bench tasks::

    python -m elt_rl.scripts.train backend=official elt_bench_root=/path/to/ELT-Bench \\
        tasks=books,airline group_size=4 groups_per_batch=2
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from elt_rl.env import ELTDatasetBuilder, EnvConfig, StageSchedule
from elt_rl.grading.grader import RewardConfig
from elt_rl.scripts.common import BackendConfig, resolve_backend

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig(BackendConfig):
    """Command-line configuration for ELT RL training.

    Inherits the task/infrastructure options of :class:`BackendConfig`
    (``backend=``, ``tasks=``, ``elt_bench_root=``, ...).
    """

    start_stage: StageSchedule = "full"
    eval_tasks: str | None = None
    """Comma-separated held-out task names, removed from training."""

    # Model
    model_name: str = "moonshotai/Kimi-K2.6"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 4096
    temperature: float = 1.0

    # Environment and reward
    env: EnvConfig = chz.field(default_factory=EnvConfig)
    reward: RewardConfig = chz.field(default_factory=RewardConfig)
    zero_reward_on_limit: bool = False
    """Clamp the reward of rollouts stopped by a budget to <= 0 (the cookbook's
    agentic default). Off here: partial pipelines are still informative."""

    # Training
    group_size: int = 4
    groups_per_batch: int = 4
    n_batches: int | None = None
    """Batches per pass over the tasks; default is one group per task."""
    num_epochs: int = 1
    """Passes over the task list. Steps = min(num_epochs * n_batches, max_steps)."""
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    max_steps: int | None = None
    sampling_steps_ahead: int | None = None
    """Enable async RL: sampling may run this many steps ahead of training.
    Rollouts are slow (sandbox + warehouse), so overlapping them with
    training is the main wall-clock lever."""
    rollout_error_tolerance: bool = True
    """Retry rollouts that fail for infrastructure reasons instead of failing the step."""
    remove_constant_reward_groups: bool = False
    """Skip groups whose rollouts all got the same reward (no advantage signal)."""

    # Logging / eval / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 5
    save_every: int = 5
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> Config:
    tasks, runtime_config = resolve_backend(cli_config)
    eval_names = (
        {t.strip() for t in cli_config.eval_tasks.split(",") if t.strip()}
        if cli_config.eval_tasks
        else set()
    )
    train_tasks = [t for t in tasks if t.name not in eval_names]
    eval_tasks = [t for t in tasks if t.name in eval_names]
    missing = eval_names - {t.name for t in eval_tasks}
    if missing:
        raise ValueError(f"eval_tasks not found: {sorted(missing)}")
    if not train_tasks:
        raise ValueError("No training tasks left after removing eval_tasks")

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"elt-{cli_config.backend}-{cli_config.start_stage}-{model_tag}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.group_size}group-{cli_config.groups_per_batch}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or cli_utils.runs_path("elt_rl", run_name)
    env_config = cli_config.env
    if env_config.max_generation_tokens is None:
        env_config = chz.replace(env_config, max_generation_tokens=cli_config.max_tokens)

    logger.info(
        "ELT RL: %d training task(s) %s, %d eval task(s), stage=%s",
        len(train_tasks),
        [t.name for t in train_tasks],
        len(eval_tasks),
        cli_config.start_stage,
    )
    dataset_builder = ELTDatasetBuilder(
        tasks=train_tasks,
        eval_tasks=eval_tasks or None,
        runtime_config=runtime_config,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        start_stage=cli_config.start_stage,
        n_batches=cli_config.n_batches,
        env_config=env_config,
        reward_config=cli_config.reward,
    )
    return Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_elt_rl",
        lora_rank=cli_config.lora_rank,
        renderer_name=renderer_name,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        session_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        num_epochs=cli_config.num_epochs,
        rollout_error_tolerance=cli_config.rollout_error_tolerance,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        termination=(
            TerminationRewardPolicy(zero_reward_on_limit=True)
            if cli_config.zero_reward_on_limit
            else None
        ),
        async_config=(
            AsyncConfig(
                sampling_steps_ahead=cli_config.sampling_steps_ahead,
                groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.sampling_steps_ahead is not None
            else None
        ),
    )


async def cli_main(cli_config: CLIConfig) -> None:
    config = build_config(cli_config)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
