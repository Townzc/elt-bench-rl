"""Run rollouts on ELT tasks and print the execution-verified result.

No training: this is the tool for checking that the environment works end to
end, and for evaluating a model or checkpoint.

Credential-free, with a scripted policy (no API key needed)::

    python -m elt_rl.scripts.rollout backend=local policy=oracle

A Tinker model on the local tasks::

    python -m elt_rl.scripts.rollout backend=local policy=tinker model_name=moonshotai/Kimi-K2.6

A Tinker model on an official ELT-Bench task (needs the ELT-Bench stack)::

    python -m elt_rl.scripts.rollout backend=official elt_bench_root=/path/to/ELT-Bench \\
        tasks=books policy=tinker model_name=moonshotai/Kimi-K2.6
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers.base import Message

from elt_rl.env import ELTEnvGroupBuilder, EnvConfig
from elt_rl.grading.grader import GradeResult, RewardConfig, StartStage
from elt_rl.scripts.common import BackendConfig, resolve_backend

logger = logging.getLogger(__name__)


@chz.chz
class RolloutCLIConfig(BackendConfig):
    """Options of the rollout script (plus those of :class:`BackendConfig`)."""

    policy: str = "oracle"
    """``tinker`` samples from a Tinker model; any other value names a scripted
    policy from ``elt_rl.scripted`` (local backend only), e.g. ``oracle``."""
    start_stage: StartStage = "full"
    group_size: int = 1
    """Rollouts per task."""

    # Tinker policy
    model_name: str = "moonshotai/Kimi-K2.6"
    renderer_name: str | None = None
    checkpoint_path: str | None = None
    """``tinker://...`` sampler weights to evaluate instead of the base model."""
    max_tokens: int = 4096
    temperature: float = 1.0
    base_url: str | None = None

    env: EnvConfig = chz.field(default_factory=EnvConfig)
    reward: RewardConfig = chz.field(default_factory=RewardConfig)
    output_dir: str | None = None


def _serializable(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message["role"], "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in message["tool_calls"]
        ]
    return json.loads(json.dumps(out, default=str))


def _grade_record(grade: GradeResult) -> dict[str, Any]:
    return {
        "task": grade.task,
        "start_stage": grade.start_stage,
        "reward": grade.reward,
        "metrics": grade.metrics,
        "sources": [asdict(s) | {"ok": s.ok} for s in grade.sources],
        "models": [
            {
                "name": m.name,
                "exists": m.exists,
                "match": m.match,
                "column_frac": m.column_frac,
                "matched": m.comparison.matched,
                "unmatched": m.comparison.unmatched,
                "missed": m.comparison.missed,
                "error": m.error,
            }
            for m in grade.models
        ],
    }


async def _tinker_policy(config: RolloutCLIConfig) -> Any:
    import tinker
    from tinker_cookbook.completers import TinkerTokenCompleter

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if config.checkpoint_path:
        sampling_client = await service_client.create_sampling_client(
            model_path=config.checkpoint_path, base_model=config.model_name
        )
    else:
        sampling_client = await service_client.create_sampling_client(base_model=config.model_name)
    return TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


async def run(config: RolloutCLIConfig) -> list[dict[str, Any]]:
    tasks, runtime_config = resolve_backend(config)
    if config.policy != "tinker" and config.backend != "local":
        raise ValueError("Scripted policies only exist for the local backend")
    output_dir = Path(
        config.output_dir
        or cli_utils.runs_path("elt_rl", "rollouts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config = config.env
    if env_config.max_generation_tokens is None:
        env_config = chz.replace(env_config, max_generation_tokens=config.max_tokens)

    policy = await _tinker_policy(config) if config.policy == "tinker" else None
    records: list[dict[str, Any]] = []

    async def run_task(task: Any) -> None:
        builder = ELTEnvGroupBuilder(
            task=task,
            runtime_config=runtime_config,
            model_name=config.model_name,
            renderer_name=config.renderer_name,
            group_size=config.group_size,
            start_stage=config.start_stage,
            env_config=env_config,
            reward_config=config.reward,
        )
        started = time.monotonic()
        total_rewards: list[float] | None = None
        if policy is None:
            from elt_rl.scripted import run_scripted_group

            await run_scripted_group(builder, config.policy)
        else:
            from tinker_cookbook.rl.rollouts import do_group_rollout

            group = await do_group_rollout(builder, policy)
            total_rewards = group.get_total_rewards()
        elapsed = time.monotonic() - started
        for i, grade in enumerate(builder.last_grades):
            record = _grade_record(grade)
            record["seconds"] = round(elapsed, 1)
            # Verified reward plus any per-step penalties (parse errors, overflow).
            record["total_reward"] = total_rewards[i] if total_rewards else grade.reward
            records.append(record)
            print(grade.summary())
            history = builder.last_histories[i] if i < len(builder.last_histories) else []
            (output_dir / f"{task.name}_{i}_transcript.json").write_text(
                json.dumps([_serializable(m) for m in history], indent=1), encoding="utf-8"
            )

    # Tasks run concurrently; each owns its sandboxes and namespaces.
    await asyncio.gather(*[run_task(task) for task in tasks])

    (output_dir / "results.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    if records:
        mean = sum(r["reward"] for r in records) / len(records)
        success = sum(r["metrics"]["elt/task_success"] for r in records) / len(records)
        print(
            f"\n{len(records)} rollout(s): mean verified reward {mean:.3f}, "
            f"task success rate {success:.3f}"
        )
    print(f"Results and transcripts: {output_dir}")
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(chz.entrypoint(RolloutCLIConfig)))
