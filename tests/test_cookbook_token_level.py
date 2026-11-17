"""Token-level integration with tinker-cookbook's RL machinery, offline.

Everything the training loop does with this environment short of calling the
Tinker API: ``do_group_rollout`` builds the envs through ``make_envs``, drives
them with a token policy through the real renderer (tool calls rendered to
tokens and parsed back), collects execution-derived rewards from
``compute_group_rewards``, and the result goes through the cookbook's
advantage computation and datum assembly. The only stand-in is the sampler.
"""

from __future__ import annotations

import pytest
import torch
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout

from elt_rl.env import EnvConfig
from elt_rl.grading.grader import RewardConfig
from elt_rl.scripted import SCRIPTS

from .conftest import make_builder
from .fake_model import ScriptedTokenCompleter, byte_tokenizer

# One token per byte makes prompts ~4x longer than with a real tokenizer.
ENV = EnvConfig(max_trajectory_tokens=400_000, max_generation_tokens=4096)


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = byte_tokenizer()
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda model_name: tokenizer)


@pytest.mark.parametrize("start_stage", ["full", "transform"])
async def test_group_rollout_to_training_data(tasks, runtime_config, start_stage) -> None:
    # Three policies of different quality in one group, so advantages are not all zero.
    if start_stage == "full":
        policies = ["oracle", "extract_load_only", "noop"]
        expected = [1.0, RewardConfig().w_el, 0.0]
    else:
        policies = ["oracle", "cheat_tamper_sources", "noop"]
        expected = [1.0, 0.0, 0.0]
    builder = make_builder(
        tasks["bookshop"], runtime_config, start_stage=start_stage, group_size=3, env_config=ENV
    )
    policy = ScriptedTokenCompleter(builder, lambda i, ctx: SCRIPTS[policies[i]](ctx))

    group = await do_group_rollout(builder, policy)

    # Rewards are the execution-verified ones; steps themselves carry none.
    assert group.get_total_rewards() == pytest.approx(expected)
    for trajectory in group.trajectories_G:
        assert all(t.reward == 0.0 for t in trajectory.transitions)
    assert [g.reward for g in builder.last_grades] == pytest.approx(expected)
    metrics = group.metrics_G
    assert [m["elt/task_success"] for m in metrics] == [1.0, 0.0, 0.0]
    assert metrics[0]["elt/submitted"] == 1.0
    # Tool calls survived the round trip through tokens: the oracle really ran dbt.
    assert metrics[0]["elt/commands_run"] >= 1
    assert builder.last_histories[0][-1]["content"] == "Submission received."
    # The group was cleaned up by do_group_rollout.
    assert builder._rollouts == []
    assert policy.calls >= len(group.trajectories_G[0].transitions)

    advantages = compute_advantages([group])
    assert advantages[0].tolist() == pytest.approx(
        (torch.tensor(expected) - torch.tensor(expected).mean()).tolist()
    )
    data, metadata = assemble_training_data([group], advantages)
    assert len(metadata) == len(data) >= 3
    best = [d for d, m in zip(data, metadata, strict=True) if m["traj_idx"] == 0]
    assert best, "no training datum for the successful trajectory"
    for datum in best:
        weights = datum.loss_fn_inputs["advantages"].to_torch()
        assert weights.max().item() == pytest.approx(advantages[0][0].item())
        assert (weights != 0).sum().item() > 0
        assert datum.loss_fn_inputs["logprobs"].to_torch().min().item() == pytest.approx(-0.25)


async def test_context_budget_stops_a_rollout_and_it_is_still_graded(
    tasks, runtime_config
) -> None:
    """A rollout that runs out of context ends cleanly and keeps its verified credit."""
    # Enough for the extract/load turns of the oracle, not for the dbt project.
    tight = EnvConfig(max_trajectory_tokens=12_000, max_generation_tokens=1024)
    builder = make_builder(tasks["clinic"], runtime_config, env_config=tight)
    policy = ScriptedTokenCompleter(builder, lambda i, ctx: SCRIPTS["oracle"](ctx))
    group = await do_group_rollout(builder, policy)
    (grade,) = builder.last_grades
    assert policy.max_prompt_tokens <= tight.max_trajectory_tokens
    assert grade.metrics["elt/submitted"] == 0.0
    assert grade.metrics["elt/task_success"] == 0.0
    # The data that was really loaded before the budget ran out is credited...
    assert grade.metrics["elt/el_success"] == 1.0
    assert grade.reward == pytest.approx(RewardConfig().w_el)
    # ...and the overflow itself costs the configured step penalty.
    assert group.get_total_rewards()[0] == pytest.approx(
        grade.reward + tight.context_overflow_reward
    )
