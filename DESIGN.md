# ELT-Bench RLVR environment: implementation design

This is the design of `elt-rl`, a Tinker RLVR recipe for ELT-Bench built on
tinker-cookbook. It follows the five steps of the task description. The last
section lists what has been run: the credential-free path is tested end to end,
and the credentialed paths (Tinker, Docker, Airbyte + Snowflake) are written but
have not been run.

## 1. The ELT task

**Components.** An ELT-Bench task is one source database spread over several
systems and a set of target data models.

| Component | Official benchmark | Local tasks in this repo |
|---|---|---|
| Sources | PostgreSQL, MongoDB, S3 (LocalStack), flat files, a REST API | SQLite, JSON-lines collections, CSV objects, flat files |
| Extract and load | Airbyte, configured with its Terraform provider | `minielt`: declare, apply, sync, check status |
| Destination | Snowflake (also Databricks, Redshift) | DuckDB |
| Transform | a dbt project | a dbt project |
| Task inputs | `config.yaml`, `data_model.yaml`, `schemas/`, `documentation/`, `elt/main.tf` | same layout |
| Grading assets | expected row counts, ground-truth CSVs, unique keys, evaluation SQL | same layout |

**What the agent sees.** A system prompt and the task instruction (the official
runner's text, with paths templated), then tool results. Three tools: `bash`,
`write_file` (confined to `<workspace>/elt`) and `submit`. The official runner ends
an episode when the model prints a `RESULT:` line; here the model calls `submit`,
which returns a fixed string and no information about correctness.

**Terminal conditions.** `submit`; a turn with no tool call; the turn budget; the
context budget; repeated malformed tool calls; the wall-clock budget. The episode
is graded in every case.

**Is the environment stateless? No.** A step changes state in three places:

1. the workspace filesystem (Terraform state, the dbt project, logs);
2. the extract/load control plane (Airbyte sources, destinations, connections, jobs);
3. the warehouse (raw tables, source tables, data models).

Consequences for the design:

- Each rollout gets a private copy of all three. `ELTRuntime.start()` creates a
  workspace, a sandbox and a warehouse namespace; `stop()` destroys them, including
  the Airbyte resources the rollout created (`terraform destroy`) and Airbyte's raw
  tables for that namespace.
- A group is `group_size` rollouts of the same task from the same initial state, so
  the group-centered advantage compares like with like. The official runner uses one
  schema per task; here it is one schema per rollout (`<TASK>__<ROLLOUT_ID>`), and the
  agent's `config.yaml` is rewritten to point at it.
- `Env` objects are single-use, as tinker-cookbook requires. The builder is picklable
  (it holds configuration only) and a fresh builder is made for every group in a batch.
- The initial state can be chosen. `full` starts empty. `transform` starts from a clone
  of a correct load of the source tables (ELT-Bench's transformation-only setting).

## 2. Reward

**What "correct" means.** A submitted pipeline is correct when, run against the
source systems, it leaves the warehouse with (a) every source table loaded completely
and exactly once, and (b) every data model of `data_model.yaml` holding the right
rows. Both are properties of warehouse state, so both can be checked by execution.

**How it is verified.**

- *Stage 1, extract and load.* Each expected source table must exist as a base table
  with the expected row count. That is the whole official check, and it can be passed
  with made-up rows, so when a reference load exists the content must match too: an
  order-independent hash computed inside the warehouse (`HASH_AGG` on Snowflake) is
  compared with the reference's, ignoring Airbyte's metadata columns. If the hashes
  differ the table is re-compared value by value with the tolerant comparator, so
  `1` loaded as `1.0` is not a failure.
- *Stage 2, transform.* Each data model is compared with its ground-truth CSV using
  a port of the official comparator: sort both sides by the model's unique key, then
  compare column by column; integers must be equal, decimals within 1% and within one
  unit of the last decimal place both sides have.
- *Replay.* Before grading, every relation except the source tables is dropped and
  `dbt run` is executed in the submitted project. What is graded is what the pipeline
  produces from the loaded sources, not whatever tables happen to exist. On by default
  for the local tasks, optional (`verify_replay=True`) for the official ones.

**Reward.**

```
reward = 0.3 * el_frac + 0.5 * model_frac + 0.2 * column_frac
```

`el_frac` is the fraction of source tables that pass stage 1, `model_frac` the
fraction of data models that match completely, `column_frac` the mean fraction of
matching non-key columns per model. With the `transform` start the first term is
dropped, the rest renormalized, and the reward is zero if a source table was changed.

**Outcome or process reward?** Outcome only, with partial credit.

- Process signals ("`terraform apply` succeeded", "a sync job finished", "`dbt run`
  printed PASS") are things the policy can print, fake or repeat. Paying for them is
  the standard way to get a policy that optimizes the signal.
- The task is long and the official success criterion is sparse and gated: stage 2
  only counts when stage 1 passed completely. A base model rarely gets there, so
  almost every group would have zero advantage. The three verified fractions give a
  gradient at every level (one more table loaded, one more column right) and each can
  only be earned by data that is in the warehouse.
- The gating is therefore not applied to the reward by default
  (`reward.gate_models_on_el=True` turns it on). The official metrics are logged
  unshaped next to the reward (`official/stage1_success`, `official/stage2_frac`,
  `elt/task_success`), so shaping cannot hide a regression on the real metric.
- The only process terms are the cookbook's small penalties for a malformed tool call
  and for running out of context (-0.1 each). They are never positive.

**Infrastructure failures are not rewards.** If every check of a rollout errors, the
warehouse was unreachable, not wrong: grading raises, and the cookbook's rollout
strategy retries or drops the group instead of training on a made-up zero.

## 3. Implementation

| Piece | File | Built on (tinker-cookbook) |
|---|---|---|
| Group builder, dataset, dataset builder | `env.py` | `EnvGroupBuilder`, `RLDataset`, `RLDatasetBuilder` |
| Episode loop, tool dispatch, budgets | `env.py` | `build_agent_tool_env`, `AgentToolMessageEnv`, `RolloutConfig` |
| Tools | `tools.py` | `@tool`, `simple_tool_result`, `SandboxInterface` |
| Sandboxes | `sandbox/` | `SandboxInterface` (same as the `harbor_rl` recipe) |
| Per-rollout state | `runtime/` | none |
| Warehouse access | `destinations/` | none |
| Verification and reward | `grading/` | returned from `compute_group_rewards` |
| Training, evaluation | `scripts/train.py`, `scripts/rollout.py` | `rl.train.main`, `do_group_rollout`, `TinkerTokenCompleter` |

`scripts/train.py` builds a `Config` and calls the cookbook's `main`. No sampling,
advantage or optimizer code is written here.

**Destinations.** The environment never names a warehouse. `Destination` has two
sides: host side (create, clone and drop a namespace; list tables; run a query; build
the fingerprint SQL; drop a table) and agent side (the `config.yaml` block that points
the agent at its namespace). Adding a warehouse is one subclass and one
`register_destination` call. DuckDB and Snowflake are implemented.

**Runtimes.** `OfficialRuntime` follows the official runner: copy the task bundle
without credential files, reset the namespace, start a container on the benchmark's
Docker network with only `/workspace/elt` writable. `LocalRuntime` does the same with
files instead of services, in a subprocess sandbox (tests) or a container with
`--network none`. Both store tasks in the ELT-Bench directory layout, so one loader
and one grader serve both.

**Testing.** Scripted policies drive the real environment: an oracle that solves each
local task, and one policy per reward-hacking shortcut. A token-level test runs
`do_group_rollout` with the cookbook's real `qwen3_instruct` renderer on an in-memory
byte tokenizer and a scripted sampler, then `compute_advantages` and
`assemble_training_data`: everything the training loop does short of the Tinker API.
The official code path runs against a fake Snowflake account and a fake container.

## 4. Reward hacking

| Loophole | Mitigation | Status |
|---|---|---|
| Claim success. The official runner trusts a printed `RESULT:` line. | The conversation is never read by the grader. | Tested |
| Use the grader as an oracle: submit, read feedback, adjust. | `submit` ends the episode and returns a fixed string. | Tested |
| Fake stage 1: tables with the right row count and arbitrary rows. | Content check against a reference load. | Tested. Needs `reference_database` on the official backend. |
| A view or a placeholder instead of a loaded table. | Source tables must be base tables. | Tested |
| Duplicated or partial loads. | Exact row counts. | Tested |
| Read the answers from disk. | Ground truth, evaluation SQL, keys, expected counts and credential files are never copied into a workspace; the grader runs on the host with its own credential. | Tested |
| Download the answers. The official ground truth is a public dataset. | Local: containers run with `--network none`. Official: internal Docker network plus an allow-listing proxy (`docker/egress/`). | Written, not run |
| Write the final tables by hand, or make dbt models read hand-made copies. | Replay from the source tables only. | Tested |
| Collect column credit for nothing: right keys, NULLs elsewhere. | Key columns earn no column credit; a wrong row count fails every column. | Tested |
| Transformation-only start: drop or reload source tables. | Reward is zero unless all source tables are intact. | Tested |
| Leave a process running that holds a lock or keeps writing during grading. | All sandbox processes are killed before the replay and grading. | Tested (process sandbox) |
| Make an infrastructure failure look like a result. | All checks erroring fails the group. | Tested |
| Copy from another rollout of the same task, or from the reference database, inside the shared Snowflake account. | Random schema names, schemas dropped at the end. The reference database must be readable by the admin role only. | **Open.** The agent role can read other rollouts' schemas. Needs one role per rollout. |
| Change or delete shared things: source data, other rollouts' Airbyte connections. | Each rollout destroys only what it created. | **Open.** Needs read-only source credentials and Airbyte workspaces per rollout. |
| Load the data without the intended tool. | None. Stage 1 is defined by its result. | Accepted |
| Column credit for pass-through columns that are not keys. | Bounded by `w_column` (0.2 by default; set it to 0 to remove). | Accepted |
| A data model defined as a view that never finishes, stalling grading. | The replay rebuilds models from the project; the cookbook's `grader_timeout_seconds` can bound grading. | Not configured |

The process sandbox isolates nothing, so the two mitigations for reading or
downloading the answers do not hold there. It is for tests and trusted debugging.

## 5. Efficiency

**Where the time goes.** A rollout costs setup + sum over turns of (sampling + tool
execution) + verification. On the official stack tool execution dominates: Airbyte
sync jobs take minutes and dbt on Snowflake tens of seconds per run, against a few
seconds of sampling per turn. Verification is small by comparison if it avoids moving
tables.

Measured on the local tasks (4 cores, scripted oracle, so no model latency):

| Phase | Full | Transform-only |
|---|---|---|
| Create workspace, sandbox, namespace | 0.05 s | under 0.01 s |
| Tool execution over the episode | 5.3 s | 4.3 s |
| Replay check (one `dbt run`) | 3.5 s | 3.5 s |
| Grading | 0.1 to 0.7 s | 0.15 s |

There are no measurements on the official stack.

**Transitions.**

- *Transformation-only start.* A zero-copy clone on Snowflake (a file copy on DuckDB)
  puts a rollout directly after stage 1. It removes the slowest part of the episode and
  gives stage-2 signal to a policy that cannot pass stage 1 yet. The `mixed` schedule
  alternates stages per task, which also halves the load on Airbyte.
- *Async RL.* `sampling_steps_ahead` lets sampling run ahead of training, so a step
  does not wait for the slowest rollout of its batch.
- *Concurrency.* The rollouts of a group and the groups of a batch are set up and run
  concurrently. Blocking work (warehouse calls, file copies) runs in threads.
- *Tokens.* Tool output keeps its head and tail (8,000 characters per stream by
  default), where errors are. Shorter prompts are faster turns.
- *Budgets.* Per command, per rollout wall-clock, per rollout turns and tokens.

**Reward computation.**

- Stage 1 compares one hash row per table instead of fetching tables. Hashes of the
  reference load are computed once per process.
- A data model with the wrong row count fails without being fetched.
- Ground-truth CSVs are parsed once and cached.
- One table listing and one connection per grading pass; the checks of a rollout run
  in parallel on Snowflake (8 threads); the rollouts of a group are graded concurrently.
  On DuckDB this took grading from about 0.5 s to about 0.12 s per rollout.

**Not implemented.** Pre-warmed containers; a Terraform plugin cache shared by
rollouts; reusing Airbyte source and destination definitions across the rollouts of a
task; a cap on concurrent full-stage rollouts matched to Airbyte's worker count.

## Status

Run: the local tasks end to end; the reward-hacking tests; the token-level cookbook
integration; the official code path against fakes. 63 tests pass.

Not run: any call to Tinker (sampling or an optimizer step); the Docker sandbox; a
rollout against Airbyte and Snowflake; the egress proxy. No model has been trained, so
the reward weights and budgets are first guesses.
