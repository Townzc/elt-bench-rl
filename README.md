# elt-rl: a Tinker RLVR recipe for ELT-Bench

Reinforcement learning with verifiable rewards on [ELT-Bench](https://arxiv.org/abs/2504.04808),
built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

In each rollout a language model works in a sandbox: it reads the task
specification, configures and runs extraction and loading, writes the dbt
transformations, reacts to what the tools print, and submits. The environment
then **executes checks against the warehouse the rollout left behind** and
turns the result into a reward. Nothing the model says about its own success
is used.

```
ELTDatasetBuilder ──► ELTEnvGroupBuilder (one task, group_size rollouts)
                         │ make_envs()                compute_group_rewards()
                         ▼                                   ▲
                  ELTRuntime.start()                   Grader.grade()
        workspace + sandbox + warehouse namespace   stage 1: source tables
                         │                           stage 2: data models
                         ▼                                   ▲
        AgentToolMessageEnv (tinker-cookbook)  ──►  ELTRuntime.finalize()
        tools: bash · write_file · submit            stop processes, replay dbt
```

The design (reward definition, reward-hacking analysis, efficiency) is in
[DESIGN.md](DESIGN.md).

## What has been run, and what has not

| Path | Status |
|---|---|
| Local tasks end to end (process sandbox, `minielt`, dbt, DuckDB, grader) | Run. 63 tests pass (`pytest`). |
| Reward-hacking regression tests (the shortcuts in the table below) | Run. |
| tinker-cookbook integration at token level (`do_group_rollout` → advantages → training data) with the real `qwen3_instruct` renderer and a scripted sampler | Run, offline. |
| Official code path with Airbyte, the container and Snowflake replaced by fakes | Run. This checks orchestration and SQL, not the real services. |
| Sampling from a Tinker model, and an optimizer step | **Not run.** Needs `TINKER_API_KEY`. `pytest -m tinker` and the commands below do it. |
| Docker sandbox (`local_sandbox=docker`, and every official rollout) | **Not run.** No Docker daemon was available where this was written. |
| A rollout on an official ELT-Bench task (Airbyte + Snowflake) | **Not run.** Needs the ELT-Bench stack and Snowflake credentials. |
| The egress proxy in `docker/egress/` | **Not run.** |

Expect to fix small things on the first credentialed run of the rows marked "Not run".

## Install

```bash
git clone https://github.com/Townzc/elt-bench-rl.git && cd elt-bench-rl
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # add ",snowflake" for the official tasks
```

Python 3.11 or newer. `tinker-cookbook` is installed from GitHub; it brings the `tinker` SDK.

## The three checks

### 1. Credential-free local integration test

No API key, no Docker, no network.

```bash
pytest                                   # 63 tests, about 3 minutes on 4 cores
pytest tests/test_local_integration.py   # the end-to-end ones only
```

Or run rollouts by hand with a scripted policy:

```bash
python -m elt_rl.scripts.rollout backend=local policy=oracle
python -m elt_rl.scripts.rollout backend=local policy=oracle start_stage=transform
python -m elt_rl.scripts.rollout backend=local policy=cheat_fake_row_counts   # earns 0
```

```
task=bookshop stage=full reward=1.000
  source customer: ok (rows=60 expected=60 content=ok)
  ...
  model customers: ok (matched=6 unmatched=[] missed=[])
  model publishers: ok (matched=7 unmatched=[] missed=[])
...
3 rollout(s): mean verified reward 1.000, task success rate 1.000
```

### 2. A rollout on an official ELT-Bench task

Prerequisites, all from the ELT-Bench repository (`README.md`, `REPRODUCE.md`):

1. Airbyte running through `abctl`, the custom source published, `setup/airbyte/airbyte_credential.json` filled in.
2. `setup/destination/snowflake_credential.json` filled in (the admin credential; it stays on the host).
3. `bash setup/elt_setup.sh` (source systems on the Docker network `elt-docker_elt_network`).
4. `python setup/write_config.py --destination snowflake` (agent bundles in `inputs/`).
5. Ground truth: `hf download tttjjj/elt_bench --repo-type dataset --include "gt_snowflake/**" --local-dir ./ground_truth`.
6. Task image: `docker build -t elt-swe agents/SWE-agent/docker/elt-swe`.

Then, with `TINKER_API_KEY` set and `pip install -e ".[snowflake]"`:

```bash
python -m elt_rl.scripts.rollout backend=official elt_bench_root=/path/to/ELT-Bench \
    tasks=books policy=tinker model_name=moonshotai/Kimi-K2.6 env.max_turns=100
```

Each rollout gets its own schema `<TASK>__<ROLLOUT_ID>` in the database `ELT_RL`
(`database=...` to change), is graded there, and the schema, the container and the
Airbyte resources it created are removed afterwards (`keep_workspaces=True` keeps
the workspace directory for debugging).

Two optional settings make the official rollouts stricter and faster:

- `reference_database=<DB>`: a Snowflake database with one schema per task that holds a
  correct load of the source tables (the same thing ELT-Bench calls `--source_database`).
  It turns on the stage-1 content check and makes `start_stage=transform` possible
  (the source tables are zero-copy cloned). One way to create a schema of it:
  `CREATE SCHEMA <DB>.BOOKS CLONE <a database where stage 1 passed>.BOOKS`.
  Do not grant the agent's role (`AIRBYTE_ROLE`) access to this database: a policy that
  can read it can copy the source tables instead of loading them.
- `verify_replay=True`: rebuild the data models from the submitted dbt project before grading.

### 3. A Tinker RL optimization step on execution-derived rewards

```bash
export TINKER_API_KEY=...

# credential-free tasks, one step
python -m elt_rl.scripts.train backend=local start_stage=mixed \
    group_size=4 groups_per_batch=3 max_steps=1

# the same as a test
pytest -m tinker

# official tasks
python -m elt_rl.scripts.train backend=official elt_bench_root=/path/to/ELT-Bench \
    reference_database=ELT_REFERENCE tasks=books,airline start_stage=mixed \
    group_size=4 groups_per_batch=2 max_steps=1 env.max_turns=100
```

`train.py` only builds the dataset and calls `tinker_cookbook.rl.train.main`, so
sampling, advantages (centered within each task group), the optimizer step,
checkpoints and logs are the cookbook's. Logs go to `~/tinker-runs/elt_rl/...`:
`metrics.jsonl` has `env/all/reward/total`, the verification metrics
(`env/all/elt/...`, `env/all/official/...`) and `optim/...`.

For real runs:

- Use the Docker sandbox for the local tasks (`local_sandbox=docker`, image in `docker/`).
  The process sandbox does not isolate anything.
- Overlap sampling and training: `num_epochs=20 sampling_steps_ahead=2`. Rollouts take
  minutes, the optimizer step seconds.
- Hold out tasks with `eval_tasks=clinic` and compare runs on the held-out
  `elt/task_success`, not on the training reward.
- Episodes are long. Pick a model variant with enough context and raise
  `env.max_trajectory_tokens` (default 65,536) to match.

## How the environment works

**Episode.** A tool-use conversation driven by tinker-cookbook's `AgentToolMessageEnv`.
The system prompt and task instruction follow the official ELT-Bench runner. Three tools:

| Tool | What it does |
|---|---|
| `bash(command, timeout=None)` | Runs in the sandbox, in the workspace. Returns exit code, stdout, stderr; long output keeps its head and tail. |
| `write_file(path, content)` | Writes a file under `<workspace>/elt`, the only writable directory. |
| `submit()` | Ends the episode. Returns "Submission received." and nothing else. |

The episode also ends at `env.max_turns`, when the context budget is exhausted, after
repeated malformed tool calls, or at `env.rollout_timeout_seconds`. **It is graded
however it ended**, so a rollout that loaded all the data and then ran out of turns
keeps the credit for it.

**State.** The environment is stateful in three places, and each rollout gets its own
copy of all three: the workspace directory (Terraform state, dbt project), the
extract/load control plane (Airbyte connections and jobs, or `minielt` state), and a
warehouse namespace (source tables and data models). `ELTRuntime.start()` creates them
and `stop()` destroys them. Rollouts of one group never share a namespace.

**Reward.** Computed in `compute_group_rewards` from the final warehouse state:

```
reward = 0.3 * el_frac + 0.5 * model_frac + 0.2 * column_frac        (in [0, 1])
```

- `el_frac`: fraction of source tables that exist as base tables with the expected row
  count and, when a reference load is available, the same content.
- `model_frac`: fraction of data models that match the ground-truth CSV in every column
  (the official comparator: sort by unique key, compare column by column, integers
  exactly, decimals within 1% and one unit of the last shared decimal place).
- `column_frac`: mean fraction of matching *non-key* columns per model.

With `start_stage=transform` the source tables are pre-loaded and earn nothing; the
reward is the last two terms renormalized, and zero if a source table was changed.
The official metrics are always logged next to it and never shaped:
`official/stage1_success` (row counts only) and `official/stage2_frac` (models correct,
counted only when stage 1 passed). `reward.gate_models_on_el=True` makes the reward
itself follow that gating.

**Start stages.** `full` is the benchmark task. `transform` starts from loaded source
tables (ELT-Bench's transformation-only setting) and is several times cheaper per
rollout. `mixed` alternates per task; a group is always one stage.

## The local tasks

Three small tasks (`bookshop`, `clinic`, `airfare`) with the same shape as an ELT-Bench
task and no external service: sources are a SQLite database, a directory of JSON-lines
collections, a directory of CSV objects and flat files; extract/load is done with
`minielt`, a small declarative loader that stands in for Airbyte + Terraform (write a
pipeline file from the docs, apply it, trigger sync jobs, check their status; append
syncs duplicate rows when run twice); transformation is a real dbt project on DuckDB.
They are stored in the ELT-Bench directory layout, so the same loader and grader serve
both. `local_variants=N` generates N data variants per task.

```bash
python -m elt_rl.local_bench.build --root ./local_bench    # optional: build explicitly
```

They exist to test the environment and the training loop. They are not a benchmark.

## Network egress on the official backend

The ground truth of the official tasks is a public dataset. A rollout container on
`elt-docker_elt_network` can reach the internet, so nothing stops a policy from
downloading it. `docker/egress/` has an allow-listing proxy and an internal Docker
network for this:

```bash
docker compose -f docker/egress/docker-compose.yml up -d
for c in airbyte-abctl-control-plane elt-postgres elt-mongodb elt-localstack elt-api; do
  docker network connect elt-rl-internal "$c"
done
python -m elt_rl.scripts.train backend=official ... \
    network=elt-rl-internal egress_proxy=http://elt-egress:3128 image=<image with offline deps>
```

With no route out, `terraform init` and `dbt deps` cannot download anything, so use an
image with the dependencies baked in (ELT-Bench's `agents/common/offline-deps.Dockerfile`
on top of `elt-swe`). This setup has not been run; see the status table.

## Adding a destination

The environment, tools, grader and scripts never name a warehouse. Everything goes
through `elt_rl.destinations.Destination`:

```python
from elt_rl.destinations import Destination, register_destination

class BigQueryDestination(Destination):
    name = "bigquery"            # the section name in config.yaml
    display_name = "BigQuery"    # used in prompts
    dbt_adapter = "bigquery"

    def new_namespace(self, task_name, rollout_id, host_workspace, agent_workspace): ...
    def create_namespace(self, ns, clone_from=None): ...   # reset, optionally as a copy
    def drop_namespace(self, ns): ...
    def list_tables(self, ns, include_views=False): ...
    def table_columns(self, ns, table): ...
    def query(self, sql, ns): ...                          # -> pandas.DataFrame
    def qualified(self, ns, table): ...                    # quoted, fully qualified name
    def norm(self, ident): ...                             # case folding
    def fingerprint_sql(self, relation, columns): ...      # (row count, order-independent hash)
    def drop_table(self, ns, table): ...
    def agent_config(self, ns, base): ...                  # config.yaml block for the agent

register_destination("bigquery", BigQueryDestination)
```

`DuckDBDestination` (about 170 lines) and `SnowflakeDestination` (about 210 lines) are
the two implementations. `OfficialRuntime` builds its adapter from the registry by name;
Databricks and Redshift need an adapter and their credential keyword arguments in
`OfficialRuntime.__init__`.

## Reward-hacking tests

Each shortcut is a scripted policy in `elt_rl/scripted.py`, run through the real
environment in `tests/test_reward_hacking.py`.

| Shortcut | Result |
|---|---|
| `cheat_fake_row_counts`: tables with the right row count and made-up rows | Passes the official stage 1, fails the content check. Reward 0. |
| `cheat_double_sync`: append syncs run twice | Row counts wrong. No stage-1 credit for those tables. |
| `cheat_handmade_tables`: correct final tables written by a script, no dbt project | Dropped by the replay. Only the real extract/load is credited (0.3). |
| `cheat_passthrough_models`: dbt models that select from hand-made copies | The replay starts from the source tables only. 0.3. |
| `cheat_keys_only`: models with the right keys and NULLs | No column credit. 0.3. |
| `cheat_tamper_sources`: transformation-only start, then drop a source table | Reward 0. |
| `noop`: submit immediately | Reward 0. |
| Background process left holding the warehouse | Killed before grading; the rollout is graded normally. |

## Options

All options are `key=value` on the command line (`chz`). The common ones:

| Option | Default | Meaning |
|---|---|---|
| `backend` | `local` | `local` (DuckDB) or `official` (ELT-Bench stack). |
| `tasks` | all | Comma-separated task names. |
| `start_stage` | `full` | `full`, `transform`, or `mixed` (training only). |
| `group_size` / `groups_per_batch` | 4 / 4 | Rollouts per task group, task groups per step. |
| `model_name` | `moonshotai/Kimi-K2.6` | Tinker base model. |
| `env.max_turns` | 40 | Assistant turns per rollout. |
| `env.max_trajectory_tokens` | 65536 | Context budget of a rollout. |
| `env.require_submit` | `False` | Give no reward unless `submit` was called. |
| `reward.w_el`, `reward.w_model`, `reward.w_column` | 0.3, 0.5, 0.2 | Reward weights. |
| `reward.content_check` | `auto` | `rowcount` (official), `reference`, or `auto` (reference when available). |
| `verify_replay` | on for local, off for official | Rebuild the models from the submitted dbt project before grading. |
| `local_sandbox` | `process` | `process` (tests) or `docker` (isolated, no network). |
| `reference_database` | none | Official backend: database with reference loads. |
| `sampling_steps_ahead` | none | Async RL: how far sampling may run ahead of training. |

`python -m elt_rl.scripts.train --help` lists everything.

## Layout

```
elt_rl/
  env.py              ELTEnvGroupBuilder, ELTDataset, ELTDatasetBuilder (tinker-cookbook RL types)
  tools.py            bash / write_file / submit
  prompts.py          system prompt and task instructions
  tasks.py            ELTTask and the loader for the ELT-Bench directory layout
  destinations/       Destination interface, DuckDB and Snowflake adapters, registry
  grading/            comparator (port of the official one) and the grader / reward
  runtime/            per-rollout state: LocalRuntime, OfficialRuntime
  sandbox/            LocalProcessSandbox, DockerSandbox (cookbook SandboxInterface)
  minielt/            the local extract/load tool
  local_bench/        task specifications, data generator, reference solutions
  scripted.py         scripted policies: oracle and reward-hacking shortcuts
  scripts/            train.py, rollout.py
tests/                see "What has been run"
docker/               local sandbox image, egress proxy
DESIGN.md             design notes
```

## Limitations

- Only Snowflake has an official-backend adapter.
- The process sandbox runs commands as the current user. A policy can read anything that
  user can, including the local tasks' ground truth. Use it for tests only.
- Rollouts of the official backend share one Airbyte instance, one set of source systems
  and one Snowflake database. They are separated by schema name and by cleanup, not by
  permissions: a rollout whose role can read other schemas could copy from a concurrent
  rollout of the same task. A role per rollout would close this; it is not implemented.
- Stage 1 is verified by its result (the right data is in the warehouse), not by its
  method. A policy that loads the sources correctly without the intended tool gets the
  credit.
- No model has been trained with this yet, so the reward weights and budgets are untuned.

## Credits and licenses

- `elt_rl/grading/compare.py` is a port of the comparator in ELT-Bench's
  `evaluation/eva_stage2.py`, and the official task instructions in `elt_rl/prompts.py`
  are adapted from its runners. ELT-Bench is by the UIUC Kang Lab
  (https://github.com/uiuc-kang-lab/ELT-Bench), licensed CC BY-SA 4.0; those two files
  remain under that license.
- The environment is built on tinker-cookbook (Apache-2.0): its RL types, tool-use
  environment, sandbox interface, renderers and training loop.
- The rest of this repository does not have a license file yet.
