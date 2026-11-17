"""Prompts for the ELT agent.

The official task text is taken from the ELT-Bench runners
(``agents/codex_agent/run.py`` and ``agents/common/transform_only.py``,
https://github.com/uiuc-kang-lab/ELT-Bench, CC BY-SA 4.0) with two changes:
paths are templated on the sandbox workspace, and the episode ends with the
``submit`` tool instead of a printed ``RESULT:`` line, because a result the
model merely reports is never trusted.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a data engineer skilled in databases, SQL, and building ELT pipelines.

You are working in a sandbox. Your working directory is `{workspace}`. Your tools execute inside the sandbox; relative paths are resolved against {workspace}. You may only modify files under {workspace}/elt.

{goal}

TOOL USAGE:
  - bash: run a shell command in the sandbox ({tooling}, plus everyday utilities such as cat, ls, grep, find, head, tail) and get its exit code and output.
  - write_file: create or overwrite a file under {workspace}/elt. Use this for all file edits.
  - submit: declare the pipeline finished. This ends the task, and the tables in {destination} are then verified. Call it once, when the final tables are populated, or when you cannot make further progress.

First, you should always include a general thought about what you're going to do next. Then issue the corresponding tool call. Wait for the tool result before continuing with more discussion and commands."""

FULL_GOAL = """Your goal is to build an ELT pipeline by extracting data from multiple sources, such as {source_kinds}.
The extracted data will be loaded into a target system, {destination}, followed by writing transformation queries to construct final tables for downstream use.
This task is divided into two stages:
1. Data Extraction and Loading - Using {el_tool}.
2. Data Transformation - Using the DBT Project workflow for {destination}."""

TRANSFORM_GOAL = """Your goal is to build the data transformation stage of an ELT pipeline. The source data has already been extracted and loaded into the target system, {destination}. You write transformation queries to construct final tables for downstream use, using the DBT Project workflow for {destination}."""

# -- official ELT-Bench (Airbyte + Terraform) -------------------------------------
OFFICIAL_FULL_INSTRUCTION = """We're currently building the ELT pipeline using Airbyte Terraform and DBT. Here's the detailed instruction:

INSTRUCTIONS:
 # Stage 1: Data Extraction and Loading Hints#
1. Initialize the Airbyte Provider: Use {workspace}/config.yaml to configure the username, password, and server URL in {workspace}/elt/main.tf. You can refer to {workspace}/documentation/airbyte_Provider.md, then run `terraform init`.
• Important: You must not modify any provided code in {workspace}/elt/main.tf.
2. Configure Sources and Destinations: Use {workspace}/config.yaml to set up the listed sources and destinations in in Terraform file located in {workspace}/elt. The values in config.yaml represent the actual configurations required for the project.
3. Refer to Documentation for Configuration Guidance: You must consult the Airbyte documentation located in {workspace}/documentation to understand how to configure sources and destinations.
• Important: The values in the documentation are for reference only. DO NOT use them directly. And you must not fill the fields with random values or placeholders. Instead, fill in the fields based on the field names in {workspace}/documentation and the realistic values specified in {workspace}/config.yaml. The field names in {workspace}/documentation and {workspace}/config.yaml may not match exactly. In such cases, you need to infer values for certain fields. If you forget the realistic values for configuring sources and destinations in Terraform file located in {workspace}/elt, you must go back and read {workspace}/config.yaml to retrieve them.
4. Establish Connections: After creating sources and destinations, establish connections between them by following {workspace}/documentation/connection.md. For each connection, include the name and sync_mode fields in configuration.streams to streamline only the tables defined in {workspace}/config.yaml.
5. Apply the Configuration: Once all necessary configurations are written, run `terraform apply` to create the resources.
6. Retrieve Connection IDs: After successfully creating sources, destinations and connections, obtain the connection IDs for all newly created connections from the terraform.tfstate file, as these will be needed to trigger the jobs.
7. Trigger All Sync Jobs Using the Airbyte API: Use the Airbyte API with every retrieved connection ID. Refer to {workspace}/documentation/trigger_job.md and use {workspace}/config.yaml for the required values.
8. Monitor Job Status: After all jobs have been triggered, check the status of all jobs by running `python {workspace}/check_job_status.py --server <server> --username <username> --password <password>` (replace <server>, <username>, and <password> with the actual values). Keep checking until all jobs have completed. Proceed to the next stage only if all data extraction and loading jobs are successful. If any job fails, fix it and retry before moving on.
9. Error Handling: If an error occurs, DO NOT modify any provided files. Instead, verify the configuration against the provided documentation and review previous steps to identify and resolve the issue. If the issue persists after multiple retries, terminate the task.

#  Stage 2: Data Transformation Hints#
1. Initialize the DBT Project: Set up a new DBT project by configuring it with {workspace}/config.yaml, and remove the example directory under the models directory.
2. Understand the Data Model: Review data_model.yaml in {workspace} to understand the required data models and their column descriptions. Then, write SQL queries to generate these defined data models, referring to the files in the {workspace}/schemas directory to understand the schemas of source tables.
• Important: Write a separate query for each data model, and if using any DBT project variables, ensure they have already been declared.
3. Validate Table Locations: Ensure all SQL queries reference the correct database and schema names for source tables. If you encounter a "table not found" error, refer to {workspace}/config.yaml to obtain the correct configuration. If the table does not exist, the issue is likely due to a failure in data extraction and loading in Stage 1, and you should return to Stage 1 to resolve it.
4. Run the DBT Project: Execute `dbt run` to apply transformations and generate the final data models in {destination}, fixing any errors reported by DBT.
5. Terminate the Task: Call the `submit` tool if all transformations align with data_model.yaml and the final tables in {destination} are accurate and verified. Alternatively, call `submit` if you are unable to resolve the issues after multiple retries."""

OFFICIAL_TRANSFORM_INSTRUCTION = """We're currently building the data transformation stage of the ELT pipeline using DBT. Data extraction and loading is already complete: every source table described in {workspace}/schemas has been loaded into the {destination} database and schema given in {workspace}/config.yaml. Here's the detailed instruction:

INSTRUCTIONS:
#  Data Transformation Hints#
1. Initialize the DBT Project: Set up a new DBT project by configuring it with {workspace}/config.yaml, and remove the example directory under the models directory.
2. Understand the Data Model: Review data_model.yaml in {workspace} to understand the required data models and their column descriptions. Then, write SQL queries to generate these defined data models, referring to the files in the {workspace}/schemas directory to understand the schemas of source tables.
• Important: Write a separate query for each data model, and if using any DBT project variables, ensure they have already been declared.
3. Validate Table Locations: Ensure all SQL queries reference the correct database and schema names for source tables. If you encounter a "table not found" error, refer to {workspace}/config.yaml to obtain the correct configuration. Do not modify, drop, or reload the source tables.
4. Run the DBT Project: Execute `dbt run` to apply transformations and generate the final data models in {destination}, fixing any errors reported by DBT.
5. Terminate the Task: Call the `submit` tool if all transformations align with data_model.yaml and the final tables in {destination} are accurate and verified. Alternatively, call `submit` if you are unable to resolve the issues after multiple retries."""

# -- local benchmark (minielt) -----------------------------------------------------
LOCAL_FULL_INSTRUCTION = """We're currently building the ELT pipeline using minielt and DBT. Here's the detailed instruction:

INSTRUCTIONS:
 # Stage 1: Data Extraction and Loading Hints#
1. Read the Configuration: {workspace}/config.yaml lists every source system, the tables to extract from it, the sync mode, and the destination. The values in config.yaml represent the actual configurations required for the project. The source data is under {workspace}/sources and must not be modified.
2. Refer to Documentation for Configuration Guidance: You must consult {workspace}/documentation/minielt.md to understand how to declare sources, the destination and connections. The values in the documentation are for reference only; fill in the fields from {workspace}/config.yaml.
3. Write the Pipeline: Declare all sources, the destination and the connections in {workspace}/elt/pipeline.yaml, streaming only the tables defined in {workspace}/config.yaml, each loaded into a table of the same name.
4. Apply the Configuration: Run `minielt apply -f elt/pipeline.yaml` from {workspace} to register the connections and obtain their connection IDs.
5. Trigger Sync Jobs: Trigger a sync job for every connection with `minielt sync`.
6. Monitor Job Status: Run `minielt status` and proceed to the next stage only if all extraction and loading jobs succeeded. If any job fails, fix the pipeline and retry before moving on.
7. Error Handling: If an error occurs, DO NOT modify any provided files. Verify the configuration against the documentation and review the previous steps to resolve the issue.

#  Stage 2: Data Transformation Hints#
1. Initialize the DBT Project: Set up a new DBT project under {workspace}/elt, configured for the destination in {workspace}/config.yaml (see {workspace}/documentation/destination_{destination_key}.md).
2. Understand the Data Model: Review data_model.yaml in {workspace} to understand the required data models and their column descriptions. Then, write SQL queries to generate these defined data models, referring to the files in the {workspace}/schemas directory to understand the schemas of source tables.
• Important: Write a separate query for each data model, named after the data model, and create it in the same schema as the source tables.
3. Validate Table Locations: Ensure all SQL queries reference the correct schema names for source tables. If the table does not exist, the issue is likely due to a failure in data extraction and loading in Stage 1, and you should return to Stage 1 to resolve it.
4. Run the DBT Project: Execute `dbt run` to apply transformations and generate the final data models in {destination}, fixing any errors reported by DBT.
5. Terminate the Task: Call the `submit` tool if all transformations align with data_model.yaml and the final tables in {destination} are accurate and verified. Alternatively, call `submit` if you are unable to resolve the issues after multiple retries."""

LOCAL_TRANSFORM_INSTRUCTION = """We're currently building the data transformation stage of the ELT pipeline using DBT. Data extraction and loading is already complete: every source table described in {workspace}/schemas has been loaded into the {destination} database and schema given in {workspace}/config.yaml. Here's the detailed instruction:

INSTRUCTIONS:
#  Data Transformation Hints#
1. Initialize the DBT Project: Set up a new DBT project under {workspace}/elt, configured for the destination in {workspace}/config.yaml (see {workspace}/documentation/destination_{destination_key}.md).
2. Understand the Data Model: Review data_model.yaml in {workspace} to understand the required data models and their column descriptions. Then, write SQL queries to generate these defined data models, referring to the files in the {workspace}/schemas directory to understand the schemas of source tables.
• Important: Write a separate query for each data model, named after the data model, and create it in the same schema as the source tables.
3. Validate Table Locations: Ensure all SQL queries reference the correct schema names for source tables. Do not modify, drop, or reload the source tables.
4. Run the DBT Project: Execute `dbt run` to apply transformations and generate the final data models in {destination}, fixing any errors reported by DBT.
5. Terminate the Task: Call the `submit` tool if all transformations align with data_model.yaml and the final tables in {destination} are accurate and verified. Alternatively, call `submit` if you are unable to resolve the issues after multiple retries."""


REPLAY_NOTE = (
    "\n\nYour submission is verified by running `dbt run` once more from your dbt project "
    "directory, starting from the source tables only: every other table in the schema is "
    "dropped first. The project must therefore build every data model from the source tables, "
    "from that directory, without extra flags."
)


def render_system_prompt(
    *,
    workspace: str,
    destination: str,
    transform_only: bool,
    source_kinds: str,
    el_tool: str,
    tooling: str,
) -> str:
    goal = (
        TRANSFORM_GOAL.format(destination=destination)
        if transform_only
        else FULL_GOAL.format(source_kinds=source_kinds, destination=destination, el_tool=el_tool)
    )
    return SYSTEM_PROMPT.format(
        workspace=workspace, goal=goal, tooling=tooling, destination=destination
    )
