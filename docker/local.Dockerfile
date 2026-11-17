# Sandbox image for the credential-free local tasks (local_sandbox=docker).
#
#   docker build -t elt-rl-local -f docker/local.Dockerfile .
#
# Everything a rollout needs is installed here, because rollout containers run
# with `--network none`: nothing can be downloaded (or uploaded) at run time.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        "duckdb>=1.1" \
        "dbt-core>=1.8" \
        "dbt-duckdb>=1.8" \
        "pandas>=2.0" \
        "pyyaml>=6.0"

# minielt is a single self-contained file.
COPY elt_rl/minielt/cli.py /usr/local/bin/minielt
RUN chmod 755 /usr/local/bin/minielt

ENV DBT_SEND_ANONYMOUS_USAGE_STATS=false \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["sleep", "infinity"]
