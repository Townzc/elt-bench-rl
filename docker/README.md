# Sandbox images

| Image | Used by | Build |
|---|---|---|
| `elt-rl-local` | `backend=local local_sandbox=docker` | `docker build -t elt-rl-local -f docker/local.Dockerfile .` (from the repository root) |
| `elt-swe` | `backend=official` | `docker build -t elt-swe agents/SWE-agent/docker/elt-swe` (from the ELT-Bench checkout, see its `REPRODUCE.md`) |

`backend=official` uses the benchmark's own task image, so the agent sees the
same Terraform, dbt and client versions as in the published runs. Pass another
tag with `image=...`, for example one built on top of `elt-swe` with
`agents/common/offline-deps.Dockerfile` when the container's network is
restricted.
