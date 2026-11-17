"""Docker sandbox: one container per rollout.

Mirrors how the official ELT-Bench runners isolate an agent:

* the task directory is bind-mounted read-only at ``/workspace`` and only
  ``/workspace/elt`` is writable, which enforces "you may only modify files
  under /workspace/elt" for every tool;
* the container joins the Docker network that reaches the source systems and
  Airbyte (``elt-docker_elt_network``), or runs with ``--network none`` for
  the hermetic local benchmark;
* the container runs as the host user so files stay host-owned.

Commands go through ``docker exec``; nothing is trusted from inside.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from pathlib import Path

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from elt_rl.sandbox.local import DEFAULT_MAX_OUTPUT_BYTES, truncate_middle

CONTAINER_WORKDIR = "/workspace"
CONTAINER_HOME = "/tmp/home"


def docker_run_argv(
    name: str,
    image: str,
    host_workspace: Path,
    network: str | None,
    writable_subdirs: tuple[str, ...] = ("elt",),
    uid: int | None = None,
    gid: int | None = None,
    env: dict[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """``docker run`` command line for a rollout container (pure, testable)."""
    uid = os.getuid() if uid is None else uid
    gid = os.getgid() if gid is None else gid
    mount = str(Path(host_workspace).resolve())
    argv = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        "--network", network or "none",
        "--user", f"{uid}:{gid}",
        # The agent gets no extra privileges; it cannot change its own firewall.
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-e", f"HOME={CONTAINER_HOME}",
        "-e", "USER=agent",
        "-e", "DBT_SEND_ANONYMOUS_USAGE_STATS=false",
        "-v", f"{mount}:{CONTAINER_WORKDIR}:ro",
    ]  # fmt: skip
    for sub in writable_subdirs:
        argv += ["-v", f"{mount}/{sub}:{CONTAINER_WORKDIR}/{sub}"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += [*extra_args, "-w", CONTAINER_WORKDIR, image, "sleep", "infinity"]
    return argv


class DockerSandbox:
    """A running container implementing tinker-cookbook's ``SandboxInterface``."""

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    async def create(
        cls,
        image: str,
        host_workspace: Path,
        network: str | None,
        writable_subdirs: tuple[str, ...] = ("elt",),
        env: dict[str, str] | None = None,
        extra_args: tuple[str, ...] = (),
        name: str | None = None,
    ) -> DockerSandbox:
        name = name or f"elt-rl-{uuid.uuid4().hex[:12]}"
        for sub in writable_subdirs:
            (Path(host_workspace) / sub).mkdir(parents=True, exist_ok=True)
        argv = docker_run_argv(
            name, image, host_workspace, network, writable_subdirs, env=env, extra_args=extra_args
        )
        code, _, err = await _run(argv, timeout=300)
        if code != 0:
            raise RuntimeError(f"docker run failed: {err.strip()}")
        sandbox = cls(name)
        uid, gid = os.getuid(), os.getgid()
        # The host uid has no passwd entry in the image; add one so tools that
        # look up the current user work, and create a writable HOME (~/.dbt).
        await _run(
            [
                "docker", "exec", "-u", "0", name, "sh", "-c",
                f"echo 'agent:x:{uid}:{gid}::{CONTAINER_HOME}:/bin/bash' >> /etc/passwd"
                f" && (getent group {gid} >/dev/null || echo 'agent:x:{gid}:' >> /etc/group)"
                f" && mkdir -p {CONTAINER_HOME} && chown {uid}:{gid} {CONTAINER_HOME}",
            ],  # fmt: skip
            timeout=60,
        )
        return sandbox

    @property
    def sandbox_id(self) -> str:
        return self.name

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT_BYTES
        # `timeout` inside the container kills the command's process tree; the
        # outer asyncio timeout is only a backstop for a stuck docker client.
        inner = f"timeout -k 5 {int(timeout)} bash -lc {shlex.quote(command)}"
        argv = ["docker", "exec", "-w", workdir or CONTAINER_WORKDIR, self.name, "sh", "-c", inner]
        code, out, err = await _run(argv, timeout=timeout + 30)
        if code == 124:
            err += f"\n[command timed out after {timeout}s and was killed]"
        return SandboxResult(
            stdout=truncate_middle(out, limit), stderr=truncate_middle(err, limit), exit_code=code
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        command = f"cat {shlex.quote(path)}"
        if max_bytes is not None:
            command = f"head -c {int(max_bytes)} {shlex.quote(path)}"
        code, out, err = await _run(
            ["docker", "exec", "-w", CONTAINER_WORKDIR, self.name, "sh", "-c", command],
            timeout=timeout,
        )
        return SandboxResult(stdout=out, stderr=err, exit_code=code)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        data = content.encode("utf-8") if isinstance(content, str) else content
        quoted = shlex.quote(path)
        command = f'mkdir -p "$(dirname {quoted})" && cat > {quoted}'
        if executable:
            command += f" && chmod +x {quoted}"
        code, out, err = await _run(
            ["docker", "exec", "-i", "-w", CONTAINER_WORKDIR, self.name, "sh", "-c", command],
            timeout=timeout,
            stdin=data,
        )
        return SandboxResult(stdout=out, stderr=err, exit_code=code)

    async def kill_background(self) -> None:
        """Kill every process in the container except its init (``sleep infinity``)."""
        await _run(["docker", "exec", self.name, "sh", "-c", "kill -9 -1"], timeout=30)

    async def cleanup(self) -> None:
        await _run(["docker", "rm", "-f", self.name], timeout=60)


async def _run(
    argv: list[str], timeout: float, stdin: bytes | None = None
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, "", f"docker command timed out after {timeout}s"
    return (
        process.returncode if process.returncode is not None else -1,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )
