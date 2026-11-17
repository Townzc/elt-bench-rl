"""Subprocess sandbox for local development and the credential-free tests.

Implements tinker-cookbook's ``SandboxInterface`` by running commands in a
per-rollout workspace directory. It provides *no* security isolation: commands
run as the current user and can read the host filesystem. Use it for tests and
trusted debugging; use :class:`~elt_rl.sandbox.docker.DockerSandbox` (or a
remote sandbox) whenever a learning policy controls the commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import uuid
from pathlib import Path

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of ``text``; errors usually sit at the end."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


class LocalProcessSandbox:
    """Run commands with ``bash -c`` inside a workspace directory."""

    def __init__(
        self,
        workspace: Path,
        env: dict[str, str] | None = None,
        remove_on_cleanup: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._id = f"local-{uuid.uuid4().hex[:8]}"
        self._remove_on_cleanup = remove_on_cleanup
        self._process_groups: set[int] = set()

        home = self.workspace / ".home"
        home.mkdir(exist_ok=True)
        base_env = {
            # Make the interpreter that runs the environment (and its dbt /
            # minielt entry points) the one the agent's commands resolve.
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")]
            ),
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
            "DO_NOT_TRACK": "1",
        }
        base_env.update(env or {})
        self._env = base_env

    @property
    def sandbox_id(self) -> str:
        return self._id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.workspace / p

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT_BYTES
        cwd = self._resolve(workdir) if workdir else self.workspace
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=str(cwd),
            env=self._env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group, so a timeout kills children too
        )
        self._process_groups.add(process.pid)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            exit_code = process.returncode if process.returncode is not None else -1
        except TimeoutError:
            self._kill_group(process.pid)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
            except TimeoutError:
                stdout, stderr = b"", b""
            stderr += f"\n[command timed out after {timeout}s and was killed]".encode()
            exit_code = 124
        return SandboxResult(
            stdout=truncate_middle(stdout.decode("utf-8", errors="replace"), limit),
            stderr=truncate_middle(stderr.decode("utf-8", errors="replace"), limit),
            exit_code=exit_code,
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        try:
            data = self._resolve(path).read_bytes()
        except OSError as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        if max_bytes is not None:
            data = data[:max_bytes]
        return SandboxResult(stdout=data.decode("utf-8", errors="replace"), stderr="", exit_code=0)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            if executable:
                target.chmod(target.stat().st_mode | 0o111)
        except OSError as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    def _kill_group(self, pgid: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

    async def kill_background(self) -> None:
        """Kill everything earlier commands left running (e.g. ``cmd &``).

        Such processes would keep files, and the DuckDB write lock, open.
        """
        for pgid in self._process_groups:
            self._kill_group(pgid)
        self._process_groups.clear()

    async def cleanup(self) -> None:
        await self.kill_background()
        if self._remove_on_cleanup:
            shutil.rmtree(self.workspace, ignore_errors=True)
