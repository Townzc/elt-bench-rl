"""Agent tools: a shell, a file writer, and an explicit submit action.

Built on tinker-cookbook's ``@tool`` decorator and ``SandboxInterface``, the
same building blocks as the ``harbor_rl`` recipe.
"""

from __future__ import annotations

import posixpath
from typing import Annotated

from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import Tool, ToolResult, simple_tool_result, tool

from elt_rl.sandbox.local import truncate_middle

DEFAULT_MAX_OUTPUT_CHARS = 8000


class ELTTools:
    """The three tools of one rollout, bound to its sandbox."""

    def __init__(
        self,
        sandbox: SandboxInterface,
        workspace: str,
        command_timeout: int = 120,
        max_command_timeout: int = 900,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._sandbox = sandbox
        self._workspace = workspace.rstrip("/")
        self._command_timeout = command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_chars = max_output_chars
        self.submitted = False
        self.commands_run = 0

    @property
    def tools(self) -> list[Tool]:
        return [self.bash, self.write_file, self.submit]

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
        timeout: Annotated[
            int | None,
            "Optional timeout in seconds, for commands that are expected to run long.",
        ] = None,
    ) -> ToolResult:
        """Execute a bash command in the sandbox and return its exit code and output.

        Commands start in the workspace directory. Each call is a fresh shell:
        environment variables and `cd` do not persist between calls.
        """
        seconds = self._command_timeout if timeout is None else timeout
        seconds = max(1, min(int(seconds), self._max_command_timeout))
        self.commands_run += 1
        result = await self._sandbox.run_command(
            command, workdir=self._workspace, timeout=seconds
        )
        parts = [f"exit_code: {result.exit_code}"]
        if result.stdout.strip():
            parts.append("stdout:\n" + truncate_middle(result.stdout, self._max_output_chars))
        if result.stderr.strip():
            parts.append("stderr:\n" + truncate_middle(result.stderr, self._max_output_chars))
        return simple_tool_result("\n".join(parts))

    @tool
    async def write_file(
        self,
        path: Annotated[str, "File path, absolute or relative to the workspace."],
        content: Annotated[str, "The full content of the file."],
    ) -> ToolResult:
        """Create or overwrite a file under the writable elt/ directory of the workspace."""
        target = path if posixpath.isabs(path) else posixpath.join(self._workspace, path)
        target = posixpath.normpath(target)
        writable = posixpath.join(self._workspace, "elt")
        if not (target + "/").startswith(writable + "/") or target == writable:
            return simple_tool_result(
                f"error: {target} is outside {writable}/, the only directory you may modify."
            )
        result = await self._sandbox.write_file(target, content)
        if result.exit_code != 0:
            return simple_tool_result(f"error: could not write {target}: {result.stderr.strip()}")
        return simple_tool_result(f"wrote {len(content)} characters to {target}")

    @tool
    async def submit(self) -> ToolResult:
        """Declare the pipeline finished. Ends the task; the warehouse is then verified."""
        self.submitted = True
        # No verification detail is returned: the grader is not an oracle the
        # agent can query within an episode.
        return simple_tool_result("Submission received.", should_stop=True)
