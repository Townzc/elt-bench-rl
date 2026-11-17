"""Sandbox backends implementing tinker-cookbook's ``SandboxInterface``."""

from elt_rl.sandbox.docker import DockerSandbox
from elt_rl.sandbox.local import LocalProcessSandbox

__all__ = ["DockerSandbox", "LocalProcessSandbox"]
