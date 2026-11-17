"""Rollout runtimes: the stateful backends behind the environment."""

from __future__ import annotations

from elt_rl.runtime.base import ELTRuntime, RolloutContext, new_rollout_id
from elt_rl.runtime.local import LocalRuntime, LocalRuntimeConfig
from elt_rl.runtime.official import OfficialRuntime, OfficialRuntimeConfig

RuntimeConfig = LocalRuntimeConfig | OfficialRuntimeConfig

# One runtime per configuration per process: warehouse connections and the
# grader's reference fingerprints are shared by all groups instead of being
# rebuilt for each. Environment builders hold the (picklable) configuration
# and resolve the live runtime here.
_RUNTIMES: dict[str, ELTRuntime] = {}


def get_runtime(config: RuntimeConfig) -> ELTRuntime:
    key = repr(config)
    if key not in _RUNTIMES:
        _RUNTIMES[key] = config.build()
    return _RUNTIMES[key]


__all__ = [
    "ELTRuntime",
    "LocalRuntime",
    "LocalRuntimeConfig",
    "OfficialRuntime",
    "OfficialRuntimeConfig",
    "RolloutContext",
    "RuntimeConfig",
    "get_runtime",
    "new_rollout_id",
]
