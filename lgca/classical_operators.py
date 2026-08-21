"""Native interaction operators for classical volume-exclusion LGCA."""

from __future__ import annotations

from typing import Any, Mapping

from .operator_base import PluginInfo, ReorientationOperator


class NativeClassicalRandomWalkOperator(ReorientationOperator):
    """Uniformly permute each complete local channel state."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")

    def apply(self, context, step: int) -> None:
        context.lgca.nodes = context.lgca.rng.permuted(context.lgca.nodes, axis=-1)
