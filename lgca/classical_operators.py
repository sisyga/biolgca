"""Native interaction operators for classical volume-exclusion LGCA."""

from __future__ import annotations

from typing import Any, Mapping

from .operator_base import ConservationLaw, PluginInfo, ReorientationOperator


CLASSICAL_RANDOM_WALK_INFO = PluginInfo(
    name="classical.random_walk",
    aliases=("random_walk",),
    operator_kind="reorientation",
    backend_families=("classical",),
    legacy_source="lgca.interactions.random_walk",
    conservation_law=ConservationLaw(
        conserves_total_particles=True,
        conserves_phenotype_particles=True,
        conserves_momentum=False,
        changes=("channel occupancy",),
    ),
    port_status="native",
    test_status="unit_tested",
    description="Uniform channel permutation for volume-exclusion classical LGCA.",
)


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


def create_classical_random_walk(parameters=None):
    """Create the classical random-walk operator from its canonical definition."""

    return NativeClassicalRandomWalkOperator(CLASSICAL_RANDOM_WALK_INFO, parameters)


def register_classical_random_walk(register) -> None:
    """Register the complete classical random-walk vertical slice."""

    register(CLASSICAL_RANDOM_WALK_INFO, create_classical_random_walk)
