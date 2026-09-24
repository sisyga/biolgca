"""The legacy interaction functions of biolgca, kept as a reference for tests.

biolgca now runs the interaction names of ``get_lgca`` as stacks of rules
(:mod:`lgca.legacy_names`). The functions they replaced live here, with the
``set_interaction`` code that set them up (``setup.py``), so that tests can
compare the rules with them:

- ``legacy_lgca(interaction=..., **kwargs)`` builds a model like ``get_lgca``
  whose time steps run the legacy function;
- in a ``ModelSpec`` pipeline, ``{"name": "legacy.<family>.<name>"}`` (e.g.
  ``"legacy.classical.alignment"``) runs it as one operator.
"""

from lgca import get_lgca
from lgca.operator_base import (
    BirthDeathOperator,
    InteractionOperator,
    PhenotypeSwitchOperator,
    ReorientationOperator,
)
from lgca.plugins import default_registry, list_plugins, register_plugin

from .setup import set_legacy_interaction

__all__ = ["LegacyInteractionOperator", "legacy_lgca", "set_legacy_interaction"]

# get_lgca keywords that the legacy constructors used themselves
_CONSTRUCTOR = {"geometry", "ib", "ve", "n_species", "nodes", "dims", "restchannels", "density", "bc", "seed",
                "propagation", "r_int"}
# prefixed names whose legacy get_lgca name differs
_LEGACY_NAMES = {"classical.persistent_walk": "persistent_motion", "nove_ib.evo_steric": "steric_evolution",
                 "nove_ib.go_or_grow_kappa_chemo": "go_or_grow_kappa"}
# prefixed names whose function the legacy setup did not pick (a function of nove_ib_interactions)
_FUNCTIONS = {"nove_ib.go_or_grow_kappa_chemo": "go_or_grow_kappa_chemo"}


def _no_interaction(lgca):
    """Placeholder while the model is built; the legacy setup replaces it."""


def legacy_lgca(interaction=None, **kwargs):
    """``get_lgca(interaction=..., **kwargs)`` with the legacy interaction function."""
    lgca = get_lgca(interaction=_no_interaction, **kwargs)
    lgca.__dict__.pop("_compiled_model", None)
    parameters = {key: value for key, value in kwargs.items() if key not in _CONSTRUCTOR}
    if interaction is not None:
        parameters["interaction"] = interaction
    set_legacy_interaction(lgca, **parameters)
    return lgca


class LegacyInteractionOperator(InteractionOperator):
    """Pipeline operator that runs a legacy interaction function."""

    def __init__(self, info, legacy_interaction, parameters=None, function=None):
        super().__init__(info=info, parameters=parameters)
        self.legacy_interaction = legacy_interaction
        self.function = function  # a function other than the one the legacy setup picks
        self._interaction = None
        self._interaction_params = {}

    def setup(self, context) -> None:
        lgca = context.lgca
        previous = getattr(lgca, "interaction", None), dict(getattr(lgca, "interaction_params", {}))
        parameters = dict(self.parameters)
        capacity = context.spec.state.capacity
        if capacity is not None and "capacity" not in parameters:  # the model's, as the rules use
            parameters["capacity"] = capacity
        set_legacy_interaction(lgca, interaction=self.legacy_interaction, **parameters)
        if self.function is not None:
            from . import nove_ib_interactions

            lgca.interaction = getattr(nove_ib_interactions, self.function)
        self._interaction = lgca.interaction
        self._interaction_params = {**lgca.interaction_params, **parameters}
        lgca.interaction, lgca.interaction_params = previous

    def apply(self, context, step: int) -> None:
        if self._interaction is None:
            self.setup(context)
        lgca = context.lgca
        previous = getattr(lgca, "interaction", None), dict(getattr(lgca, "interaction_params", {}))
        lgca.interaction, lgca.interaction_params = self._interaction, dict(self._interaction_params)
        try:
            self._interaction(lgca)
        finally:
            lgca.interaction, lgca.interaction_params = previous


_BASES = {"birth_death": BirthDeathOperator, "phenotype_switch": PhenotypeSwitchOperator,
          "reorientation": ReorientationOperator}


def _register(info):
    from dataclasses import replace

    family, name = info.name.split(".", 1)
    legacy = replace(info, name=f"legacy.{info.name}", aliases=(), deprecated="",
                     backend_families=(family,))
    operator = type("LegacyOperator", (_BASES[info.operator_kind], LegacyInteractionOperator), {})
    interaction = _LEGACY_NAMES.get(info.name, name)

    def factory(parameters=None):
        return operator(legacy, interaction, {**_defaults(info), **dict(parameters or {})}, _FUNCTIONS.get(info.name))

    register_plugin(legacy, factory, replace=True)


def _defaults(info):
    """Legacy defaults that the rules compute (None) are left to the legacy setup."""
    return {key: spec.default for key, spec in info.parameter_specs.items()
            if spec.default is not None and key != "capacity"}


for _info in list_plugins(deprecated=True):
    if _info.deprecated and _info.name.split(".", 1)[0] in ("classical", "nove", "ib", "nove_ib", "multispecies"):
        _register(_info)
del _info, default_registry
