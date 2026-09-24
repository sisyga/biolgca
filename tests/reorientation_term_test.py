"""Reorientation terms written with @reorientation_term steer the Boltzmann sampler."""

import numpy as np
import pytest

from lgca import reorientation_term
from lgca.model import (
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    run_model,
)
from lgca.pipeline import (
    _REORIENTATION_TERMS,
    _TERM_ALIASES,
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
    list_reorientation_terms,
)


@pytest.fixture(autouse=True)
def _restore_terms():
    terms, aliases = dict(_REORIENTATION_TERMS), dict(_TERM_ALIASES)
    yield
    _REORIENTATION_TERMS.clear(), _REORIENTATION_TERMS.update(terms)
    _TERM_ALIASES.clear(), _TERM_ALIASES.update(aliases)


def _drift():
    @reorientation_term(coupling="flux")
    def drift(state, direction=(1.0, 0.0)):
        """Cells move in a fixed direction.

        Parameters
        ----------
        direction : sequence of float
            Direction of motion.
        """
        return np.asarray(direction, dtype=float)

    return drift


def _mean_flux(terms, geometry="square", n_species=1, steps=1):
    result = run_model(ModelSpec(
        description=Description(title="term"),
        space=SpaceSpec(geometry=geometry, dims=(30, 30)),
        state=StateSpec(density=1.0, restchannels=1, n_species=n_species),
        time=TimeSpec(steps=steps, seed=4),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=terms)], propagation=False),
    ), showprogress=False)
    lgca = result.lgca
    nodes = lgca.nodes[lgca.nonborder]
    per_species = nodes if n_species > 1 else nodes[..., None, :]
    return np.stack([lgca.calc_flux(per_species[..., s, :].astype(float)).sum(axis=(0, 1))
                     for s in range(n_species)]) / nodes.sum()


def test_calling_a_term_gives_its_spec():
    drift = _drift()

    assert drift(beta=2.0, direction=[0, 1]) == ReorientationTermSpec(
        name="drift", beta=2.0, parameters={"direction": [0, 1]})
    assert "drift" in list_reorientation_terms()
    assert "direction (default (1.0, 0.0))" in str(drift)
    with pytest.raises(ValueError, match="did you mean 'direction'"):
        drift(directoin=[0, 1])


@pytest.mark.parametrize("geometry", ["square", "hex"])
def test_a_flux_term_moves_cells_along_its_field(geometry):
    drift = _drift()

    flux = _mean_flux([drift(beta=3.0, direction=[0, 1])], geometry)[0]

    assert flux[1] > 0.3 and abs(flux[0]) < 0.05


def test_a_term_can_act_on_one_species():
    drift = _drift()

    flux = _mean_flux([drift(beta=3.0, species=1)], n_species=2)

    assert flux[1, 0] > 0.15 and abs(flux[0, 0]) < 0.03


def test_nematic_rest_and_channel_couplings():
    @reorientation_term(coupling="nematic")
    def vertical_axis(state):
        """Cells move along the y axis."""
        return np.diag([0.0, 1.0])

    @reorientation_term(coupling="rest")
    def stay(state, strength=1.0):
        """Cells rest."""
        return np.full(state.dims, strength)

    @reorientation_term(coupling="channels")
    def east_only(state):
        """All channels but east are penalised."""
        weights = np.full(state.K, -5.0)
        weights[0] = 0
        return weights

    vertical = _mean_flux([vertical_axis(beta=5.0)])[0]
    assert abs(vertical[1]) < 0.05  # up and down equally

    compiled = build_model(ModelSpec(
        description=Description(title="couplings"),
        space=SpaceSpec(geometry="square", dims=(30, 30)),
        state=StateSpec(density=1.0, restchannels=1),
        time=TimeSpec(steps=1, seed=4),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[stay(beta=10.0)])],
                                         propagation=False),
    ))
    compiled.step()
    interior = compiled.lgca.nodes[compiled.lgca.nonborder]
    assert interior[..., 4].sum() == (interior.sum(-1) > 0).sum()  # every occupied node fills its rest channel
    assert _mean_flux([east_only(beta=1.0)])[0][0] > 0.4


def test_a_field_of_the_wrong_shape_is_explained():
    @reorientation_term(coupling="flux")
    def broken(state):
        """Returns one number instead of a vector per node."""
        return np.ones(state.dims + (3,))

    with pytest.raises(ValueError, match=r"terms\[0\] broken must return an array that broadcasts"):
        build_model(ModelSpec(
            space=SpaceSpec(geometry="square", dims=(4, 4)),
            dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[broken()])]),
        ))


def test_beta_and_species_are_reserved():
    with pytest.raises(TypeError, match="beta"):
        reorientation_term(coupling="flux")(lambda state, beta=1.0: 0.0)


def test_terms_that_read_fields_list_them_as_inputs():
    signal = np.arange(16.0).reshape(4, 4)
    compiled = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(4, 4)),
        state=StateSpec(density=0.5, restchannels=1, fields={"signal": signal}),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
            ReorientationTermSpec("chemotaxis", parameters={"field": "signal"})])]),
    ))

    assert "inputs=signal" in compiled.pipeline.describe_schedule()


def test_redefining_a_term_in_the_same_module_replaces_it():
    _drift()
    second = _drift()

    assert _REORIENTATION_TERMS["drift"] is second
    with pytest.raises(ValueError, match="already registered"):
        reorientation_term(coupling="flux", name="aggregation")(lambda state: 0.0)
