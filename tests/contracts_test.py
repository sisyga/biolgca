"""Rules keep the contract of their kind.

An operation that the kind does not allow warns once per rule and model (an
error under strict contracts, before anything changes) and is applied as the
wider kind would apply it. The cell arrays are read-only, states given to
reorientation terms and stack builders cannot be changed, and a model whose
step failed refuses further steps.
"""

import numpy as np
import pytest

from lgca import ContractWarning, interaction, reorientation_term, stack
from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    _REORIENTATION_TERMS,
    _TERM_ALIASES,
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.plugins import default_registry
from lgca.testing import strict_contracts

_IDENTITY = ("ib", "nove_ib")
_CLASSICAL = ("classical", "nove")


@pytest.fixture(autouse=True)
def _restore_registries():
    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)
    terms, term_aliases = dict(_REORIENTATION_TERMS), dict(_TERM_ALIASES)
    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases
    _REORIENTATION_TERMS.clear(), _REORIENTATION_TERMS.update(terms)
    _TERM_ALIASES.clear(), _TERM_ALIASES.update(term_aliases)


# rule bodies, with the narrowest kind that allows them; only the operations draw random numbers, so that
# a strict contract leaves the random stream as it was

def _half(cells):
    return np.arange(len(cells)) % 2 == 0


def _set_trait(state):
    state.cells.set_trait(_half(state.cells), "fitness", 0.9)


def _found_families(state):
    state.cells.found_families(_half(state.cells))


def _divide(state):
    state.cells.divide(_half(state.cells))


def _kill(state):
    state.cells.kill(_half(state.cells))


def _divide_and_kill_the_daughters(state):
    # the cells per node stay the same, but labels, traits and families were made
    cells = state.cells
    cells.kill(cells.divide(_half(cells)))


def _move(state):
    state.cells.move(np.ones(len(state.cells), dtype=bool), "all")


def _shuffle(state):
    state.shuffle_cells()


def _remove_cells(state):
    state.remove_cells(0.3)


def _divide_cells(state):
    state.divide_cells(0.3)


def _add_cells(state):
    state.add_cells(1)


def _switch_species(state):
    state.switch_phenotype([[0, 0.5], [0.5, 0]])


def _roll_counts(state):  # moves cells to another node: only assigned counts can do this
    state.counts = np.roll(state.counts, 1, axis=0)


_OPERATIONS = [
    (_set_trait, "phenotype_switch", "set traits", _IDENTITY),
    (_found_families, "phenotype_switch", "found families", _IDENTITY),
    (_divide, "birth_death", "divide cells", _IDENTITY),
    (_kill, "birth_death", "remove cells", _IDENTITY),
    (_divide_and_kill_the_daughters, "birth_death", "divide cells", _IDENTITY),
    (_move, "reorientation", "move cells", _IDENTITY),
    (_shuffle, "reorientation", "move cells", _IDENTITY + _CLASSICAL),
    (_remove_cells, "birth_death", "remove cells", _IDENTITY + _CLASSICAL),
    (_divide_cells, "birth_death", "divide cells", _IDENTITY + _CLASSICAL),
    (_add_cells, "birth_death", "add cells", _CLASSICAL),
    (_switch_species, "phenotype_switch", "switch species", _CLASSICAL),
    (_roll_counts, "birth_death", "change the number of cells at a node", _CLASSICAL),
]
_WIDTH = ("field", "reorientation", "phenotype_switch", "birth_death")
_CASES = [pytest.param(body, kind, needed, what, family, id=f"{body.__name__[1:]}-{kind}-{family}")
          for body, needed, what, families in _OPERATIONS for family in families
          for kind in _WIDTH[:_WIDTH.index(needed)]]


def _model(body, kind, family, steps=3):
    rule = interaction(kind=kind, families=family, name=f"{body.__name__[1:]}_as_{kind}")(body)
    identity = family in _IDENTITY
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=12),
        state=StateSpec(density=1.2, restchannels=1, identity_based=identity,
                        volume_exclusion=family in ("classical", "ib"),
                        n_species=1 if identity else 2,  # a classical phenotype switch needs two
                        traits={"fitness": 1.0} if identity else {}),
        time=TimeSpec(steps=steps, seed=4),
        dynamics=InteractionPipelineSpec(operators=[rule()]),
    ))


def _snapshot(model):
    """Cells (labels, nodes, channels), traits, labels and families handed out, and the random stream."""
    lgca = model.lgca
    traits = getattr(lgca, "props", None)
    if traits is None:
        return lgca.nodes.tolist(), lgca.rng.bit_generator.state
    cells = LatticeState(lgca).cells
    return ([cells.label.tolist(), cells.index.tolist(), cells.channel.tolist()],
            {name: np.asarray(values).tolist() for name, values in traits.items()}, int(lgca.maxlabel),
            int(getattr(lgca, "maxfamily", 0)), repr(getattr(lgca, "family_props", None)),
            lgca.rng.bit_generator.state)


@pytest.mark.parametrize("body, kind, needed, what, family", _CASES)
def test_strict_contracts_stop_a_rule_before_it_changes_the_model(body, kind, needed, what, family):
    model = _model(body, kind, family)
    before = _snapshot(model)

    with strict_contracts(), pytest.raises(ContractWarning, match=f"a {kind} rule may not {what}.*kind='{needed}'"):
        model.step()

    assert _snapshot(model) == before


@pytest.mark.parametrize("body, kind, needed, what, family", _CASES)
def test_a_rule_that_breaks_its_kind_warns_once_and_runs_as_the_wider_kind(body, kind, needed, what, family):
    model = _model(body, kind, family)
    with pytest.warns(ContractWarning, match=f"_as_{kind}: a {kind} rule may not {what}") as caught:
        model.run(showprogress=False)
    reference = _model(body, "birth_death", family)
    reference.run(showprogress=False)

    assert len([warning for warning in caught if warning.category is ContractWarning]) == 1  # not every step
    assert _snapshot(model) == _snapshot(reference)  # cells, traits, labels, families and random numbers
    # a model built again warns again
    with pytest.warns(ContractWarning):
        _model(body, kind, family).step()


@pytest.mark.parametrize("body, kind, family", [
    pytest.param(body, kind, family, id=f"{body.__name__[1:]}-{kind}-{family}")
    for body, needed, _, families in _OPERATIONS for family in families
    for kind in _WIDTH[_WIDTH.index(needed):]])
def test_a_rule_may_use_what_its_kind_allows(body, kind, family):
    with strict_contracts():
        _model(body, kind, family).run(showprogress=False)


@pytest.mark.parametrize("write, error", [
    (lambda cells: cells.channel.__setitem__(slice(None), 0), ValueError),
    (lambda cells: cells.label.__setitem__(0, cells.label[1]), ValueError),
    (lambda cells: cells.index.__iadd__(1), ValueError),
    (lambda cells: setattr(cells, "index", cells.index + 1), AttributeError),
    (lambda cells: setattr(cells, "channel", np.zeros(len(cells), dtype=int)), AttributeError),
])
@pytest.mark.parametrize("family", _IDENTITY)
def test_the_cell_arrays_are_read_only(write, error, family):
    # a direct write can lose cells or duplicate labels, which no warning could repair
    state = LatticeState(_model(_divide, "birth_death", family).lgca, kind="birth_death")
    cells = state.cells
    with pytest.raises(error, match="read-only"):
        write(cells)
    cells.divide(np.ones(len(cells), dtype=bool))  # operations replace the arrays, read-only again
    with pytest.raises(error, match="read-only"):
        write(cells)


def test_reorientation_terms_get_a_state_they_cannot_change():
    @reorientation_term(coupling="rest", name="meddling")
    def meddling(state):
        """Moves the cells while it scores them."""
        state.shuffle_cells()
        return np.zeros(state.dims)

    with pytest.raises(TypeError, match="for reading only and cannot move cells"):
        build_model(ModelSpec(space=SpaceSpec(geometry="square", dims=(4, 4)), state=StateSpec(density=0.5),
                              dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[meddling()])])))


def test_stack_builders_get_a_state_they_cannot_change():
    @stack(kind="birth_death", families="classical", name="meddling_stack")
    def meddling_stack(state):
        """Removes cells while the model is built."""
        state.remove_cells(0.5)
        return [{"name": "random_walk"}]

    with pytest.raises(TypeError, match="for reading only and cannot remove cells"):
        build_model(ModelSpec(state=StateSpec(density=0.5),
                              dynamics=InteractionPipelineSpec(operators=[meddling_stack()])))


def test_reactions_of_a_field_get_a_state_they_cannot_change():
    from lgca.fields import PDESpec, reaction

    @reaction(name="meddling_reaction")
    def meddling_reaction(state, c):
        state.remove_cells(0.5)
        return 0.0, 0.1

    model = build_model(ModelSpec(state=StateSpec(density=0.5, fields={"u": 1.0}), dynamics=InteractionPipelineSpec(
        operators=[PDESpec(field="u", diffusion=1.0, reactions=[{"name": "meddling_reaction"}])])))
    with pytest.raises(TypeError, match="for reading only and cannot remove cells"):
        model.step()


def test_a_term_that_draws_random_numbers_leaves_the_stream_alone_at_build():
    @reorientation_term(coupling="rest", name="restless")
    def restless(state):
        """Rest with a random preference per node."""
        return state.rng.random(state.dims)

    def built(term):
        return build_model(ModelSpec(
            space=SpaceSpec(geometry="square", dims=(6, 6)), state=StateSpec(density=0.5, restchannels=1),
            time=TimeSpec(seed=3), dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])])))

    with_draws, without = built(restless(beta=1.0)), built(ReorientationTermSpec("resting_bias", beta=1.0))

    assert with_draws.lgca.rng.bit_generator.state == without.lgca.rng.bit_generator.state


def test_a_stack_narrower_than_its_operators_warns_when_the_model_is_built():
    @stack(kind="reorientation", families="classical", name="growing_walk")
    def growing_walk(state):
        """Cells divide and walk."""
        return [{"name": "birth_death", "parameters": {"birth_rate": 0.2}}, {"name": "random_walk"}]

    spec = ModelSpec(state=StateSpec(density=0.5, restchannels=1),
                     dynamics=InteractionPipelineSpec(operators=[growing_walk()]))
    message = r"growing_walk is a stack of kind 'reorientation', but operators\[0\] \(birth_death\)"
    with pytest.warns(ContractWarning, match=message):
        model = build_model(spec)
    model.step()  # each operator runs by its own kind
    with strict_contracts(), pytest.raises(ContractWarning, match=message):
        build_model(spec)


def test_an_interrupted_reorientation_leaves_the_labels_in_place(monkeypatch):
    # the sampler wrote the occupied channels (ones) and then the labels: an interrupt between the two
    # writes replaced every label with 1
    import lgca.lattice_state

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(6, 6)),
        state=StateSpec(density=0.8, restchannels=1, identity_based=True),
        time=TimeSpec(seed=5),
        dynamics=InteractionPipelineSpec(operators=[
            ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=1.0)])], propagation=False)))
    interior = model.lgca.nonborder
    before = model.lgca.nodes[interior].copy()

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(lgca.lattice_state, "place_labels", interrupted)
    with pytest.raises(KeyboardInterrupt):
        model.step()

    np.testing.assert_array_equal(model.lgca.nodes[interior], before)


def test_a_model_without_rollback_whose_step_failed_refuses_further_steps():
    steps = []

    @interaction(kind="birth_death", families="classical", name="fails_at_step_2")
    def fails_at_step_2(state):
        steps.append(state.step)
        state.remove_cells(0.1)
        if state.step == 2:
            raise ValueError("broken rule")

    model = build_model(ModelSpec(state=StateSpec(density=0.5), time=TimeSpec(steps=5, seed=1),
                                  dynamics=InteractionPipelineSpec(operators=[fails_at_step_2()])))
    model.rollback = False  # with it, the failed step is undone and the model goes on (rollback_test.py)
    model.step()
    with pytest.raises(ValueError, match="broken rule"):
        model.step()
    for go_on in (model.step, lambda: model.run(showprogress=False), model.lgca.timestep):
        with pytest.raises(RuntimeError, match=r"step 2 of this model failed \(ValueError: broken rule\).*rebuild"):
            go_on()

    assert steps == [1, 2]
    build_model(model.spec).step()  # a rebuilt model runs
