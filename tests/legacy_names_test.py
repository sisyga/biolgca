"""The interaction names of earlier versions run stacks of rules that match the legacy functions.

``get_lgca(interaction=name)`` translates every legacy name of every family
(``lgca.legacy_names``). One interaction step of the translation and of the
legacy function (kept in ``tests/legacy``) must agree in distribution: the
mean number of cells per node, of resting cells and the mean flux, over many
nodes. The research models, cues and growth rules have closer tests of their
own (``research_models_test``, ``single_cue_test``, ``birth_death_test``).
"""

import warnings

import numpy as np
import pytest

from lgca import get_lgca
from lgca.legacy_names import legacy_names
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.plugins import describe_plugin, list_plugins
from tests.legacy import legacy_lgca

DIMS = (100, 100)
# family: get_lgca keywords and how the initial nodes are drawn
FAMILIES = {
    "classical": {"ve": True, "ib": False, "n_species": 1, "restchannels": 2},
    "nove": {"ve": False, "ib": False, "n_species": 1, "restchannels": 1, "capacity": 8},
    "ib": {"ve": True, "ib": True, "n_species": 1, "restchannels": 2},
    "nove_ib": {"ve": False, "ib": True, "n_species": 1, "restchannels": 1, "capacity": 8},
    "multispecies_ve": {"ve": True, "ib": False, "n_species": 2, "restchannels": 4},
    "multispecies": {"ve": False, "ib": False, "n_species": 2, "restchannels": 1, "capacity": 8},
}
# parameters that keep a documented difference out of the comparison
PARAMETERS = {
    "go_or_grow_kappa_chemo": {"kappa": 0.0},  # the legacy neighbourhood mean divided by one node too few
}
# get_lgca names that the legacy classes did not have, so there is nothing to compare with
NEW = {"classical": {"persistent_walk"}, "nove_ib": {"evo_steric", "go_or_grow_kappa_chemo"}}
LEGACY_MULTISPECIES = {"multispecies_ve": ("random_walk", "only_propagation", "excitable_medium_ms"),
                       "multispecies": ("birth", "birthdeath", "go_or_grow", "only_propagation")}


# legacy restrictions of the model
MODEL = {"dd_alignment": {"restchannels": 0}, "di_alignment": {"restchannels": 0}}


def _nodes(keywords, rng):
    K = 4 + keywords["restchannels"]
    shape = DIMS + ((keywords["n_species"],) if keywords["n_species"] > 1 else ()) + (K,)
    if keywords["ve"]:
        occupied = rng.random(shape) < 0.4
        if not keywords["ib"]:
            return occupied
        return np.where(occupied, np.cumsum(occupied).reshape(shape), 0).astype(np.uint64)
    counts = rng.poisson(0.5, shape)
    if not keywords["ib"]:
        return counts
    nodes = np.empty(shape, dtype=object)
    first = 1
    for index in np.ndindex(shape):
        nodes[index] = list(range(first, first + counts[index]))
        first += counts[index]
    return nodes


def _cases():
    for family in FAMILIES:
        names = LEGACY_MULTISPECIES.get(family) or set(legacy_names(family, ndim=2)) - NEW.get(family, set())
        for name in sorted(names):
            yield pytest.param(family, name, id=f"{family}-{name}")


def _statistics(lgca):
    """Cells, resting cells and flux per node."""
    counts = lgca._channel_counts(lgca.nodes[lgca.nonborder]).astype(float)
    n = counts.sum(-1)
    rest = counts[..., lgca.velocitychannels:].sum(-1)
    flux = counts[..., :lgca.velocitychannels] @ lgca.c.T
    return [n.ravel(), rest.ravel(), flux[..., 0].ravel(), flux[..., 1].ravel()]


@pytest.mark.parametrize("family, name", list(_cases()))
def test_every_legacy_name_matches_the_legacy_function(family, name):
    keywords = {**FAMILIES[family], **MODEL.get(name, {})}
    nodes = _nodes(keywords, np.random.default_rng(3))
    parameters = PARAMETERS.get(name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. "system will die out" for go_or_grow with volume exclusion
        new = get_lgca(geometry="square", nodes=nodes, interaction=name, seed=1, **keywords, **parameters)
        legacy = legacy_lgca(geometry="square", nodes=nodes, interaction=name, seed=2, **keywords, **parameters)
    before = _statistics(new)
    new.interaction(new)
    legacy.interaction(legacy)
    legacy.update_dynamic_fields()
    # both start from the same nodes: compare the changes, whose spread is small
    for statistic, start, a, b in zip(("cells", "resting", "flux x", "flux y"), before,
                                      _statistics(new), _statistics(legacy)):
        a, b = a - start, b - start
        error = np.sqrt((a.var() + b.var()) / len(a))
        assert abs(a.mean() - b.mean()) <= 5 * error + 1e-12, (statistic, a.mean(), b.mean())


@pytest.mark.parametrize("family, name", [("classical", "alignment"), ("ib", "birthdeath"),
                                          ("nove_ib", "go_or_grow"), ("multispecies", "go_or_grow")])
def test_a_prefixed_name_in_a_model_file_warns_and_runs_what_get_lgca_runs(family, name):
    keywords = FAMILIES[family]
    nodes = _nodes(keywords, np.random.default_rng(4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # get_lgca uses the legacy names without a warning
        lgca = get_lgca(geometry="square", nodes=nodes, interaction=name, seed=5, **keywords)
    lgca.timeevo(3, record=True, showprogress=False)
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=keywords["restchannels"], volume_exclusion=keywords["ve"],
                        identity_based=keywords["ib"], n_species=keywords["n_species"],
                        capacity=keywords.get("capacity") if not keywords["ib"] else 8 if not keywords["ve"] else None),
        time=TimeSpec(steps=3, seed=5),
        dynamics=InteractionPipelineSpec(operators=[{"name": f"{family}.{name}"}]))
    with pytest.warns(FutureWarning, match=f"'{family}.{name}' is deprecated"):
        model = build_model(spec)
    model.run(False)
    np.testing.assert_array_equal(lgca._channel_counts(lgca.nodes[lgca.nonborder]),
                                  model.lgca._channel_counts(model.lgca.nodes[model.lgca.nonborder]))


def test_deprecated_names_are_described_but_not_listed():
    listed = {info.name for info in list_plugins()}
    assert "classical.alignment" not in listed and "polar_alignment" in listed
    assert "classical.alignment" in {info.name for info in list_plugins(deprecated=True)}
    assert "polar_alignment" in describe_plugin("classical.alignment").deprecated


def test_unknown_names_list_the_names_of_the_family():
    with pytest.raises(ValueError, match="Unknown interaction 'alignmnet'.*alignment"):
        get_lgca(geometry="square", dims=(4, 4), interaction="alignmnet")


def test_get_lgca_keeps_its_interface():
    lgca = get_lgca(geometry="hex", dims=(6, 6), interaction="birthdeath", r_b=0.3, seed=1)
    assert lgca.interaction.__name__ == "birthdeath"
    assert lgca.interaction_params == {"r_b": 0.3, "r_d": 0.05}
    assert "alignment" in lgca.interactions
    before = lgca.nodes.copy()
    lgca.interaction(lgca)  # one interaction step, without propagation
    assert not np.array_equal(before, lgca.nodes)
    lgca.timeevo(2, showprogress=False)
