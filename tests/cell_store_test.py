"""Identity-based models without volume exclusion keep their cells in a table between rules.

Boundary conditions and propagation move the table with lookup tables; they must move every
cell exactly as the list code does, and lgca.nodes must show the same state.
"""

import numpy as np
import pytest

from lgca import nove_ib_base
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)

GEOMETRIES = [("lin", (10,)), ("square", (6, 6)), ("hex", (6, 6)), ("cubic", (4, 4, 4)), ("moore", (3, 3, 3))]


def _model(geometry, dims, bc, operators=(), seed=1, density=3):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims, boundary=bc),
        state=StateSpec(density=density, restchannels=1, volume_exclusion=False, identity_based=True,
                        capacity=8, traits={"kappa": 4.0}),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=list(operators))))


def _labels(nodes):
    return [sorted(channel) for channel in nodes.reshape(-1)]


@pytest.mark.parametrize("bc", ["periodic", "reflecting", "absorbing"])
@pytest.mark.parametrize("geometry,dims", GEOMETRIES)
def test_the_table_moves_cells_like_the_lists(geometry, dims, bc):
    lists = _model(geometry, dims, bc).lgca
    table = _model(geometry, dims, bc).lgca
    assert table._cell_table() is not None
    for _ in range(4):
        for lgca in (lists, table):
            lgca.apply_boundaries()
            lgca.update_dynamic_fields()
        np.testing.assert_array_equal(table.channel_pop, lists.channel_pop)
        for lgca in (lists, table):
            lgca.propagation()
            lgca.update_dynamic_fields()
        assert table.__dict__["_store"] is not None  # the table is still the state
        inner = lists.nonborder
        np.testing.assert_array_equal(table.channel_pop[inner], lists.channel_pop[inner])
        if bc != "periodic":  # ghost nodes hold cells in flight beyond the wall
            np.testing.assert_array_equal(table.channel_pop, lists.channel_pop)
    assert _labels(table.nodes[inner]) == _labels(lists.nodes[inner])


def test_a_pipeline_of_rules_never_builds_lists(monkeypatch):
    built = []
    original = nove_ib_base.NoVE_IBLGCA_base._nodes_from_store

    def counted(self, *args):
        built.append(1)
        return original(self, *args)

    monkeypatch.setattr(nove_ib_base.NoVE_IBLGCA_base, "_nodes_from_store", counted)
    model = _model("hex", (10, 10), "reflecting", [
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01, "mutation": {"kappa": 0.1}}},
        ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=1.0)]),
    ])
    for _ in range(5):
        model.step()
    assert built == []
    nodes = model.lgca.nodes  # reading the state builds the lists once
    assert built == [1]
    assert sum(len(channel) for channel in nodes[model.lgca.nonborder].flat) == model.lgca.cell_density[
        model.lgca.nonborder].sum()


def test_rules_and_legacy_code_can_take_turns():
    model = _model("square", (10, 10), "periodic", [
        {"name": "nove_ib.random_walk"},  # legacy: reads and writes lists
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5}},
        {"name": "random_walk"},
    ])
    lgca = model.lgca
    before = sorted(label for channel in lgca.nodes[lgca.nonborder].flat for label in channel)
    for _ in range(5):
        model.step()
    after = sorted(label for channel in lgca.nodes[lgca.nonborder].flat for label in channel)
    assert after == before


def test_changes_to_the_lists_reach_the_table():
    lgca = _model("lin", (6,), "periodic").lgca
    lgca._cell_table()
    node = (lgca.r_int, 0)
    lgca.nodes[node] = lgca.nodes[node] + [10_000]  # e.g. a notebook edits a node
    labels, _ = lgca._cell_table()
    assert 10_000 in labels


@pytest.mark.parametrize("bc", ["periodic", "reflecting"])
def test_the_node_recorder_stores_cell_tables(bc, monkeypatch):
    from lgca.model import AnalysisSpec, run_model
    from lgca.simulation import NodeRecorder

    built = []
    original = nove_ib_base.NoVE_IBLGCA_base._nodes_from_store

    def counted(self, *args):
        built.append(1)
        return original(self, *args)

    monkeypatch.setattr(nove_ib_base.NoVE_IBLGCA_base, "_nodes_from_store", counted)
    spec = ModelSpec(
        space=SpaceSpec(geometry="hex", dims=(8, 8), boundary=bc),
        state=StateSpec(density=3, restchannels=1, volume_exclusion=False, identity_based=True, capacity=8,
                        traits={"kappa": 4.0}),
        time=TimeSpec(steps=6, seed=2),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5}},
            {"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=[NodeRecorder()]))
    lgca = run_model(spec, showprogress=False).lgca
    assert len(lgca.cells_t) == 7 and built in ([], [1])  # at most once, to estimate the recording size
    snapshot = lgca.cells_t[-1]
    counts = np.zeros(lgca.dims + (lgca.K,), dtype=int)
    np.add.at(counts, snapshot.node + (snapshot.channel,), 1)
    np.testing.assert_array_equal(counts, lgca.channel_pop[lgca.nonborder])
    nodes_t = lgca.nodes_t  # built from the tables on first read
    assert nodes_t.shape == (7,) + lgca.dims + (lgca.K,)
    for time, recorded in enumerate(lgca.cells_t):
        assert sorted(label for channel in nodes_t[time].flat for label in channel) == sorted(recorded.label)
