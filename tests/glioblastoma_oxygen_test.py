"""The glioblastoma model with oxygen: lattice units, vessels as sources, clogging, resection and evolution."""

import numpy as np
import pytest

from lgca.examples import glioblastoma_oxygen as go
from lgca.lattice_state import LatticeState
from lgca.model import build_model, run_model


def test_the_physical_parameters_become_lattice_units():
    lattice = go.Parameters().lattice()
    assert lattice["diffusion"] == pytest.approx(2000 / 50**2)  # nodes² per s
    # a full node of 100 cells consumes 100 x 1e-11 µmol/s from (50 µm)³ of tissue at 60 mmHg
    assert 100 * lattice["uptake"] * (50e-4) ** 3 * 60 * go.MMHG == pytest.approx(100 * 1e-11)
    assert lattice["michaelis_constant"] == pytest.approx(2.5 / 60)
    assert go.Parameters().vessel_probability(1) == 0.25 and go.Parameters().vessel_probability(2) == 1 / 16


def test_open_vessels_hold_oxygen_at_the_bloods_and_cells_take_it_up():
    model = build_model(go.build_spec("lin", size=200, steps=1, seed=3))
    state = LatticeState(model.lgca)
    oxygen, vessels = state.field("oxygen"), state.field("vessels")
    assert oxygen[vessels > 0] == pytest.approx(1, abs=1e-4)
    assert oxygen.max() <= 1 + 1e-9 and oxygen.min() > 0
    # the cells at the first node lower the oxygen around them
    gap = np.flatnonzero(vessels == 0)
    assert oxygen[0] < np.median(oxygen[gap])


def test_vessels_stay_open_without_clogging_and_close_only_under_cells():
    for clogging in (0.0, 1.0):
        result = run_model(go.build_spec("lin", size=200, steps=40, clogging=clogging, seed=4),
                           showprogress=False)
        vessels = np.asarray(result.data["vessels"])
        cells = result.lgca.cell_density[result.lgca.nonborder]
        if clogging == 0:
            np.testing.assert_array_equal(vessels[-1], vessels[0])
        else:
            closed = (vessels[0] > 0) & (vessels[-1] == 0)
            assert closed.any()
            reached = np.flatnonzero(cells > 0).max()
            assert not closed[reached + 1:].any()  # no cell has been beyond the front
            assert np.all(np.diff(vessels.sum(-1)) <= 0)  # closed vessels stay closed


def test_resection_removes_the_dense_tumour_and_its_vessels():
    spec = go.build_spec("lin", size=200, steps=60, resect_at=50, seed=5)
    model = build_model(spec)
    for _ in range(49):  # the resection happens in step 50, which leads to time 50
        model.step()
    before = model.lgca.cell_density[model.lgca.nonborder].copy()
    vessels_before = LatticeState(model.lgca).field("vessels").copy()
    model.step()
    after = model.lgca.cell_density[model.lgca.nonborder]
    dense = before >= 0.1 * spec.state.capacity
    assert dense.sum() > 5
    assert after[dense].sum() < 0.01 * before[dense].sum() + 10
    vessels = LatticeState(model.lgca).field("vessels")
    assert not vessels[dense].any() and (vessels[~dense] == vessels_before[~dense]).all()


def test_the_switch_reads_the_oxygen_with_each_cells_kappa_and_theta():
    # all cells at the first node, with κ = 2 and θ = 0.3: they rest with (1 + tanh(2 (O - 0.3))) / 2
    spec = go.build_spec("lin", size=100, steps=1, cells=2000, kappa_0=(2, 2), theta_0=(0.3, 0.3),
                         division=(0, 0, 0), death=(0, 0, 0), parameters=go.Parameters(capacity=4000), seed=6)
    model = build_model(spec)
    oxygen = LatticeState(model.lgca).field("oxygen")[0]
    expected = (1 + np.tanh(2 * (oxygen - 0.3))) / 2
    model.step()
    resting = LatticeState(model.lgca).cells.in_channels("rest").mean()
    assert resting == pytest.approx(expected, abs=4 * np.sqrt(expected * (1 - expected) / 2000))


def test_the_tumour_invades_and_its_traits_evolve_within_bounds():
    for geometry, size, steps in (("lin", 300, 150), ("hex", 40, 30)):
        result = run_model(go.build_spec(geometry, size=size, steps=steps, clogging=0.02, seed=7),
                           showprogress=False)
        cells = LatticeState(result.lgca).cells
        assert len(cells) > 100
        theta = np.asarray(cells["theta"], dtype=float)
        assert theta.min() >= 0 and theta.max() <= 1
        assert np.unique(np.asarray(cells["kappa"])).size > 100  # daughters mutate


def test_kymographs_record_cells_oxygen_and_traits():
    record = go.kymographs(go.build_spec("lin", size=80, steps=20, seed=8), every=10)
    np.testing.assert_array_equal(record["steps"], [0, 10, 20])
    for key in ("cells", "migrating", "oxygen", "vessels", "kappa", "theta"):
        assert record[key].shape == (3, 80)
    assert np.all(record["migrating"] <= record["cells"])
    assert np.isnan(record["kappa"][0, 1:]).all() and np.isfinite(record["kappa"][0, 0])


def test_cells_divide_and_die_with_the_probabilities_of_their_oxygen_zone():
    spec = go.build_spec("lin", size=100, steps=1, seed=9)
    model = build_model(spec)
    lgca = model.lgca
    oxygen = np.array([1.0, 0.05, 0.001])  # normoxic, hypoxic, anoxic
    field = LatticeState(lgca).field("oxygen")
    state = LatticeState(lgca, kind="phenotype_switch")
    nodes = np.asarray(state.cells.node)
    values = np.zeros_like(field)
    values[0] = oxygen[0]
    lgca.oxygen[lgca.nonborder] = values
    go.oxygen_zones(state, hypoxia=0.1, anoxia=0.01, division=(1, 0.5, 0), death=(0.1, 0.2, 0.5))
    state.commit()
    cells = LatticeState(lgca).cells
    assert (np.asarray(cells["r_b"]) == 1).all() and (np.asarray(cells["r_d"]) == 0.1).all()
    for value, (r_b, r_d) in zip(oxygen[1:], [(0.5, 0.2), (0, 0.5)]):
        lgca.oxygen[lgca.nonborder] = np.full_like(field, value)
        state = LatticeState(lgca, kind="phenotype_switch")
        go.oxygen_zones(state, hypoxia=0.1, anoxia=0.01, division=(1, 0.5, 0), death=(0.1, 0.2, 0.5))
        state.commit()
        cells = LatticeState(lgca).cells
        assert (np.asarray(cells["r_b"]) == r_b).all() and (np.asarray(cells["r_d"]) == r_d).all()
    assert nodes.max() == 0


def test_stable_vessels_never_clog():
    vessels = go.build_vessels((200,), seed=1, stable=5)
    assert (vessels == 2).sum() == 5 and set(np.unique(vessels)) == {0.0, 1.0, 2.0}
    with pytest.raises(ValueError, match="more than"):
        go.build_vessels((8,), seed=1, stable=9)
    result = run_model(go.build_spec("lin", size=60, steps=60, clogging=1.0, stable=3, seed=4), showprogress=False)
    vessels = np.asarray(result.data["vessels"])
    stable = vessels[0] == 2
    assert (vessels[:, stable] == 2).all()
    assert ((vessels[0] == 1) & (vessels[-1] == 0)).any()  # the others clog
    oxygen = LatticeState(result.lgca).field("oxygen")
    assert oxygen[stable] == pytest.approx(1, abs=1e-4)
