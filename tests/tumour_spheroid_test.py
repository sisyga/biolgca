"""The tumour spheroid example: its parameters in lattice units, its layers, and its helpers."""

import math

import numpy as np
import pytest

from lgca.examples import tumour_spheroid as ts
from lgca.model import build_model, run_model


def test_the_physical_parameters_become_lattice_units():
    parameters = ts.Parameters()
    assert parameters.capacity("moore") == 13 and parameters.capacity("hex") == 11  # 0.65 x 27 pL / 1.4 pL
    lattice = ts.lattice_parameters(parameters, "moore")
    assert lattice["diffusion"] == pytest.approx(2000 * 3600 / 30**2)  # nodes² per hour
    # 13 cells at the capacity consume 22 mmHg/s; relative to the medium's 140 mmHg, per hour and cell
    assert 13 * lattice["uptake"] * 140 / 3600 == pytest.approx(22)
    assert lattice["quiescence_oxygen"] == pytest.approx(5 / 140)
    # a cell divides with probability 1 - 2^(-1/20) per hour: the number doubles in 20 hours
    assert (1 - lattice["division"]) ** -20 == pytest.approx(2)
    assert ts.probability(10.0) == pytest.approx(1 - math.exp(-0.1))


@pytest.mark.parametrize("seed", [2002, 1])
def test_the_spheroid_grows_into_a_necrotic_core_a_quiescent_ring_and_a_proliferating_rim(seed):
    result = run_model(ts.build_spec("hex", dims=(50, 50), steps=120, seed=seed), showprogress=False)
    density = np.asarray(result.data["density"])  # every 24 hours
    radii = ts.layers(density, "hex")
    assert np.all(np.diff(radii["spheroid"]) > 30)  # µm per day
    assert radii["necrotic"][0] == radii["quiescent"][0] == 0  # it starts with proliferating cells
    assert 0 < radii["necrotic"][-1] < radii["quiescent"][-1] < radii["spheroid"][-1]
    # outside in: proliferating cells (uptake 1), quiescent cells (0.4) and necrotic material (0)
    _, uptake = ts.radial_profile(ts.mean_uptake(result.lgca), "hex", density[-1])
    assert uptake[:3].max() < 0.3 and 0.3 < uptake[6] < 0.7 and uptake[11] > 0.95
    # oxygen runs out at the centre and is the medium's beyond the unstirred layer
    _, oxygen = ts.radial_profile(result.lgca.oxygen[result.lgca.nonborder], "hex", density[-1])
    assert oxygen[:4].max() * 140 < 0.5 and oxygen[-1] == pytest.approx(1)


def test_the_spheroid_runs_on_the_moore_lattice():
    model = build_model(ts.build_spec("moore", dims=(14, 14, 14), steps=2))
    oxygen = model.lgca.oxygen[model.lgca.nonborder]
    assert oxygen[7, 7, 7] < oxygen[0, 0, 0] == pytest.approx(1)  # solved when the model is built
    for _ in range(2):
        model.step()
    medium = model.lgca.medium[model.lgca.nonborder]
    cells = model.lgca.cell_density[model.lgca.nonborder]
    assert medium[cells > 0].max() == 0 and medium[0, 0, 0] == 1


def test_the_stirred_medium_starts_beyond_the_unstirred_layer():
    occupied = np.zeros((9, 9), dtype=bool)
    occupied[4, 4] = True
    stirred = ts._stirred("square", (9, 9), occupied, 2.0)  # nodes on a square grid
    distance = np.hypot(*(np.indices((9, 9)) - 4))
    np.testing.assert_array_equal(stirred, (distance > 2.0).astype(float))
    assert ts._stirred("hex", (6, 6), np.zeros((6, 6), dtype=bool), 2.0).all()  # no cells: all medium


def test_the_layers_are_balls_of_tissue_at_the_capacity():
    parameters = ts.Parameters()
    density = np.zeros((10, 10, 10, 3))
    density[0, 0, 0] = [1300, 650, 650]  # 2600 cells: 200 nodes of tissue
    radii = ts.layers(density, "moore")
    node = parameters.node_size
    assert radii["spheroid"] == pytest.approx((3 * 200 * node**3 / (4 * math.pi)) ** (1 / 3))
    assert radii["necrotic"] == pytest.approx((3 * 50 * node**3 / (4 * math.pi)) ** (1 / 3))
    assert radii["quiescent"] == pytest.approx((3 * 100 * node**3 / (4 * math.pi)) ** (1 / 3))


def test_the_mean_uptake_is_a_property_of_the_classes():
    density = np.array([[[4, 0, 0], [0, 4, 0], [2, 0, 2], [0, 0, 0]]])
    np.testing.assert_allclose(ts.mean_uptake(None, density), [[1.0, 0.4, 0.5, np.nan]])


def test_an_unknown_geometry_is_refused():
    with pytest.raises(ValueError, match="geometry must be one of hex, moore"):
        ts.build_spec("square")
