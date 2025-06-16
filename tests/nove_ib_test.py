import numpy as np
import copy
import warnings

import pytest

from lgca import get_lgca
from lgca.lgca_3dmoore import LGCA_3dMoore
from tests.common_test import T_LGCA_Common
from tests.classical_test import Test_LGCA_classical as T_LGCA_classical
import tests.classical_test
from tests.ib_test import matching_import
matching_import("^nodes_cb_", tests.classical_test, globals())
matching_import("^out_cb_", tests.classical_test, globals())


def counts_to_lists(arr):
    """Convert integer occupation numbers to identity-based lists."""
    arr = np.asarray(arr, dtype=int)
    labels = list(range(int(arr.sum())))
    out = np.empty(arr.shape, dtype=object)
    counter = 0
    for idx in np.ndindex(arr.shape):
        n = int(arr[idx])
        out[idx] = labels[counter:counter + n]
        counter += n
    return out


def manual_propagate(nodes, geom, bc="pbc"):
    """Manually propagate list-based nodes for one timestep."""
    if geom == "lin":
        L, K = nodes.shape
        rest = K - 2
        opp = {0: 1, 1: 0}
        out = np.empty_like(nodes, dtype=object)
        for idx in np.ndindex(nodes.shape):
            out[idx] = []
        for x in range(L):
            # right-moving
            for pid in nodes[x, 0]:
                dest = x + 1
                ch = 0
                if dest >= L:
                    if bc == "pbc":
                        dest = 0
                    elif bc == "rbc":
                        dest = x
                        ch = opp[0]
                    else:
                        continue
                out[dest, ch].append(pid)
            # left-moving
            for pid in nodes[x, 1]:
                dest = x - 1
                ch = 1
                if dest < 0:
                    if bc == "pbc":
                        dest = L - 1
                    elif bc == "rbc":
                        dest = x
                        ch = opp[1]
                    else:
                        continue
                out[dest, ch].append(pid)
            # rest
            out[x, rest].extend(nodes[x, rest])
        return out

    elif geom == "square":
        Lx, Ly, K = nodes.shape
        rest = K - 4
        opp = {0: 2, 1: 3, 2: 0, 3: 1}
        out = np.empty_like(nodes, dtype=object)
        for idx in np.ndindex(nodes.shape):
            out[idx] = []
        for x in range(Lx):
            for y in range(Ly):
                cell = nodes[x, y]
                # right
                for pid in cell[0]:
                    nx = x + 1
                    ny = y
                    ch = 0
                    if nx >= Lx:
                        if bc == "pbc":
                            nx = 0
                        elif bc == "rbc":
                            nx = x
                            ch = opp[0]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # up
                for pid in cell[1]:
                    nx = x
                    ny = y + 1
                    ch = 1
                    if ny >= Ly:
                        if bc == "pbc":
                            ny = 0
                        elif bc == "rbc":
                            ny = y
                            ch = opp[1]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # left
                for pid in cell[2]:
                    nx = x - 1
                    ny = y
                    ch = 2
                    if nx < 0:
                        if bc == "pbc":
                            nx = Lx - 1
                        elif bc == "rbc":
                            nx = x
                            ch = opp[2]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # down
                for pid in cell[3]:
                    nx = x
                    ny = y - 1
                    ch = 3
                    if ny < 0:
                        if bc == "pbc":
                            ny = Ly - 1
                        elif bc == "rbc":
                            ny = y
                            ch = opp[3]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # rest
                out[x, y, 4:].extend(cell[4:])
        return out

    elif geom == "hex":
        Lx, Ly, K = nodes.shape
        rest = K - 6
        opp = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
        out = np.empty_like(nodes, dtype=object)
        for idx in np.ndindex(nodes.shape):
            out[idx] = []
        for x in range(Lx):
            for y in range(Ly):
                cell = nodes[x, y]
                parity = y % 2
                # channel 0 (east)
                for pid in cell[0]:
                    nx, ny, ch = x + 1, y, 0
                    if nx >= Lx:
                        if bc == "pbc":
                            nx = 0
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[0]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # channel 1 (ne)
                for pid in cell[1]:
                    nx = x + parity
                    ny = y + 1
                    ch = 1
                    if nx >= Lx or ny >= Ly:
                        if bc == "pbc":
                            nx %= Lx
                            ny %= Ly
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[1]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # channel 2 (nw)
                for pid in cell[2]:
                    nx = x - 1 + parity
                    ny = y + 1
                    ch = 2
                    if nx < 0 or ny >= Ly:
                        if bc == "pbc":
                            nx %= Lx
                            ny %= Ly
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[2]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # channel 3 (west)
                for pid in cell[3]:
                    nx, ny, ch = x - 1, y, 3
                    if nx < 0:
                        if bc == "pbc":
                            nx = Lx - 1
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[3]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # channel 4 (sw)
                for pid in cell[4]:
                    nx = x - 1 + parity
                    ny = y - 1
                    ch = 4
                    if nx < 0 or ny < 0:
                        if bc == "pbc":
                            nx %= Lx
                            ny %= Ly
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[4]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # channel 5 (se)
                for pid in cell[5]:
                    nx = x + parity
                    ny = y - 1
                    ch = 5
                    if nx >= Lx or ny < 0:
                        if bc == "pbc":
                            nx %= Lx
                            ny %= Ly
                        elif bc == "rbc":
                            nx, ny, ch = x, y, opp[5]
                        else:
                            continue
                    out[nx, ny, ch].append(pid)
                # rest
                out[x, y, 6:].extend(cell[6:])
        return out

    else:
        raise ValueError("unknown geometry")


class Test_LGCA_NoVE_IB(T_LGCA_Common):
    """Tests for identity-based LGCA without volume exclusion."""

    com = T_LGCA_Common
    ve = False
    ib = True

    # reuse absorbing boundary condition tests from classical LGCA
    test_abc_1d = T_LGCA_classical.test_abc_1d
    test_abc_square = T_LGCA_classical.test_abc_square
    test_abc_hex = T_LGCA_classical.test_abc_hex
    test_pbc_cubic = T_LGCA_classical.test_pbc_cubic
    test_rbc_cubic = T_LGCA_classical.test_rbc_cubic
    test_abc_cubic = T_LGCA_classical.test_abc_cubic
    test_pbc_moore = T_LGCA_classical.test_pbc_moore
    test_rbc_moore = T_LGCA_classical.test_rbc_moore
    test_abc_moore = T_LGCA_classical.test_abc_moore

    def t_prop_counts_template(self, geom, nodes, expected, bc="pbc"):
        nodes_ib = counts_to_lists(nodes)
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes_ib, bc=bc, interaction="only_propagation")
        lgca.timeevo(timesteps=1, recorddens=False, showprogress=False)
        result_counts = np.vectorize(len)(lgca.nodes)
        expected_lists = manual_propagate(nodes_ib, geom, bc)
        expected_counts = np.vectorize(len)(expected_lists)
        assert result_counts[lgca.nonborder].sum() == expected_counts.sum(), "Particles appear or disappear"
        assert np.array_equal(result_counts[lgca.nonborder], expected_counts), "Node configuration after propagation not correct"
        assert np.array_equal(lgca.nodes[lgca.nonborder], expected_lists), "Particle IDs not propagated correctly"
        assert np.array_equal(lgca.occupied[lgca.nonborder].sum(-1), lgca.cell_density[lgca.nonborder]), \
            "Cell density field not updated correctly"
        assert np.array_equal(lgca.occupied, result_counts.astype(bool)), \
            "Occupation field not updated correctly"
        return lgca

    @pytest.mark.parametrize("geom,dims", [
        ("lin", (com.xdim_1d,)),
        ("square", (com.xdim_square, com.ydim_square)),
        ("hex", (com.xdim_hex, com.ydim_hex)),
        ("cubic", (com.xdim_cubic, com.ydim_cubic, com.zdim_cubic)),
        ("moore", (com.xdim_moore, com.ydim_moore, com.zdim_moore))
    ])
    def test_recording(self, geom, dims):
        lgca_1 = get_lgca(geometry=geom, ve=False, ib=True, dims=dims, density=0.5, interaction="only_propagation")
        lgca_2 = copy.deepcopy(lgca_1)

        lgca_1.timeevo(timesteps=2, recorddens=False, showprogress=False)
        assert not hasattr(lgca_1, "dens_t")
        assert not hasattr(lgca_1, "nodes_t")
        assert not hasattr(lgca_1, "n_t")
        assert not hasattr(lgca_1, "restcells_t") and not hasattr(lgca_1, "velcells_t")
        assert not hasattr(lgca_1, "ent_t") and not hasattr(lgca_1, "normEnt_t") and not hasattr(lgca_1, "polAlParam_t") \
               and not hasattr(lgca_1, "meanAlign_t")
        del lgca_1

        lgca_3 = copy.deepcopy(lgca_2)
        lgca_2.timeevo(timesteps=2, showprogress=False)
        assert hasattr(lgca_2, "dens_t")
        del lgca_2

        lgca_4 = copy.deepcopy(lgca_3)
        lgca_3.timeevo(timesteps=2, record=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_3, "nodes_t")
        del lgca_3

        lgca_5 = copy.deepcopy(lgca_4)
        lgca_4.timeevo(timesteps=2, recordN=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_4, "n_t")
        del lgca_4


    def test_propagation(self):
        # 1D
        nodes = np.zeros((self.xdim_1d, self.b_1d + 1), dtype=int)
        nodes[0, 0] = 1
        nodes[1, 2] = 2
        nodes[2, 1] = 3
        expected = np.zeros_like(nodes)
        expected[1, 0] = 1
        expected[1, 2] = 2
        expected[1, 1] = 3
        self.t_prop_counts_template("lin", nodes, expected)

        # 2D square
        nodes = np.zeros((self.xdim_square, self.ydim_square, self.b_square + 1), dtype=int)
        nodes[0, 1, 0] = 1
        nodes[1, 0, 1] = 2
        nodes[2, 1, 2] = 3
        nodes[1, 2, 3] = 4
        nodes[1, 1, 4] = 5
        expected = np.zeros_like(nodes)
        expected[1, 1, 0] = 1
        expected[1, 1, 1] = 2
        expected[1, 1, 2] = 3
        expected[1, 1, 3] = 4
        expected[1, 1, 4] = 5
        self.t_prop_counts_template("square", nodes, expected)

        # 2D hex
        nodes = np.zeros((self.xdim_hex, self.ydim_hex, self.b_hex + 1), dtype=int)
        nodes[0, 1, 0] = 1
        nodes[0, 0, 1] = 2
        nodes[1, 0, 2] = 3
        nodes[2, 1, 3] = 4
        nodes[1, 2, 4] = 5
        nodes[0, 2, 5] = 6
        nodes[1, 1, 6] = 7
        expected = np.zeros_like(nodes)
        expected[1, 1, 0] = 1
        expected[1, 1, 1] = 2
        expected[1, 1, 2] = 3
        expected[1, 1, 3] = 4
        expected[1, 1, 4] = 5
        expected[1, 1, 5] = 6
        expected[1, 1, 6] = 7
        self.t_prop_counts_template("hex", nodes, expected)

        # 3D moore
        nodes = np.zeros((self.xdim_moore, self.ydim_moore, self.zdim_moore, self.b_moore + 1), dtype=int)
        nodes[0, 1, 1, 21] = 1
        nodes[2, 1, 1, 4] = 2
        nodes[1, 0, 1, 15] = 3
        nodes[1, 2, 1, 10] = 4
        nodes[1, 1, 0, 13] = 5
        nodes[1, 1, 2, 12] = 6
        nodes[1, 1, 1, 26] = 7
        expected = np.zeros_like(nodes)
        expected[1, 1, 1, [21, 4, 15, 10, 13, 12, 26]] = [1, 2, 3, 4, 5, 6, 7]
        self.t_prop_counts_template("moore", nodes, expected)

    def test_propagation_moore_all_channels(self):
        restchannels = 2
        nodes = np.zeros(
            (self.xdim_moore, self.ydim_moore, self.zdim_moore, restchannels + self.b_moore),
            dtype=int,
        )
        center = (1, 1, 1)
        for idx, (dx, dy, dz) in enumerate(LGCA_3dMoore.velocities):
            nodes[center[0] - dx, center[1] - dy, center[2] - dz, idx] = 1
        nodes[center][self.b_moore:] = 1
        expected = np.zeros_like(nodes)
        expected[center] = 1
        self.t_prop_counts_template("moore", nodes, expected)

    @pytest.mark.parametrize("geom,nodes,b", [
        ("lin", com.nodes_nove_1d, com.b_1d),
        ("square", com.nodes_nove_square, com.b_square),
        ("hex", com.nodes_nove_hex, com.b_hex),
        ("cubic", com.nodes_nove_cubic, com.b_cubic),
        ("moore", com.nodes_nove_moore, com.b_moore)
    ])
    def test_getlgca_capacity(self, geom, nodes, b):
        capacity = 10
        density = 0.5
        nodes_ib = counts_to_lists(nodes)
        lgca = get_lgca(geometry=geom, ve=False, ib=True, nodes=nodes_ib, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, nodes=nodes_ib, capacity=capacity, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, density=density, capacity=capacity, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, density=density, capacity=capacity, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, nodes=nodes_ib, restchannels=1, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, density=density, restchannels=1, interaction="only_propagation")
        assert lgca.capacity == b + 1
        lgca = get_lgca(geometry=geom, ve=False, ib=True, density=density, restchannels=1, interaction="only_propagation")
        assert lgca.capacity == b + 1

    @pytest.mark.parametrize("geom,nodes", [
        ("lin", com.nodes_nove_1d),
        ("square", com.nodes_nove_square),
        ("hex", com.nodes_nove_hex),
        ("cubic", com.nodes_nove_cubic),
        ("moore", com.nodes_nove_moore)
    ])
    def test_characteristics(self, geom, nodes):
        nodes_ib = counts_to_lists(nodes)
        ref_lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib)
        for interaction in ref_lgca.interactions:
            self.t_characteristics(geom, nodes_ib, interaction, "pbc")
            self.t_characteristics(geom, nodes_ib, interaction, "rbc")
            self.t_characteristics(geom, nodes_ib, interaction, "abc")

    def t_characteristics(self, geom, nodes, interaction, bc):
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=copy.deepcopy(nodes), interaction=interaction, bc=bc)
        lgca.timeevo(timesteps=2, recorddens=False, record=True, showprogress=False)
        current = [id_ for node in lgca.nodes[lgca.nonborder].flat for id_ in node]
        if len(current) == 0 and len([id_ for node in lgca.nodes_t[0].flat for id_ in node]) != 0 and bc != "abc":
            warnings.warn("System died out in " + str(interaction))
        for i in range(3):
            labels = [id_ for node in lgca.nodes_t[i].flat for id_ in node]
            assert len(labels) == len(set(labels)), "Uniqueness principle is broken"
            if labels:
                assert max(labels) <= lgca.maxlabel, "IDs not updated correctly"
            if lgca.props:
                for propname in lgca.props.keys():
                    assert len(lgca.props[propname]) == lgca.maxlabel + 1, "Properties not updated for all particles"

    def test_pbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        out_1d_rbound[0, 0] = 2
        self.t_prop_counts_template("lin", nodes_1d_rbound, out_1d_rbound, bc="pbc")
        out_1d_lbound[-1, 1] = 2
        self.t_prop_counts_template("lin", nodes_1d_lbound, out_1d_lbound, bc="pbc")

    def test_rbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        out_1d_rbound[-1, 1] = 2
        self.t_prop_counts_template("lin", nodes_1d_rbound, out_1d_rbound, bc="rbc")
        out_1d_lbound[0, 0] = 2
        self.t_prop_counts_template("lin", nodes_1d_lbound, out_1d_lbound, bc="rbc")

    def test_pbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        out_sq_rbound[0, 3, 0] = 1
        out_sq_rbound[0, 2, 0] = 2
        out_sq_rbound[0, 1, 0] = 3
        out_sq_rbound[0, 0, 0] = 4
        self.t_prop_counts_template("square", nodes_sq_rbound, out_sq_rbound, bc="pbc")

        out_sq_lbound[-1, 3, 2] = 1
        out_sq_lbound[-1, 2, 2] = 2
        out_sq_lbound[-1, 1, 2] = 3
        out_sq_lbound[-1, 0, 2] = 4
        self.t_prop_counts_template("square", nodes_sq_lbound, out_sq_lbound, bc="pbc")

        out_sq_tbound[0, 0, 1] = 1
        out_sq_tbound[1, 0, 1] = 2
        out_sq_tbound[2, 0, 1] = 3
        self.t_prop_counts_template("square", nodes_sq_tbound, out_sq_tbound, bc="pbc")

        out_sq_bbound[0, -1, 3] = 1
        out_sq_bbound[1, -1, 3] = 2
        out_sq_bbound[2, -1, 3] = 3
        self.t_prop_counts_template("square", nodes_sq_bbound, out_sq_bbound, bc="pbc")

    def test_rbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        out_sq_rbound[-1, 3, 2] = 1
        out_sq_rbound[-1, 2, 2] = 2
        out_sq_rbound[-1, 1, 2] = 3
        out_sq_rbound[-1, 0, 2] = 4
        self.t_prop_counts_template("square", nodes_sq_rbound, out_sq_rbound, bc="rbc")

        out_sq_lbound[0, 3, 0] = 1
        out_sq_lbound[0, 2, 0] = 2
        out_sq_lbound[0, 1, 0] = 3
        out_sq_lbound[0, 0, 0] = 4
        self.t_prop_counts_template("square", nodes_sq_lbound, out_sq_lbound, bc="rbc")

        out_sq_tbound[0, -1, 3] = 1
        out_sq_tbound[1, -1, 3] = 2
        out_sq_tbound[2, -1, 3] = 3
        self.t_prop_counts_template("square", nodes_sq_tbound, out_sq_tbound, bc="rbc")

        out_sq_bbound[0, 0, 1] = 1
        out_sq_bbound[1, 0, 1] = 2
        out_sq_bbound[2, 0, 1] = 3
        self.t_prop_counts_template("square", nodes_sq_bbound, out_sq_bbound, bc="rbc")

    def test_pbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        out_hex_rbound[0, :, 0] = np.arange(1, 7)
        self.t_prop_counts_template("hex", nodes_hex_rbound, out_hex_rbound, bc="pbc")

        out_hex_trbound[0, 1::2, 1] = np.array([1, 3, 5])
        out_hex_trbound[-1, 0, 1] = 6
        out_hex_trbound[0, 0, 1] = 7
        out_hex_trbound[1, 0, 1] = 8
        self.t_prop_counts_template("hex", nodes_hex_trbound, out_hex_trbound, bc="pbc")

        out_hex_tlbound[-1, 2::2, 2] = np.array([2, 4])
        out_hex_tlbound[-1, 0, 2] = 6
        out_hex_tlbound[0, 0, 2] = 7
        out_hex_tlbound[-2, 0, 2] = 8
        self.t_prop_counts_template("hex", nodes_hex_tlbound, out_hex_tlbound, bc="pbc")

        out_hex_lbound[-1, :, 3] = np.arange(1, 7)
        self.t_prop_counts_template("hex", nodes_hex_lbound, out_hex_lbound, bc="pbc")

        out_hex_blbound[-1, 0::2, 4] = np.array([2, 4, 6])
        out_hex_blbound[-2:, -1, 4] = np.array([7, 8])
        out_hex_blbound[0, -1, 4] = 1
        self.t_prop_counts_template("hex", nodes_hex_blbound, out_hex_blbound, bc="pbc")

        out_hex_brbound[0, 1::2, 5] = np.array([3, 5, 1])
        out_hex_brbound[1, -1, 5] = 7
        out_hex_brbound[2, -1, 5] = 8
        self.t_prop_counts_template("hex", nodes_hex_brbound, out_hex_brbound, bc="pbc")

    def test_rbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        out_hex_rbound[-1, :, 3] = np.arange(1, 7)
        self.t_prop_counts_template("hex", nodes_hex_rbound, out_hex_rbound, bc="rbc")

        out_hex_trbound[-1, 0::2, 4] = np.array([1, 3, 5])
        out_hex_trbound[0, -1, 4] = 7
        out_hex_trbound[1, -1, 4] = 8
        out_hex_trbound[-1, -1, 4] = 6
        self.t_prop_counts_template("hex", nodes_hex_trbound, out_hex_trbound, bc="rbc")

        out_hex_tlbound[0, 1::2, 5] = np.array([2, 4, 6])
        out_hex_tlbound[1, -1, 5] = 7
        out_hex_tlbound[-1, -1, 5] = 8
        self.t_prop_counts_template("hex", nodes_hex_tlbound, out_hex_tlbound, bc="rbc")

        out_hex_lbound[0, :, 0] = np.arange(1, 7)
        self.t_prop_counts_template("hex", nodes_hex_lbound, out_hex_lbound, bc="rbc")

        out_hex_blbound[0, 1::2, 1] = np.array([2, 4, 6])
        out_hex_blbound[0, 0, 1] = 1
        out_hex_blbound[-2:, 0, 1] = np.array([7, 8])
        self.t_prop_counts_template("hex", nodes_hex_blbound, out_hex_blbound, bc="rbc")

        out_hex_brbound[-1, 0::2, 2] = np.array([1, 3, 5])
        out_hex_brbound[0, 0, 2] = 7
        out_hex_brbound[1, 0, 2] = 8
        self.t_prop_counts_template("hex", nodes_hex_brbound, out_hex_brbound, bc="rbc")

