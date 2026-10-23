"""Transport and boundaries of the lattices, checked against independent oracles.

The multi-species and identity-based classes reuse the transport of the classical lattices; the tests compare
them with a single-species classical model run on the same cells, which is deterministic for propagation
and boundary conditions.
"""

import importlib
import sys
import warnings

import numpy as np
import pytest

from lgca import get_lgca
from lgca.list_utils import get_arr_of_empty_lists


@pytest.mark.parametrize("bc", ["reflecting", "periodic", "absorbing"])
@pytest.mark.parametrize("geometry, dims", [("moore", (4, 3, 5)), ("cubic", (4, 3, 5)), ("square", (5, 4)),
                                            ("hex", (5, 4)), ("lin", (7,))])
def test_each_species_moves_like_a_single_species_model(geometry, dims, bc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the rest-channel default of the models without volume exclusion
        multi = get_lgca(geometry=geometry, ve=False, n_species=2, dims=dims, density=2, bc=bc, seed=2,
                         interaction="only_propagation")
        singles = [get_lgca(geometry=geometry, ve=False, nodes=multi.nodes[multi.nonborder][..., s, :], bc=bc,
                            interaction="only_propagation") for s in range(2)]
    for _ in range(4):
        multi.timestep()
        for single in singles:
            single.timestep()
    for s, single in enumerate(singles):
        np.testing.assert_array_equal(multi.nodes[multi.nonborder][..., s, :], single.nodes[single.nonborder])


def _label_lists(counts):
    """Label lists with ``counts`` cells per channel, labels 1, 2, ... in C order."""
    nodes = get_arr_of_empty_lists(counts.shape)
    label = 1
    for index in np.ndindex(counts.shape):
        nodes[index] = list(range(label, label + counts[index]))
        label += counts[index]
    return nodes


@pytest.mark.parametrize("geometry, dims", [("moore", (3, 4, 3)), ("cubic", (3, 4, 3)), ("square", (5, 6)),
                                            ("hex", (5, 6)), ("lin", (7,))])
def test_reflecting_label_lists_move_like_counts(geometry, dims):
    # the list representation of identity-based models without volume exclusion: cells leaving the lattice
    # come back in the reflected channel, as counted by the classical model, and keep their labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts_model = get_lgca(geometry=geometry, ve=False, dims=dims, density=0, restchannels=1,
                                bc="reflecting", interaction="only_propagation")
        ib = get_lgca(geometry=geometry, ve=False, ib=True, dims=dims, density=0, bc="reflecting",
                      interaction="random_walk")
    counts = np.random.default_rng(0).integers(0, 3, ib.nodes.shape)
    ib.nodes = _label_lists(counts)
    counts_model.nodes = counts.astype(counts_model.nodes.dtype)
    labels_before = {label for channel in ib.nodes[ib.nonborder].flat for label in channel}
    ib.apply_rbc()
    counts_model.apply_rbc()
    lengths = np.vectorize(len, otypes=[int])(ib.nodes)
    np.testing.assert_array_equal(lengths, counts_model.nodes)
    labels_after = [label for channel in ib.nodes[ib.nonborder].flat for label in channel]
    assert len(labels_after) == len(set(labels_after))
    assert labels_before <= set(labels_after)


def test_base_extensions_is_deprecated():
    sys.modules.pop("lgca.base_extensions", None)
    with pytest.warns(DeprecationWarning, match="lgca.ib_base"):
        module = importlib.import_module("lgca.base_extensions")
    from lgca.ib_base import IBLGCA_base

    assert module.IBLGCA_base is IBLGCA_base
