import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="requires matplotlib for plotting tests")

from lgca.plots import IdentityColourMapper, get_cmap


def test_colour_consistency_and_cycle():
    # simple family tree: 0 has children 1 and 2; 1 has child 3
    children_nlist = [[1, 2], [3], [], []]
    parent_list = [0, 0, 0, 1]
    cmap = ['red', 'green', 'blue']
    mapper = IdentityColourMapper(cmap, children_nlist, parent_list)

    col_root = mapper.get_colour(0)
    col_child1 = mapper.get_colour(1)
    col_child2 = mapper.get_colour(2)
    col_grandchild = mapper.get_colour(3)

    # repeated calls return the same colour
    assert mapper.get_colour(1) == col_child1

    # siblings and parent get different colours until the cycle repeats
    assert col_root != col_child1
    assert col_root != col_child2
    assert col_child1 != col_child2
    # cycle repeats after third colour
    assert col_grandchild == col_root


def test_invalid_tree_raises_typeerror():
    cmap = ['red', 'green']
    parent_list = [0]
    # children_nlist not iterable/list triggers TypeError
    with pytest.raises(TypeError):
        IdentityColourMapper(cmap, 5, parent_list)
    # parent_list must be a list
    with pytest.raises(TypeError):
        IdentityColourMapper(cmap, [[1], []], 'parent')


def test_plot_helpers_work_without_cm_get_cmap(monkeypatch):
    import matplotlib.cm as cm

    monkeypatch.delattr(cm, "get_cmap", raising=False)
    children_nlist = [[1], []]
    parent_list = [0, 0]

    mapper = IdentityColourMapper("tab20", children_nlist, parent_list)
    scalar_mappable = get_cmap(np.array([[0, 1], [2, 0]]), cbar=False)

    assert mapper.get_colour(0) is not None
    assert hasattr(scalar_mappable, "to_rgba")
