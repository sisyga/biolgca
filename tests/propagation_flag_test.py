import numpy as np
from lgca import get_lgca


def test_disable_propagation():
    nodes = np.zeros((3, 3))
    nodes[1, 0] = 1  # particle moving right
    lgca = get_lgca(geometry='lin', nodes=nodes, ve=True, ib=False,
                    interaction='only_propagation', propagation=False)
    before = lgca.nodes.copy()
    lgca.timeevo(timesteps=1, recorddens=False, showprogress=False)
    assert np.array_equal(lgca.nodes, before)
