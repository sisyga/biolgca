"""Physical identity validation excludes legitimate halo copies."""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, build_model


@pytest.mark.parametrize("entry", ["direct", "model"])
@pytest.mark.parametrize("case", ["within", "channels", "sites", "negative", "fractional",
                                  "nan", "boolean", "string", "oversized", "not_list", "valid"])
def test_nove_identity_input_labels(entry, case):
    nodes = np.empty((2, 3), dtype=object)
    for coord in np.ndindex(nodes.shape):
        nodes[coord] = []
    nodes[0, 0] = [0, 7]
    nodes[1, 2] = [23]
    if case == "within":
        nodes[0, 0] = [7, 7]
    elif case == "channels":
        nodes[0, 1] = [7]
    elif case == "sites":
        nodes[1, 0] = [7]
    elif case != "valid":
        invalid = {"negative": -1, "fractional": 1.5, "nan": np.nan,
                   "boolean": True, "string": "1", "oversized": 2**64}
        nodes[0, 0] = (0, 7) if case == "not_list" else [invalid[case]]

    def construct():
        if entry == "direct":
            return get_lgca(geometry="lin", ve=False, ib=True, nodes=nodes,
                            interaction="only_propagation")
        return build_model(ModelSpec(space=SpaceSpec(geometry="lin"),
            state=StateSpec(identity_based=True, volume_exclusion=False, nodes=nodes))).lgca

    if case != "valid":
        with pytest.raises(ValueError, match="nodes"):
            construct()
    else:
        model = construct()
        assert sorted(sum(model.nodes[model.nonborder].flat, [])) == [0, 7, 23]
        assert model.maxlabel == 23
        assert model.total_population() == 3
