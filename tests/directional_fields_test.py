"""Physical-coordinate and evolving-field cue oracles."""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, build_model
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec


@pytest.mark.parametrize("geometry,dims", [("lin", (4,)), ("square", (4, 4)),
                                         ("hex", (4, 4)), ("cubic", (4, 4, 4))])
def test_chemotaxis_ramps_have_analytic_physical_gradients_and_scores(geometry, dims):
    reference = get_lgca(geometry=geometry, dims=dims, density=0, interaction="only_propagation")
    ndim = len(dims)
    coordinates = [getattr(reference, name) for name in ("xcoords", "ycoords", "zcoords")[:ndim]]
    for component, coordinate in enumerate(coordinates):
        model = build_model(ModelSpec(
            space=SpaceSpec(geometry=geometry, dims=dims),
            state=StateSpec(density=0, fields={"signal": coordinate}),
            dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
                ReorientationTermSpec("chemotaxis", parameters={"field": "signal"}),
            ])]),
        ))
        term = model.pipeline.operators[0].terms[0]
        expected = np.eye(ndim)[component]
        np.testing.assert_allclose(term.gradient, np.broadcast_to(expected, dims + (ndim,)), atol=1e-14)
        candidates = np.eye(model.lgca.K, dtype=bool)
        for spatial in np.ndindex(dims):
            coord = tuple(index + model.lgca.r_int for index in spatial)
            scores = term.score(candidates, None, model.lgca, coord)
            np.testing.assert_allclose(scores, model.lgca.c[component], atol=1e-14)
