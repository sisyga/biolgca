"""Physical-coordinate and evolving-field cue oracles."""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, build_model
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.operator_base import InteractionOperator, PluginInfo


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


@pytest.mark.parametrize("name", ["chemotaxis", "contact_guidance"])
@pytest.mark.parametrize("writer", [False, True])
def test_current_named_fields_steer_next_step_once_per_operator(name, writer, monkeypatch):
    class WriteField(InteractionOperator):
        def __init__(self):
            super().__init__(PluginInfo("write_field", "reorientation", ("classical",)))
            self.changed = False

        def apply(self, context, step):
            if self.changed:
                change_field(context.lgca)

    def change_field(lgca):
        if name == "chemotaxis":
            lgca.signal *= -1
        else:
            lgca.signal[...] = [0, 1]

    x = np.broadcast_to(np.arange(4.0)[:, None], (4, 4))
    field = x if name == "chemotaxis" else np.broadcast_to([1.0, 0], (4, 4, 2))
    nodes = np.zeros((4, 4, 4), dtype=bool)
    nodes[..., 0] = True
    field_writer = WriteField()
    operators = [field_writer] if writer else []
    operators.append(ReorientationSpec(terms=[ReorientationTermSpec(
        name, beta=1000, parameters={"field": "signal"},
    )]))
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(4, 4)),
        state=StateSpec(nodes=nodes, fields={"signal": field}),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False),
    ))
    term = model.pipeline.operators[-1].terms[0]
    calls = []
    original = term.prepare

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(term, "prepare", counted)
    model.step()
    flux = model.lgca.calc_flux(model.lgca.nodes[model.lgca.nonborder])
    np.testing.assert_array_equal(np.abs(flux[..., 0]), 1)
    if writer:
        field_writer.changed = True
    else:
        change_field(model.lgca)
    model.step()
    flux = model.lgca.calc_flux(model.lgca.nodes[model.lgca.nonborder])
    if name == "chemotaxis":
        np.testing.assert_array_equal(flux[..., 0], -1)
    else:
        np.testing.assert_array_equal(np.abs(flux[..., 1]), 1)
        # A nematic director and its negative describe the same axis.
        previous = term.director.copy()
        field_writer.changed = False
        model.lgca.signal *= -1
        model.step()
        np.testing.assert_array_equal(term.director, -previous)
        flux = model.lgca.calc_flux(model.lgca.nodes[model.lgca.nonborder])
        np.testing.assert_array_equal(np.abs(flux[..., 1]), 1)
    assert len(calls) == (3 if name == "contact_guidance" else 2)
