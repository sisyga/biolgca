"""Continuation must preserve dynamics, family recording and time provenance."""

from dataclasses import replace

import numpy as np
import pytest

from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import FamilyPopulationRecorder, NodeRecorder, SimulationRunner


@pytest.mark.parametrize("entry", ["compiled", "timeevo", "runner"])
def test_compiled_family_growth_recording_and_continuation(entry):
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=1),
        state=StateSpec(identity_based=True, nodes=np.array([[1, 0, 0, 0]], dtype=np.uint64)),
        time=TimeSpec(steps=4, seed=42),
        dynamics=InteractionPipelineSpec(operators=[{"name": "ib.go_and_grow_mutations",
            "parameters": {"r_b": 1., "r_d": 0., "r_m": 1.}}], propagation=False),
        analysis=AnalysisSpec(observers=[FamilyPopulationRecorder(), NodeRecorder()]),
    )
    uninterrupted = build_model(spec)
    uninterrupted.run(False)
    model = build_model(replace(spec, time=replace(spec.time, steps=2)))
    for _ in range(2):
        if entry == "compiled":
            model.run(False)
        elif entry == "timeevo":
            model.lgca.timeevo(2, record=True, recordfampop=True, showprogress=False)
        else:
            SimulationRunner(model.lgca, timesteps=2,
                             observers=[FamilyPopulationRecorder(), NodeRecorder()], showprogress=False).run()
        populations = np.count_nonzero(model.lgca.nodes_t, axis=(1, 2))
        np.testing.assert_array_equal(model.lgca.fam_pop_t.sum(-1), populations)
        # Every birth creates a new family; no deaths: one living cell per family.
        assert np.all(model.lgca.fam_pop_t[:, 1:] <= 1)
        assert np.all(np.diff(populations) >= 0)
        labels = model.lgca.nodes[model.lgca.nonborder]
        living = labels[labels > 0]
        families = np.asarray(model.lgca.props["family"])[living]
        assert len(np.unique(families)) == len(living)
    assert model._step == 4
    assert model.lgca.maxfamily > 1
    assert model.lgca.enable_propagation is False
    np.testing.assert_array_equal(model.lgca.nodes, uninterrupted.lgca.nodes)
    assert model.lgca.props == uninterrupted.lgca.props
    assert model.lgca.rng.bit_generator.state == uninterrupted.lgca.rng.bit_generator.state
