import numpy as np
import pytest

from lgca.model import (
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    model_spec_from_dict,
)
from lgca.pipeline import InteractionPipelineSpec


def _serialized_model():
    return {
        "schema_version": 1,
        "model": {
            "description": {"title": "strict"},
            "space": {"geometry": "square", "dims": [3, 3], "boundary": "periodic"},
            "state": {"density": 0.2, "restchannels": 1},
            "time": {"steps": 1, "seed": 1},
            "dynamics": {"operators": [], "propagation": False},
            "analysis": {"observers": []},
        },
    }


def test_unknown_serialized_section_key_reports_full_path_and_suggestion():
    data = _serialized_model()
    data["model"]["state"]["densitty"] = 0.4
    with pytest.raises(ValueError, match=r"model\.state\.densitty.*density"):
        model_spec_from_dict(data)


def test_unknown_serialized_top_level_key_is_rejected():
    data = _serialized_model()
    data["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        model_spec_from_dict(data)


@pytest.mark.parametrize(
    "path,value",
    [
        (("model", "description", "tags"), "not-a-sequence"),
        (("model", "state", "volume_exclusion"), 1),
        (("model", "state", "restchannels"), 1.5),
        (("model", "time", "seed"), True),
    ],
)
def test_wrong_serialized_scalar_and_sequence_types_are_rejected(path, value):
    data = _serialized_model()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError), match=r"\.".join(path)):
        model_spec_from_dict(data)


def test_runtime_class_operator_tags_are_not_accepted_by_wire_parser():
    data = _serialized_model()
    data["model"]["dynamics"]["operators"] = [
        {"type": "PythonClass", "module": "untrusted.module", "name": "run"}
    ]

    with pytest.raises(ValueError, match=r"model\.dynamics\.operators\[0\]"):
        model_spec_from_dict(data)


def test_serialized_observer_rejects_unknown_type_and_options():
    data = _serialized_model()
    data["model"]["analysis"]["observers"] = [
        {"type": "ExternalObserver", "import_path": "untrusted.Observer"}
    ]

    with pytest.raises(ValueError, match=r"model\.analysis\.observers\[0\]"):
        model_spec_from_dict(data)


def test_initializer_declaration_rejects_unknown_keys_with_full_path():
    data = _serialized_model()
    data["model"]["state"].pop("density")
    data["model"]["state"]["initializer"] = {
        "name": "region",
        "paramters": {"density": 0.5},
    }

    with pytest.raises(ValueError, match=r"model\.state\.initializer\.paramters.*parameters"):
        model_spec_from_dict(data)


@pytest.mark.parametrize("geometry", ["triangle", 3])
def test_invalid_geometry_is_rejected_before_build(geometry):
    with pytest.raises((TypeError, ValueError), match="model.space.geometry"):
        build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=(3, 3))))


@pytest.mark.parametrize("boundary", ["wraparound", 3])
def test_invalid_boundary_is_rejected_before_build(boundary):
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(3, 3), boundary=boundary))
    with pytest.raises((TypeError, ValueError), match="model.space.boundary"):
        build_model(spec)


@pytest.mark.parametrize("steps", [True, 1.5, -1])
def test_invalid_step_count_is_rejected_without_coercion(steps):
    with pytest.raises(ValueError, match="model.time.steps"):
        build_model(ModelSpec(time=TimeSpec(steps=steps)))


@pytest.mark.parametrize(
    "reserved", ["bc", "seed", "nodes", "density", "restchannels", "interaction", "dims"]
)
def test_state_parameters_cannot_override_canonical_configuration(reserved):
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 3)),
        state=StateSpec(density=0.2, parameters={reserved: 999}),
    )
    with pytest.raises(ValueError, match=rf"model\.state\.parameters\.{reserved}"):
        build_model(spec)


@pytest.mark.parametrize("field_name", ["nodes", "cell_density", "rng", "geometry", "timestep"])
def test_state_fields_cannot_replace_simulator_state_or_methods(field_name):
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 3)),
        state=StateSpec(density=0.2, fields={field_name: np.zeros((3, 3))}),
    )
    with pytest.raises(ValueError, match=rf"model\.state\.fields\.{field_name}"):
        build_model(spec)


def test_unknown_plugin_parameter_is_rejected_with_suggestion():
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 3)),
        state=StateSpec(density=0.2),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.birth", "parameters": {"birth_raet": 0.2}}],
            propagation=False,
        ),
    )
    with pytest.raises(ValueError, match="birth_raet"):
        build_model(spec)


def test_non_finite_plugin_probability_is_rejected():
    spec = ModelSpec(
        description=Description(title="non-finite"),
        space=SpaceSpec(geometry="square", dims=(3, 3)),
        state=StateSpec(density=0.2),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.birth", "parameters": {"r_b": np.nan}}],
            propagation=False,
        ),
    )
    with pytest.raises(ValueError, match=r"r_b.*probability"):
        build_model(spec)


def test_metadata_uses_normalized_runtime_geometry_and_boundary():
    compiled = build_model(
        ModelSpec(
            space=SpaceSpec(geometry="sq", dims=(3, 3), boundary="pbc"),
            state=StateSpec(density=0.2, restchannels=1),
            time=TimeSpec(steps=1, seed=7),
        )
    )
    assert compiled.spec.space.geometry == compiled.lgca.geometry == "square"
    assert compiled.spec.space.boundary == compiled.lgca.bc == "periodic"
    assert compiled.metadata["geometry"] == "square"
    assert compiled.metadata["boundary"] == "periodic"


def _capacity_spec(operator_capacity):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 3)),
        state=StateSpec(
            density=0.2,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            capacity=8,
        ),
        dynamics=InteractionPipelineSpec(
            operators=[
                {"name": "nove_ib.birthdeath", "parameters": {"capacity": operator_capacity}}
            ],
            propagation=False,
        ),
    )


def test_conflicting_operator_capacity_is_rejected():
    with pytest.raises(ValueError, match=r"capacity.*model.state.capacity"):
        build_model(_capacity_spec(7))


def test_equal_operator_capacity_is_temporarily_accepted_with_warning():
    with pytest.warns(DeprecationWarning, match="model.state.capacity"):
        compiled = build_model(_capacity_spec(8))
    assert compiled.lgca.capacity == 8
