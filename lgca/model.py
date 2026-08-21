"""Declarative model specifications and execution helpers."""

from __future__ import annotations

import importlib.metadata
import difflib
import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from . import get_lgca
from .pipeline import (
    BirthDeathSpec,
    InteractionPipelineSpec,
    PhenotypeSwitchSpec,
    ReorientationSpec,
    ReorientationTermSpec,
    compile_pipeline,
)


MODEL_SPEC_SCHEMA_VERSION = 1

__all__ = [
    "AnalysisSpec",
    "CompiledModel",
    "Description",
    "MODEL_SPEC_SCHEMA_VERSION",
    "ModelContext",
    "ModelRunResult",
    "ModelSpec",
    "SpaceSpec",
    "StateSpec",
    "TimeSpec",
    "build_model",
    "describe_model_graph",
    "load_model_spec",
    "migrate_model_spec_dict",
    "model_spec_from_dict",
    "model_spec_from_json",
    "model_spec_from_yaml",
    "model_spec_to_dict",
    "model_spec_to_json",
    "model_spec_to_yaml",
    "run_model",
    "save_model_spec",
]


@dataclass(frozen=True)
class Description:
    """Human-facing model metadata."""

    title: str
    details: str = ""
    tags: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return self.details


@dataclass(frozen=True)
class SpaceSpec:
    """Lattice geometry and boundary configuration."""

    geometry: str = "hex"
    dims: Any = None
    boundary: str = "periodic"


@dataclass(frozen=True)
class StateSpec:
    """Initial state and backend family selection."""

    density: float | None = None
    nodes: Any = None
    restchannels: int = 0
    volume_exclusion: bool = True
    identity_based: bool = False
    n_species: int = 1
    capacity: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeSpec:
    """Time horizon and random seed."""

    steps: int = 100
    seed: int | None = None


@dataclass(frozen=True)
class AnalysisSpec:
    """Observers and post-processing hooks."""

    observers: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelSpec:
    """Complete declarative LGCA model specification."""

    description: Description = field(default_factory=lambda: Description(title="LGCA model"))
    space: SpaceSpec = field(default_factory=SpaceSpec)
    state: StateSpec = field(default_factory=StateSpec)
    time: TimeSpec = field(default_factory=TimeSpec)
    dynamics: InteractionPipelineSpec = field(default_factory=InteractionPipelineSpec)
    analysis: AnalysisSpec | None = field(default_factory=AnalysisSpec)


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    """Convert a :class:`ModelSpec` to a JSON-compatible dictionary."""

    return {
        "schema_version": MODEL_SPEC_SCHEMA_VERSION,
        "model": {
            "description": {
                "title": spec.description.title,
                "details": spec.description.details,
                "tags": list(spec.description.tags),
            },
            "space": {
                "geometry": spec.space.geometry,
                "dims": _to_jsonable(spec.space.dims),
                "boundary": spec.space.boundary,
            },
            "state": {
                "density": _to_jsonable(spec.state.density),
                "nodes": _to_jsonable(spec.state.nodes),
                "restchannels": spec.state.restchannels,
                "volume_exclusion": spec.state.volume_exclusion,
                "identity_based": spec.state.identity_based,
                "n_species": spec.state.n_species,
                "capacity": spec.state.capacity,
                "parameters": _to_jsonable(dict(spec.state.parameters)),
                "fields": _to_jsonable(dict(spec.state.fields)),
            },
            "time": {
                "steps": spec.time.steps,
                "seed": spec.time.seed,
            },
            "dynamics": {
                "operators": [_operator_to_dict(operator) for operator in spec.dynamics.operators],
                "propagation": spec.dynamics.propagation,
                "allow_custom_order": spec.dynamics.allow_custom_order,
            },
            "analysis": None if spec.analysis is None else {
                "observers": [_observer_to_dict(observer) for observer in spec.analysis.observers],
            },
        },
    }


def model_spec_from_dict(data: Mapping[str, Any]) -> ModelSpec:
    """Build a :class:`ModelSpec` from :func:`model_spec_to_dict` output."""

    if not isinstance(data, Mapping):
        raise TypeError("model spec must be a mapping")
    data = migrate_model_spec_dict(dict(data))
    _validate_serialized_model(data)
    model = data["model"]
    description = model.get("description", {})
    space = model.get("space", {})
    state = model.get("state", {})
    time_spec = model.get("time", {})
    dynamics = model.get("dynamics", {})
    analysis = model.get("analysis")
    return ModelSpec(
        description=Description(
            title=description.get("title", "LGCA model"),
            details=description.get("details", ""),
            tags=tuple(description.get("tags", ())),
        ),
        space=SpaceSpec(
            geometry=space.get("geometry", "hex"),
            dims=_from_jsonable(space.get("dims")),
            boundary=space.get("boundary", "periodic"),
        ),
        state=StateSpec(
            density=_from_jsonable(state.get("density")),
            nodes=_from_jsonable(state.get("nodes")),
            restchannels=state.get("restchannels", 0),
            volume_exclusion=state.get("volume_exclusion", True),
            identity_based=state.get("identity_based", False),
            n_species=state.get("n_species", 1),
            capacity=state.get("capacity"),
            parameters=_from_jsonable(state.get("parameters", {})),
            fields=_from_jsonable(state.get("fields", {})),
        ),
        time=TimeSpec(
            steps=time_spec.get("steps", 100),
            seed=time_spec.get("seed"),
        ),
        dynamics=InteractionPipelineSpec(
            operators=tuple(_operator_from_dict(operator) for operator in dynamics.get("operators", ())),
            propagation=dynamics.get("propagation", "default"),
            allow_custom_order=dynamics.get("allow_custom_order", False),
        ),
        analysis=None if analysis is None else AnalysisSpec(
            observers=tuple(_observer_from_dict(observer) for observer in analysis.get("observers", ())),
        ),
    )


def model_spec_to_json(spec: ModelSpec, path: str | Path | None = None) -> str:
    """Serialize a :class:`ModelSpec` to JSON text or a file path."""

    text = json.dumps(model_spec_to_dict(spec), indent=2)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def model_spec_from_json(source: str | Path) -> ModelSpec:
    """Read a :class:`ModelSpec` from JSON text or a JSON file path."""

    return model_spec_from_dict(json.loads(_read_text_source(source)))


def model_spec_to_yaml(spec: ModelSpec, path: str | Path | None = None) -> str:
    """Serialize a :class:`ModelSpec` to YAML-compatible text.

    The emitted text is indented JSON, which is valid YAML and avoids adding a
    required YAML dependency.
    """

    data = model_spec_to_dict(spec)
    try:
        import yaml
    except ImportError:
        text = json.dumps(data, indent=2)
    else:
        text = yaml.safe_dump(data, sort_keys=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def model_spec_from_yaml(source: str | Path) -> ModelSpec:
    """Read a :class:`ModelSpec` from YAML text or a YAML file path."""

    text = _read_text_source(source)
    try:
        import yaml
    except ImportError:
        try:
            return model_spec_from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Reading YAML model specs requires PyYAML. Install the docs/dev extras "
                "or use a .json model spec."
            ) from exc
    return model_spec_from_dict(yaml.safe_load(text))


def save_model_spec(
    spec: ModelSpec,
    path: str | Path,
    file_format: str | None = None,
) -> Path:
    """Save a model specification to ``.json``, ``.yaml`` or ``.yml``.

    This is the beginner-facing persistence helper. It chooses the format from
    the file suffix unless ``file_format`` is provided.
    """

    target = Path(path)
    selected = _resolve_model_spec_format(target, file_format)
    if selected == "json":
        model_spec_to_json(spec, target)
    elif selected == "yaml":
        model_spec_to_yaml(spec, target)
    else:
        raise ValueError("Use a .json, .yaml, or .yml file for model specs.")
    return target


def load_model_spec(
    source: str | Path,
    file_format: str | None = None,
) -> ModelSpec:
    """Load a model specification from a path or JSON/YAML text."""

    path = _source_path(source)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Could not find model spec file: {path}")
        selected = _resolve_model_spec_format(path, file_format)
        if selected == "json":
            return model_spec_from_json(path)
        if selected == "yaml":
            return model_spec_from_yaml(path)
        raise ValueError("Use a .json, .yaml, or .yml file for model specs.")

    selected = _normalize_model_spec_format(file_format)
    text = str(source)
    if selected == "json" or (selected is None and text.lstrip().startswith(("{", "["))):
        return model_spec_from_json(text)
    if selected in (None, "yaml"):
        return model_spec_from_yaml(text)
    raise ValueError("file_format must be 'json' or 'yaml'.")


def migrate_model_spec_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate serialized model spec data to the current schema version."""

    version = data.get("schema_version", MODEL_SPEC_SCHEMA_VERSION)
    if version == MODEL_SPEC_SCHEMA_VERSION:
        if "schema_version" not in data:
            data = dict(data)
            data["schema_version"] = MODEL_SPEC_SCHEMA_VERSION
        return data
    raise ValueError(f"Unsupported ModelSpec schema_version {version!r}.")


def _validate_serialized_model(data: Mapping[str, Any]) -> None:
    _reject_unknown_keys(data, {"schema_version", "model"}, "")
    model = _mapping_at(data.get("model"), "model")
    _reject_unknown_keys(
        model,
        {"description", "space", "state", "time", "dynamics", "analysis"},
        "model",
    )
    description = _mapping_at(model.get("description", {}), "model.description")
    space = _mapping_at(model.get("space", {}), "model.space")
    state = _mapping_at(model.get("state", {}), "model.state")
    time_spec = _mapping_at(model.get("time", {}), "model.time")
    dynamics = _mapping_at(model.get("dynamics", {}), "model.dynamics")
    analysis = model.get("analysis")
    if analysis is not None:
        analysis = _mapping_at(analysis, "model.analysis")

    _reject_unknown_keys(description, {"title", "details", "tags"}, "model.description")
    _reject_unknown_keys(space, {"geometry", "dims", "boundary"}, "model.space")
    _reject_unknown_keys(
        state,
        {
            "density", "nodes", "restchannels", "volume_exclusion", "identity_based",
            "n_species", "capacity", "parameters", "fields",
        },
        "model.state",
    )
    _reject_unknown_keys(time_spec, {"steps", "seed"}, "model.time")
    _reject_unknown_keys(
        dynamics, {"operators", "propagation", "allow_custom_order"}, "model.dynamics"
    )
    if analysis is not None:
        _reject_unknown_keys(analysis, {"observers"}, "model.analysis")

    for key in ("title", "details"):
        if key in description and not isinstance(description[key], str):
            raise TypeError(f"model.description.{key} must be a string")
    tags = description.get("tags", ())
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("model.description.tags must be a sequence of strings")
    if not all(isinstance(tag, str) for tag in tags):
        raise TypeError("model.description.tags must be a sequence of strings")
    for key in ("volume_exclusion", "identity_based"):
        if key in state and not isinstance(state[key], bool):
            raise TypeError(f"model.state.{key} must be a boolean")
    for key in ("parameters", "fields"):
        if key in state and not isinstance(state[key], Mapping):
            raise TypeError(f"model.state.{key} must be a mapping")
    for key in ("restchannels", "n_species", "capacity"):
        value = state.get(key)
        if value is not None and (isinstance(value, bool) or int(value) != value):
            raise TypeError(f"model.state.{key} must be an integer")
    for key in ("steps", "seed"):
        value = time_spec.get(key)
        if value is not None and (isinstance(value, bool) or int(value) != value):
            raise TypeError(f"model.time.{key} must be an integer")
    for key in ("operators",):
        value = dynamics.get(key, ())
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"model.dynamics.{key} must be a sequence")
    if "allow_custom_order" in dynamics and not isinstance(
        dynamics["allow_custom_order"], bool
    ):
        raise TypeError("model.dynamics.allow_custom_order must be a boolean")
    if analysis is not None:
        observers = analysis.get("observers", ())
        if isinstance(observers, (str, bytes)) or not isinstance(observers, Sequence):
            raise TypeError("model.analysis.observers must be a sequence")


def _mapping_at(value, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _reject_unknown_keys(mapping, allowed, path: str) -> None:
    for key in sorted(set(mapping) - set(allowed)):
        full_path = f"{path}.{key}" if path else str(key)
        matches = difflib.get_close_matches(str(key), sorted(allowed), n=1)
        suggestion = f"; did you mean {matches[0]!r}?" if matches else ""
        raise ValueError(f"unknown configuration key {full_path}{suggestion}")


def describe_model_graph(spec: ModelSpec) -> dict[str, Any]:
    """Return a lightweight dependency graph for a model specification."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, kind: str, label: str | None = None) -> None:
        if not any(node["id"] == node_id for node in nodes):
            nodes.append({"id": node_id, "kind": kind, "label": label or node_id})

    add_node("state:nodes", "state", "nodes")
    for field_name in spec.state.fields:
        add_node(f"field:{field_name}", "field", field_name)

    for index, operator in enumerate(spec.dynamics.operators):
        name, dependencies = _operator_graph_info(operator)
        operator_id = f"operator:{index}:{name}"
        add_node(operator_id, "operator", name)
        for dependency in dependencies:
            add_node(f"field:{dependency}", "field", dependency)
            edges.append({"source": f"field:{dependency}", "target": operator_id})
        add_node("output:nodes", "output", "nodes")
        edges.append({"source": operator_id, "target": "output:nodes"})

    if spec.analysis is not None:
        for index, observer in enumerate(spec.analysis.observers):
            name = observer.__class__.__name__
            observer_id = f"observer:{index}:{name}"
            output = _observer_output_name(observer)
            add_node(observer_id, "observer", name)
            add_node(f"output:{output}", "output", output)
            edges.append({"source": observer_id, "target": f"output:{output}"})

    return {"schema_version": 1, "nodes": nodes, "edges": edges}


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_to_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _from_jsonable(value):
    if isinstance(value, Mapping):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=value.get("dtype"))
        if "__tuple__" in value:
            return tuple(_from_jsonable(item) for item in value["__tuple__"])
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _operator_to_dict(operator) -> dict[str, Any]:
    if isinstance(operator, BirthDeathSpec):
        return {
            "type": "BirthDeathSpec",
            "name": operator.name,
            "parameters": _to_jsonable(dict(operator.parameters)),
        }
    if isinstance(operator, PhenotypeSwitchSpec):
        return {
            "type": "PhenotypeSwitchSpec",
            "name": operator.name,
            "parameters": _to_jsonable(dict(operator.parameters)),
        }
    if isinstance(operator, ReorientationSpec):
        return {
            "type": "ReorientationSpec",
            "sampler": operator.sampler,
            "parameters": _to_jsonable(dict(operator.parameters)),
            "terms": [_reorientation_term_to_dict(term) for term in operator.terms],
        }
    if isinstance(operator, Mapping):
        return {"type": "Mapping", "value": _to_jsonable(dict(operator))}
    raise TypeError(f"Cannot serialize operator {operator!r}.")


def _operator_from_dict(data: Mapping[str, Any]):
    operator_type = data.get("type")
    if operator_type == "BirthDeathSpec":
        return BirthDeathSpec(
            name=data["name"],
            parameters=_from_jsonable(data.get("parameters", {})),
        )
    if operator_type == "PhenotypeSwitchSpec":
        return PhenotypeSwitchSpec(
            name=data["name"],
            parameters=_from_jsonable(data.get("parameters", {})),
        )
    if operator_type == "ReorientationSpec":
        return ReorientationSpec(
            terms=tuple(_reorientation_term_from_dict(term) for term in data.get("terms", ())),
            sampler=data.get("sampler", "boltzmann"),
            parameters=_from_jsonable(data.get("parameters", {})),
        )
    if operator_type == "Mapping":
        return _from_jsonable(data.get("value", {}))
    return _from_jsonable(dict(data))


def _reorientation_term_to_dict(term: ReorientationTermSpec) -> dict[str, Any]:
    return {
        "name": term.name,
        "beta": term.beta,
        "parameters": _to_jsonable(dict(term.parameters)),
        "species": term.species,
    }


def _reorientation_term_from_dict(data: Mapping[str, Any]) -> ReorientationTermSpec:
    return ReorientationTermSpec(
        name=data["name"],
        beta=data.get("beta", 1.0),
        parameters=_from_jsonable(data.get("parameters", {})),
        species=data.get("species"),
    )


def _observer_to_dict(observer) -> dict[str, Any]:
    data = {
        "type": observer.__class__.__name__,
        "schedule": _schedule_to_dict(getattr(observer, "schedule", None)),
    }
    if observer.__class__.__name__ == "DensityRecorder":
        dtype = getattr(observer, "dtype", None)
        data["dtype"] = None if dtype is None else np.dtype(dtype).name
    elif observer.__class__.__name__ == "CSVSnapshotObserver":
        data.update(
            {
                "kind": observer.kind,
                "output_dir": str(observer.output_dir),
                "filename": observer.filename,
            }
        )
    elif observer.__class__.__name__ == "ScalarTimeSeriesRecorder":
        if set(observer.metrics) != {"population"}:
            raise TypeError("ScalarTimeSeriesRecorder serialization only supports the default population metric.")
        data["output_path"] = str(observer.output_path)
    return data


def _observer_from_dict(data: Mapping[str, Any]):
    from .simulation import (
        CSVSnapshotObserver,
        ChannelDensityRecorder,
        DensityRecorder,
        FamilyPopulationRecorder,
        NodeRecorder,
        OrderParameterRecorder,
        PerTypeRecorder,
        PopulationRecorder,
        ScalarTimeSeriesRecorder,
    )

    observer_type = data["type"]
    schedule = _schedule_from_dict(data.get("schedule"))
    if observer_type == "NodeRecorder":
        return NodeRecorder(schedule=schedule)
    if observer_type == "PopulationRecorder":
        return PopulationRecorder(schedule=schedule)
    if observer_type == "DensityRecorder":
        dtype = data.get("dtype")
        return DensityRecorder(schedule=schedule, dtype=None if dtype is None else np.dtype(dtype).type)
    if observer_type == "ChannelDensityRecorder":
        return ChannelDensityRecorder(schedule=schedule)
    if observer_type == "PerTypeRecorder":
        return PerTypeRecorder(schedule=schedule)
    if observer_type == "OrderParameterRecorder":
        return OrderParameterRecorder(schedule=schedule)
    if observer_type == "FamilyPopulationRecorder":
        return FamilyPopulationRecorder(schedule=schedule)
    if observer_type == "CSVSnapshotObserver":
        return CSVSnapshotObserver(
            kind=data.get("kind", "density"),
            schedule=schedule,
            output_dir=data.get("output_dir"),
            filename=data.get("filename", "{kind}_{step:05d}.csv"),
        )
    if observer_type == "ScalarTimeSeriesRecorder":
        return ScalarTimeSeriesRecorder(schedule=schedule, output_path=data.get("output_path"))
    raise ValueError(f"Unknown observer type {observer_type!r}.")


def _schedule_to_dict(schedule) -> dict[str, Any] | None:
    if schedule is None:
        return None
    return {
        "every": schedule.every,
        "steps": None if schedule.steps is None else sorted(schedule.steps),
    }


def _schedule_from_dict(data):
    if data is None:
        return None
    from .simulation import Schedule

    return Schedule(every=data.get("every", 1), steps=data.get("steps"))


def _read_text_source(source: str | Path) -> str:
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Could not find model spec file: {source}")
        return source.read_text(encoding="utf-8")
    text = str(source)
    if text.lstrip().startswith(("{", "[")):
        return text
    path = Path(text)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return text


def _source_path(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source
    text = str(source)
    if text.lstrip().startswith(("{", "[")) or "\n" in text:
        return None
    suffix = Path(text).suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        return Path(text)
    return None


def _resolve_model_spec_format(path: Path, file_format: str | None) -> str:
    selected = _normalize_model_spec_format(file_format)
    if selected is not None:
        return selected
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    raise ValueError("Use a .json, .yaml, or .yml file for model specs.")


def _normalize_model_spec_format(file_format: str | None) -> str | None:
    if file_format is None:
        return None
    selected = file_format.lower().lstrip(".")
    if selected == "yml":
        selected = "yaml"
    if selected not in {"json", "yaml"}:
        raise ValueError("file_format must be 'json' or 'yaml'.")
    return selected


def _operator_graph_info(operator) -> tuple[str, set[str]]:
    dependencies: set[str] = set()
    if isinstance(operator, ReorientationSpec):
        for term in operator.terms:
            field = term.parameters.get("field")
            if field is not None:
                dependencies.add(str(field))
        return "reorientation.boltzmann", dependencies
    if isinstance(operator, Mapping):
        return str(operator.get("name", "<missing>")), dependencies
    name = getattr(operator, "name", "<missing>")
    parameters = getattr(operator, "parameters", {})
    field = parameters.get("field") if isinstance(parameters, Mapping) else None
    if field is not None:
        dependencies.add(str(field))
    return str(name), dependencies


def _observer_output_name(observer) -> str:
    output_by_name = {
        "NodeRecorder": "nodes_t",
        "PopulationRecorder": "n_t",
        "DensityRecorder": "dens_t",
        "ChannelDensityRecorder": "channel_pop_t",
        "PerTypeRecorder": "velcells_t",
        "OrderParameterRecorder": "order_parameters",
        "FamilyPopulationRecorder": "fam_pop_t",
        "CSVSnapshotObserver": "csv_snapshots",
        "ScalarTimeSeriesRecorder": "scalar_time_series",
    }
    return output_by_name.get(observer.__class__.__name__, observer.__class__.__name__)


@dataclass
class ModelContext:
    """Runtime context passed to plugins."""

    lgca: Any
    spec: ModelSpec
    fields: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class CompiledModel:
    """Built LGCA model plus compiled interaction pipeline."""

    lgca: Any
    spec: ModelSpec
    context: ModelContext
    pipeline: Any
    metadata: dict[str, Any]

    def run(self, showprogress: bool = True):
        return _run_compiled_model(self, showprogress=showprogress)


@dataclass
class ModelRunResult:
    """Result returned by :func:`run_model`."""

    lgca: Any
    spec: ModelSpec
    context: ModelContext
    pipeline: Any
    metadata: dict[str, Any]


def build_model(spec: ModelSpec) -> CompiledModel:
    """Build an LGCA instance and compile its interaction pipeline."""

    spec = _normalize_and_validate_spec(spec)
    lgca = _build_lgca(spec)
    _validate_field_names(lgca, spec.state.fields)
    metadata = _metadata_from_spec(spec, lgca=lgca)
    context = ModelContext(
        lgca=lgca,
        spec=spec,
        fields=dict(spec.state.fields),
        metadata=metadata,
    )
    _attach_fields(lgca, context.fields)
    pipeline = compile_pipeline(spec.dynamics, context)
    _attach_fields(lgca, context.fields)
    metadata["operator_names"] = pipeline.operator_names
    metadata["reorientation_term_names"] = pipeline.reorientation_term_names
    metadata["observer_names"] = _observer_names(spec.analysis)
    metadata["schedule"] = pipeline.describe_schedule()
    return CompiledModel(
        lgca=lgca,
        spec=spec,
        context=context,
        pipeline=pipeline,
        metadata=metadata,
    )


def run_model(spec: ModelSpec, showprogress: bool = True) -> ModelRunResult:
    """Build and run a declarative LGCA model."""

    compiled = build_model(spec)
    return compiled.run(showprogress=showprogress)


def _validate_spec(spec: ModelSpec) -> None:
    if isinstance(spec.time.steps, bool) or int(spec.time.steps) != spec.time.steps:
        raise ValueError("model.time.steps must be a non-negative integer")
    if spec.time.steps < 0:
        raise ValueError("model.time.steps must be a non-negative integer")
    if spec.time.seed is not None and (
        isinstance(spec.time.seed, bool) or int(spec.time.seed) != spec.time.seed
    ):
        raise ValueError("model.time.seed must be an integer or null")
    if spec.state.nodes is not None and spec.state.density is not None:
        raise ValueError("model.state.nodes and model.state.density are mutually exclusive")
    _validate_non_negative_integer("model.state.restchannels", spec.state.restchannels)
    _validate_positive_integer("model.state.n_species", spec.state.n_species)
    if spec.state.capacity is not None:
        _validate_positive_integer("model.state.capacity", spec.state.capacity)
    if not isinstance(spec.state.volume_exclusion, bool):
        raise ValueError("model.state.volume_exclusion must be a boolean")
    if not isinstance(spec.state.identity_based, bool):
        raise ValueError("model.state.identity_based must be a boolean")
    if spec.state.density is not None:
        if (
            isinstance(spec.state.density, bool)
            or np.asarray(spec.state.density).ndim != 0
            or not np.isfinite(float(spec.state.density))
            or float(spec.state.density) < 0
        ):
            raise ValueError("model.state.density must be a finite non-negative scalar")
    if not isinstance(spec.state.parameters, Mapping):
        raise ValueError("model.state.parameters must be a mapping")
    if not isinstance(spec.state.fields, Mapping):
        raise ValueError("model.state.fields must be a mapping")


_GEOMETRY_ALIASES = {
    "1d": "lin", "lin": "lin", "linear": "lin",
    "square": "square", "sq": "square", "rect": "square", "rectangular": "square",
    "hex": "hex", "hx": "hex", "hexagonal": "hex",
    "cubic": "cubic", "cb": "cubic",
    "moore": "moore", "moore3d": "moore",
}
_BOUNDARY_ALIASES = {
    "absorbing": "absorbing", "absorb": "absorbing", "abs": "absorbing",
    "abc": "absorbing", "fixed": "absorbing",
    "reflecting": "reflecting", "reflect": "reflecting", "refl": "reflecting",
    "rbc": "reflecting", "no_flux": "reflecting", "noflux": "reflecting",
    "periodic": "periodic", "pbc": "periodic", "inflow": "inflow",
}
_RESERVED_STATE_PARAMETERS = {
    "bc", "density", "dims", "geometry", "ib", "identity_based", "interaction",
    "n_species", "nodes", "propagation", "restchannels", "seed", "ve",
    "volume_exclusion",
}


def _normalize_and_validate_spec(spec: ModelSpec) -> ModelSpec:
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    geometry = spec.space.geometry
    if not isinstance(geometry, str) or geometry.lower() not in _GEOMETRY_ALIASES:
        raise ValueError("model.space.geometry must name a supported geometry")
    boundary = spec.space.boundary
    if not isinstance(boundary, str) or boundary.lower() not in _BOUNDARY_ALIASES:
        raise ValueError("model.space.boundary must name a supported boundary condition")
    _validate_dims(spec.space.dims)

    parameters = dict(spec.state.parameters)
    for name in sorted(set(parameters) & _RESERVED_STATE_PARAMETERS):
        raise ValueError(
            f"model.state.parameters.{name} is reserved by the canonical model configuration"
        )
    capacity = spec.state.capacity
    if "capacity" in parameters:
        legacy_capacity = parameters.pop("capacity")
        if capacity is not None and legacy_capacity != capacity:
            raise ValueError(
                "model.state.parameters.capacity conflicts with model.state.capacity"
            )
        capacity = legacy_capacity if capacity is None else capacity
        warnings.warn(
            "model.state.parameters.capacity is deprecated; use model.state.capacity",
            DeprecationWarning,
            stacklevel=3,
        )

    normalized = replace(
        spec,
        space=replace(
            spec.space,
            geometry=_GEOMETRY_ALIASES[geometry.lower()],
            boundary=_BOUNDARY_ALIASES[boundary.lower()],
        ),
        state=replace(spec.state, parameters=parameters, capacity=capacity),
    )
    _validate_spec(normalized)
    return normalized


def _validate_dims(dims) -> None:
    if dims is None:
        return
    values = (dims,) if np.asarray(dims).ndim == 0 else tuple(dims)
    if not values:
        raise ValueError("model.space.dims must not be empty")
    for value in values:
        _validate_positive_integer("model.space.dims", value)


def _validate_positive_integer(path, value) -> None:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{path} must be a positive integer")


def _validate_non_negative_integer(path, value) -> None:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")


def _validate_field_names(lgca, fields: Mapping[str, Any]) -> None:
    for name in fields:
        if not isinstance(name, str) or not name:
            raise ValueError("model.state.fields keys must be non-empty strings")
        if hasattr(lgca, name):
            raise ValueError(
                f"model.state.fields.{name} collides with simulator state or methods"
            )


def _build_lgca(spec: ModelSpec):
    kwargs = {
        "bc": spec.space.boundary,
        "restchannels": spec.state.restchannels,
        "interaction": "only_propagation",
    }
    if spec.space.dims is not None:
        kwargs["dims"] = spec.space.dims
    if spec.time.seed is not None:
        kwargs["seed"] = spec.time.seed
    if spec.state.nodes is not None:
        kwargs["nodes"] = spec.state.nodes
    elif spec.state.density is not None:
        kwargs["density"] = spec.state.density
    if spec.state.capacity is not None:
        kwargs["capacity"] = spec.state.capacity
    kwargs.update(dict(spec.state.parameters))
    return get_lgca(
        geometry=spec.space.geometry,
        ib=spec.state.identity_based,
        ve=spec.state.volume_exclusion,
        n_species=spec.state.n_species,
        **kwargs,
    )


def _metadata_from_spec(spec: ModelSpec, lgca=None) -> dict[str, Any]:
    geometry = getattr(lgca, "geometry", spec.space.geometry)
    boundary = getattr(lgca, "bc", spec.space.boundary)
    dims = tuple(getattr(lgca, "dims", spec.space.dims or ()))
    restchannels = getattr(lgca, "restchannels", spec.state.restchannels)
    capacity = getattr(lgca, "capacity", getattr(lgca, "K", spec.state.capacity))
    return {
        "title": spec.description.title,
        "biolgca_version": _package_version(),
        "model_spec_schema_version": MODEL_SPEC_SCHEMA_VERSION,
        "geometry": geometry,
        "dims": dims,
        "boundary": boundary,
        "restchannels": restchannels,
        "capacity": capacity,
        "steps": spec.time.steps,
        "seed": spec.time.seed,
        "volume_exclusion": spec.state.volume_exclusion,
        "identity_based": spec.state.identity_based,
        "n_species": spec.state.n_species,
        "propagation": spec.dynamics.propagation,
        "operator_names": [],
        "reorientation_term_names": [],
        "observer_names": _observer_names(spec.analysis),
        "output_paths": [],
        "runtime": {
            "elapsed_seconds": 0.0,
            "operator_timings": [],
        },
    }


def _attach_fields(lgca, fields: Mapping[str, Any]) -> None:
    for name, value in fields.items():
        array = np.asarray(value)
        spatial_ndim = len(lgca.dims)
        spatial_shape = tuple(lgca.dims)
        target_shape = lgca.nodes.shape[:spatial_ndim]
        if array.shape[:spatial_ndim] == spatial_shape and target_shape != spatial_shape:
            pad_width = [(lgca.r_int, lgca.r_int)] * spatial_ndim
            pad_width.extend([(0, 0)] * (array.ndim - spatial_ndim))
            array = np.pad(array, pad_width=pad_width, mode="edge")
        setattr(lgca, name, array)


def _observer_names(analysis: AnalysisSpec | None) -> list[str]:
    if analysis is None:
        return []
    return [observer.__class__.__name__ for observer in analysis.observers]


def _run_compiled_model(compiled: CompiledModel, showprogress: bool = True) -> ModelRunResult:
    lgca = compiled.lgca
    observers = list(compiled.spec.analysis.observers if compiled.spec.analysis is not None else [])
    runner = _PipelineRunner(compiled=compiled, observers=observers, showprogress=showprogress)
    runner.run()
    compiled.metadata["runtime"] = {
        "elapsed_seconds": runner.elapsed_seconds,
        "operator_timings": runner.operator_timings,
    }
    compiled.metadata["output_paths"] = _collect_output_paths(observers)
    return ModelRunResult(
        lgca=lgca,
        spec=compiled.spec,
        context=compiled.context,
        pipeline=compiled.pipeline,
        metadata=compiled.metadata,
    )


class _PipelineRunner:
    def __init__(self, compiled: CompiledModel, observers, showprogress: bool):
        self.compiled = compiled
        self.lgca = compiled.lgca
        self.timesteps = int(compiled.spec.time.steps)
        self.observers = list(observers)
        self.showprogress = showprogress
        self.elapsed_seconds = 0.0
        self.operator_timings: list[dict[str, Any]] = []

    def run(self):
        start = time.perf_counter()
        self.lgca.update_dynamic_fields()
        for observer in self.observers:
            setup = getattr(observer, "setup", None)
            if setup is not None:
                setup(self.lgca, self)

        self._notify_observers(0)
        for step in tqdm(range(1, self.timesteps + 1), disable=not self.showprogress):
            self.compiled.pipeline.execute_step(
                self.compiled.context,
                step,
                timing=self.operator_timings,
            )
            self._notify_observers(step)

        for observer in self.observers:
            finalize = getattr(observer, "finalize", None)
            if finalize is not None:
                finalize(self.lgca, self)
        self.elapsed_seconds = time.perf_counter() - start
        return self.lgca

    def _notify_observers(self, step: int) -> None:
        for observer in self.observers:
            schedule = getattr(observer, "schedule", None)
            if schedule is None or schedule.should_run(step):
                observer.on_step(self.lgca, step)


def _package_version() -> str:
    try:
        return importlib.metadata.version("biolgca")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _collect_output_paths(observers) -> list[str]:
    paths: list[str] = []
    for observer in observers:
        for attr in ("paths",):
            for path in getattr(observer, attr, []) or []:
                paths.append(str(path))
        for attr in ("output_path", "save_path"):
            path = getattr(observer, attr, None)
            if path is not None:
                paths.append(str(path))
    return paths
