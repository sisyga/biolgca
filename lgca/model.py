"""Declarative model specifications and execution helpers."""

from __future__ import annotations

import contextvars
import importlib.metadata
import importlib.resources
import difflib
import json
from ._warnings import warn_user
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import get_lgca
from .lattice_state import _pad_field
from .pipeline import (
    BirthDeathSpec,
    InteractionPipelineSpec,
    PhenotypeSwitchSpec,
    ReorientationSpec,
    ReorientationTermSpec,
    compile_pipeline,
)
from .simulation import DEFAULT_RECORDING_LIMIT_BYTES, RunData, SimulationRunner


MODEL_SPEC_SCHEMA_VERSION = 1
ARRAY_FILE_SUFFIX = ".arrays.npz"
DEFAULT_MAX_INLINE_ARRAY = 100  # arrays with more elements go to the array file of save_model_spec
MODEL_SPEC_INITIALIZER_NAMES = frozenset({"region", "from_npz"})

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
    "load_model_spec_schema",
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
    """Human-readable description of a model, stored with its files and results.

    Attributes
    ----------
    title : str
        Short name of the model.
    details : str, default=""
        Free text: the question, assumptions or references.
    tags : tuple of str, default=()
        Keywords for finding the model later.
    """

    title: str
    details: str = ""
    tags: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return self.details


@dataclass(frozen=True)
class SpaceSpec:
    """Lattice geometry, size and boundary condition.

    Attributes
    ----------
    geometry : str, default="hex"
        ``"lin"`` (1D, 2 velocity channels), ``"square"`` (4), ``"hex"``
        (hexagonal, 6), ``"cubic"`` (3D, 6) or ``"moore"`` (3D Moore
        neighbourhood, 26). Aliases such as ``"1d"`` or ``"hexagonal"`` are
        accepted and normalized.
    dims : int or tuple of int, optional
        Number of nodes along each axis, e.g. ``(50, 50)``. Defaults to 100
        nodes in 1D, 50 x 50 in 2D and 10 x 10 x 10 in 3D. Periodic hexagonal
        lattices need an even number of rows.
    boundary : str, default="periodic"
        ``"periodic"`` (opposite edges connected), ``"reflecting"`` (no-flux
        walls; cells bounce back) or ``"absorbing"`` (cells leaving the domain
        are lost). Aliases such as ``"pbc"`` or ``"refl"`` are accepted.
    """

    geometry: str = "hex"
    dims: Any = None
    boundary: str = "periodic"


@dataclass(frozen=True)
class StateSpec:
    """Model family, channels and initial state.

    ``volume_exclusion``, ``identity_based`` and ``n_species`` select the
    model family. At most one of ``density``, ``nodes`` and ``initializer``
    may be given; without any of them, cells are placed at random with a mean
    of 0.1 cells per node.

    Attributes
    ----------
    density : float, optional
        Mean number of cells per node, not the fraction of occupied channels.
        With volume exclusion, each channel is occupied independently with
        probability ``density / K``, where ``K`` is the number of channels per
        node (velocity plus rest channels), so ``density <= K``. Without volume
        exclusion, channel populations are Poisson distributed.
    nodes : array_like, optional
        Explicit initial channel state of shape ``dims + (K,)``, or
        ``dims + (n_species, K)`` for several species. Values are occupation
        (0/1) with volume exclusion, cell counts without, and cell labels
        (0 for empty) in identity-based models.
    restchannels : int, default=0
        Number of rest channels per node, in addition to the velocity channels
        of the geometry. Resting cells do not move; several growth and
        switching interactions need at least one rest channel. Identity-based
        models without volume exclusion always use one rest channel.
    volume_exclusion : bool, default=True
        With volume exclusion, each channel holds at most one cell. Without
        it, channels hold any number of cells, up to ``capacity`` per node for
        the interactions that use one.
    identity_based : bool, default=False
        Track individual cells with labels and heritable properties (e.g. a
        birth rate that mutates), as needed for evolutionary models.
    n_species : int, default=1
        Number of cell species (classical models only).
    capacity : int, optional
        Carrying capacity: the number of cells per node at which birth stops.
        Used by models without volume exclusion and by the ``birth_death``
        interaction.
    initializer : mapping, optional
        A named initial condition, e.g. a fully occupied square in the centre:
        ``{"name": "region", "parameters": {"extent": 5, "density": 4}}``, or
        ``{"name": "from_npz", "parameters": {"path": "state.npz"}}``.
    parameters : mapping, default={}
        Further keyword arguments for the LGCA constructor, such as
        ``r_int`` (interaction radius).
    fields : mapping, default={}
        Named arrays that interactions can use, e.g. a scalar signal of shape
        ``dims`` for ``chemotaxis`` or a director field of shape
        ``dims + (d,)`` for ``contact_guidance``; a number is the same value
        at every node. Each field becomes an attribute of the LGCA object. A
        field that a ``pde`` operator updates (:class:`lgca.fields.PDESpec`)
        changes during the run; the others stay as given.
    traits : mapping, default={}
        Initial traits of the cells of an identity-based model, e.g.
        ``{"kappa": 4.0, "theta": 0.6}``. A value is either one number for all
        initial cells or a sequence with one number per initial cell, in the
        order of their labels (cells are labelled along the lattice, node by
        node and channel by channel). Daughters inherit the traits of their
        mother; rules read them with ``state.cells["kappa"]``.
    """

    density: float | None = None
    nodes: Any = None
    restchannels: int = 0
    volume_exclusion: bool = True
    identity_based: bool = False
    n_species: int = 1
    capacity: int | None = None
    initializer: Mapping[str, Any] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)
    traits: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeSpec:
    """Number of time steps and random seed.

    Attributes
    ----------
    steps : int, default=100
        Number of time steps to simulate. Each step applies the interactions
        and then moves the cells (propagation).
    seed : int, optional
        Seed of the random number generator, a non-negative integer. The
        same seed and model give the same trajectory. Without a seed, every
        run draws a new one and records it in ``result.spec.time.seed`` and
        ``result.metadata["seed"]`` (and in ``model.resolved.json`` for
        command-line runs), so any run can be repeated.
    timing_trace : int, default=0
        Number of per-operator timing records to keep in the run metadata,
        for profiling.
    """

    steps: int = 100
    seed: int | None = None
    timing_trace: int = 0


@dataclass(frozen=True)
class AnalysisSpec:
    """What to record during a run.

    Attributes
    ----------
    observers : sequence of observers, default=()
        Recorders and plotting observers from :mod:`lgca.simulation` and
        :mod:`lgca.plotting`, e.g. ``[DensityRecorder(), PopulationRecorder()]``.
        Recorded data is stored on ``result.lgca`` (``dens_t``, ``n_t``, ...)
        and, for command-line runs, in ``measurements.npz``. Without observers
        only the final state is available.
    """

    observers: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelSpec:
    """Complete, shareable description of a simulation.

    A model specification holds data only. Run it with :func:`run_model`, save
    it with :func:`save_model_spec` and load it with :func:`load_model_spec`.
    A model built from a spec keeps its own copy of it: changing the spec
    afterwards, or building further models from it, does not change a model
    already built. Operator objects in ``dynamics`` are templates, of which
    every model runs its own copy (``model.pipeline.operators[i]``);
    observers are shared (they record the run for the caller).

    Attributes
    ----------
    description : Description
        Title, details and tags.
    space : SpaceSpec
        Lattice geometry, size and boundary condition.
    state : StateSpec
        Model family, channels and initial state.
    time : TimeSpec
        Number of steps and random seed.
    dynamics : InteractionPipelineSpec
        The interactions applied in every time step, before propagation.
    analysis : AnalysisSpec or None
        What to record during the run.

    Examples
    --------
    >>> from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
    >>> from lgca.pipeline import InteractionPipelineSpec
    >>> spec = ModelSpec(
    ...     space=SpaceSpec(geometry="square", dims=(20, 20)),
    ...     state=StateSpec(density=0.5),
    ...     time=TimeSpec(steps=50, seed=1),
    ...     dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
    ... )
    >>> result = run_model(spec, showprogress=False)
    """

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
                "restchannels": _to_jsonable(spec.state.restchannels),
                "volume_exclusion": _to_jsonable(spec.state.volume_exclusion),
                "identity_based": _to_jsonable(spec.state.identity_based),
                "n_species": _to_jsonable(spec.state.n_species),
                "capacity": _to_jsonable(spec.state.capacity),
                "initializer": _to_jsonable(spec.state.initializer),
                "parameters": _to_jsonable(dict(spec.state.parameters)),
                "fields": _to_jsonable(dict(spec.state.fields)),
                "traits": _to_jsonable(dict(spec.state.traits)),
            },
            "time": {
                "steps": _to_jsonable(spec.time.steps),
                "seed": _to_jsonable(spec.time.seed),
                "timing_trace": _to_jsonable(spec.time.timing_trace),
            },
            "dynamics": {
                "operators": [_operator_to_dict(operator) for operator in spec.dynamics.operators],
                "propagation": _to_jsonable(spec.dynamics.propagation),
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
            dims=_dims_from_json(_from_jsonable(space.get("dims"))),
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
            initializer=_from_jsonable(state.get("initializer")),
            parameters=_from_jsonable(state.get("parameters", {})),
            fields=_from_jsonable(state.get("fields", {})),
            traits=_from_jsonable(state.get("traits", {})),
        ),
        time=TimeSpec(
            steps=time_spec.get("steps", 100),
            seed=time_spec.get("seed"),
            timing_trace=time_spec.get("timing_trace", 0),
        ),
        dynamics=InteractionPipelineSpec(
            operators=tuple(_operator_from_dict(operator) for operator in dynamics.get("operators", ())),
            propagation=dynamics.get("propagation", "default"),
            # Files written before the order became free always carry the old default False.
            allow_custom_order=dynamics.get("allow_custom_order") or None,
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

    with _array_files(source):
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
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Reading YAML model specs requires PyYAML. Install biolgca[yaml] "
                "or use a .json model spec."
            ) from exc
    else:
        data = yaml.safe_load(text)
    with _array_files(source):
        return model_spec_from_dict(data)


def save_model_spec(
    spec: ModelSpec,
    path: str | Path,
    file_format: str | None = None,
    *,
    max_inline_array: int | None = DEFAULT_MAX_INLINE_ARRAY,
) -> Path:
    """Save a model specification to ``.json``, ``.yaml`` or ``.yml``.

    This is the beginner-facing persistence helper. It chooses the format from
    the file suffix unless ``file_format`` is provided.

    Arrays with more than ``max_inline_array`` elements, e.g. initial nodes or
    fields, go to an array file next to the model file, ``<name>.arrays.npz``
    (``model.json`` -> ``model.arrays.npz``); the model file refers to them by
    name and with a hash of each array, and :func:`load_model_spec` reads them
    from there and warns if an array changed since. Keep the two files
    together. ``max_inline_array=None`` writes every array into the model file.
    Arrays of Python objects (lists of labels) always stay in the model file.
    """

    target = Path(path)
    selected = _resolve_model_spec_format(target, file_format)
    if selected not in ("json", "yaml"):
        raise ValueError("Use a .json, .yaml, or .yml file for model specs.")
    array_file = target.with_name(target.stem + ARRAY_FILE_SUFFIX)
    arrays: dict[str, np.ndarray] = {}
    token = _ARRAY_SINK.set((array_file.name, max_inline_array, arrays))
    try:
        if selected == "json":
            model_spec_to_json(spec, target)
        else:
            model_spec_to_yaml(spec, target)
    finally:
        _ARRAY_SINK.reset(token)
    if arrays:
        np.savez_compressed(array_file, **arrays)
    elif array_file.exists():
        array_file.unlink()  # left from an earlier save of this model
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


def load_model_spec_schema() -> dict[str, Any]:
    """Load the packaged JSON Schema for the stable ModelSpec v1 wire format."""

    resource = importlib.resources.files("lgca.schemas").joinpath(
        "model-spec-v1.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


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
            "n_species", "capacity", "initializer", "parameters", "fields", "traits",
        },
        "model.state",
    )
    _reject_unknown_keys(time_spec, {"steps", "seed", "timing_trace"}, "model.time")
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
    for key in ("parameters", "fields", "traits"):
        if key in state and not isinstance(state[key], Mapping):
            raise TypeError(f"model.state.{key} must be a mapping")
    initializer = state.get("initializer")
    if initializer is not None:
        initializer = _mapping_at(initializer, "model.state.initializer")
        _reject_unknown_keys(
            initializer, {"name", "parameters"}, "model.state.initializer"
        )
        if not isinstance(initializer.get("name"), str):
            raise TypeError("model.state.initializer.name must be a string")
        if initializer["name"] not in MODEL_SPEC_INITIALIZER_NAMES:
            valid = ", ".join(sorted(MODEL_SPEC_INITIALIZER_NAMES))
            raise ValueError(
                f"model.state.initializer.name must be one of: {valid}"
            )
        if not isinstance(initializer.get("parameters", {}), Mapping):
            raise TypeError("model.state.initializer.parameters must be a mapping")
    for key in ("restchannels", "n_species", "capacity"):
        value = state.get(key)
        if value is not None and (isinstance(value, bool) or int(value) != value):
            raise TypeError(f"model.state.{key} must be an integer")
    for key in ("steps", "timing_trace"):
        value = time_spec.get(key)
        if value is not None and (isinstance(value, bool) or int(value) != value):
            raise TypeError(f"model.time.{key} must be an integer")
    _validate_seed(time_spec.get("seed"))
    for key in ("operators",):
        value = dynamics.get(key, ())
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"model.dynamics.{key} must be a sequence")
    for index, operator in enumerate(dynamics.get("operators", ())):
        _validate_serialized_operator(operator, index)
    if dynamics.get("allow_custom_order") is not None and not isinstance(
        dynamics["allow_custom_order"], bool
    ):
        raise TypeError("model.dynamics.allow_custom_order must be a boolean")
    if analysis is not None:
        observers = analysis.get("observers", ())
        if isinstance(observers, (str, bytes)) or not isinstance(observers, Sequence):
            raise TypeError("model.analysis.observers must be a sequence")
        for index, observer in enumerate(observers):
            _validate_serialized_observer(observer, index)


def _validate_serialized_operator(operator, index: int) -> None:
    path = f"model.dynamics.operators[{index}]"
    operator = _mapping_at(operator, path)
    if "type" in operator:
        _reject_unknown_keys(
            operator, {"type", "sampler", "parameters", "terms"}, path
        )
        if operator.get("type") != "reorientation":
            raise ValueError(f"{path}.type must be 'reorientation'")
        terms = operator.get("terms", ())
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
            raise TypeError(f"{path}.terms must be a sequence")
        for term_index, term in enumerate(terms):
            term_path = f"{path}.terms[{term_index}]"
            term = _mapping_at(term, term_path)
            _reject_unknown_keys(
                term, {"name", "beta", "parameters", "species", "trait", "sensed_species"}, term_path
            )
            if not isinstance(term.get("name"), str):
                raise TypeError(f"{term_path}.name must be a string")
            if not isinstance(term.get("parameters", {}), Mapping):
                raise TypeError(f"{term_path}.parameters must be a mapping")
        if not isinstance(operator.get("parameters", {}), Mapping):
            raise TypeError(f"{path}.parameters must be a mapping")
        return
    _reject_unknown_keys(operator, {"name", "parameters"}, path)
    if not isinstance(operator.get("name"), str):
        raise TypeError(f"{path}.name must be a string")
    if not isinstance(operator.get("parameters", {}), Mapping):
        raise TypeError(f"{path}.parameters must be a mapping")


def _validate_serialized_observer(observer, index: int) -> None:
    path = f"model.analysis.observers[{index}]"
    observer = _mapping_at(observer, path)
    observer_type = observer.get("type")
    simple = {
        "NodeRecorder", "PopulationRecorder", "ChannelDensityRecorder",
        "PerTypeRecorder", "OrderParameterRecorder", "FamilyPopulationRecorder",
    }
    if observer_type in simple:
        allowed = {"type", "schedule"}
    elif observer_type == "DensityRecorder":
        allowed = {"type", "schedule", "dtype"}
    elif observer_type == "CSVSnapshotObserver":
        allowed = {"type", "schedule", "kind", "output_dir", "filename"}
    elif observer_type == "ScalarTimeSeriesRecorder":
        allowed = {"type", "schedule", "output_path"}
    elif observer_type == "FieldRecorder":
        allowed = {"type", "schedule", "fields"}
        fields = observer.get("fields")
        if (isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence) or not fields
                or not all(isinstance(name, str) for name in fields)):
            raise TypeError(f"{path}.fields must be a list of field names")
    else:
        raise ValueError(f"{path}.type unknown observer {observer_type!r}")
    _reject_unknown_keys(observer, allowed, path)
    schedule = observer.get("schedule")
    if schedule is not None:
        schedule = _mapping_at(schedule, f"{path}.schedule")
        _reject_unknown_keys(schedule, {"every", "steps"}, f"{path}.schedule")


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
        if name == "pde":  # a field operator writes its field, not the cells
            output = _operator_parameters(operator).get("field")
            add_node(f"field:{output}", "field", output)
            edges.append({"source": operator_id, "target": f"field:{output}"})
            continue
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


# While save_model_spec writes: (name of the array file, largest inline array, arrays for the file)
_ARRAY_SINK: contextvars.ContextVar = contextvars.ContextVar("lgca_array_sink", default=None)
# While a model file is read: its directory and the array files read from it
_ARRAY_SOURCE: contextvars.ContextVar = contextvars.ContextVar("lgca_array_source", default=None)


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        sink = _ARRAY_SINK.get()
        if sink is not None and value.dtype != object and sink[1] is not None and value.size > sink[1]:
            from .provenance import array_hash

            filename, _, arrays = sink
            name = f"array_{len(arrays)}"
            arrays[name] = value
            # the hash tells, when the model file is read, whether another model saved next to it (e.g. a
            # .yaml beside the .json) has since written its own arrays to the shared file
            return {"__ndarray_file__": filename, "name": name, "dtype": str(value.dtype),
                    "shape": list(value.shape), "sha256": array_hash(value)}
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):  # tuples become lists
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _from_jsonable(value):
    if isinstance(value, Mapping):
        if "__ndarray__" in value:
            dtype = np.dtype(value.get("dtype")) if value.get("dtype") is not None else None
            shape = value.get("shape")
            if dtype == np.dtype(object) and shape is not None:
                # A channel's list of cell labels is one object, even when every
                # channel has the same number of labels (including zero).
                array = np.empty(shape, dtype=object)

                def restore(items, index=()):
                    if len(index) == array.ndim:
                        array[index] = items
                    else:
                        if len(items) != array.shape[len(index)]:
                            raise ValueError("inline array data does not match its shape")
                        for position, item in enumerate(items):
                            restore(item, index + (position,))

                restore(value["__ndarray__"])
                return array
            array = np.asarray(value["__ndarray__"], dtype=dtype)
            return array if shape is None else array.reshape(shape)
        if "__ndarray_file__" in value:
            return _array_from_file(value)
        if "__tuple__" in value:  # files written before tuples became lists
            return tuple(_from_jsonable(item) for item in value["__tuple__"])
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _dims_from_json(dims):
    return tuple(dims) if isinstance(dims, list) else dims


class _array_files:
    """Read the array files of the model file ``source`` (if it is a path) while building its spec."""

    def __init__(self, source):
        path = _source_path(source) if not isinstance(source, Path) else source
        self.base = None if path is None else path.resolve().parent

    def __enter__(self):
        self.token = _ARRAY_SOURCE.set((self.base, {}))

    def __exit__(self, *exc):
        _, opened = _ARRAY_SOURCE.get()
        for archive in opened.values():
            archive.close()
        _ARRAY_SOURCE.reset(self.token)


def _array_from_file(reference):
    filename, name = reference.get("__ndarray_file__"), reference.get("name")
    if (not isinstance(filename, str) or not isinstance(name, str) or Path(filename).name != filename
            or "\\" in filename or not filename.endswith(".npz")):
        raise ValueError(f"an array file must be a .npz file next to the model file, got {filename!r}")
    source = _ARRAY_SOURCE.get()
    if source is None or source[0] is None:
        raise ValueError(f"the model refers to arrays in {filename}; load it from its file with "
                         f"load_model_spec(path), with {filename} next to it")
    base, opened = source
    if filename not in opened:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"the model refers to arrays in {filename}, which is missing next to "
                                    f"the model file ({path}); keep the two files together")
        opened[filename] = np.load(path, allow_pickle=False)
    archive = opened[filename]
    if name not in archive.files:
        raise ValueError(f"{filename} has no array {name!r}")
    array = archive[name]
    if list(array.shape) != list(reference.get("shape", array.shape)) or (
            "dtype" in reference and str(array.dtype) != reference["dtype"]):
        raise ValueError(f"the array {name!r} in {filename} does not match the model file "
                         f"(shape {list(array.shape)}, dtype {array.dtype})")
    expected = reference.get("sha256")  # written since 2026-10; older model files have none
    if isinstance(expected, str):
        from .provenance import array_hash

        found = array_hash(array)
        if found != expected:
            warn_user(f"the array {name!r} in {filename} changed after the model file was saved (sha256 "
                      f"{found[:16]}..., the model file expects {expected[:16]}...): another model saved with "
                      f"the same name, e.g. a .yaml next to a .json, may have overwritten it")
    return array


def _operator_to_dict(operator) -> dict[str, Any]:
    if isinstance(operator, (BirthDeathSpec, PhenotypeSwitchSpec)):
        return _registered_operator_to_dict(operator.name, operator.parameters)
    if isinstance(operator, ReorientationSpec):
        return {
            "type": "reorientation",
            "sampler": operator.sampler,
            "parameters": _to_jsonable(dict(operator.parameters)),
            "terms": [_reorientation_term_to_dict(term) for term in operator.terms],
        }
    if isinstance(operator, Mapping):
        if "name" not in operator:
            raise TypeError(
                "ModelSpec is not portable: operator mappings require a registered plugin name."
            )
        unexpected = set(operator) - {"name", "parameters"}
        if unexpected:
            raise TypeError(
                "ModelSpec is not portable: registered operator mappings support only "
                f"'name' and 'parameters', got {sorted(unexpected)}."
            )
        return _registered_operator_to_dict(operator["name"], operator.get("parameters", {}))
    if isinstance(operator, ReorientationTermSpec):  # saved by name, it would lose beta, species and trait
        raise TypeError(
            f"ModelSpec is not portable: the reorientation term {operator.name!r} is given as an operator; "
            "put it in ReorientationSpec(terms=[...]), or give one cue by name and parameters."
        )
    name = getattr(operator, "name", None)
    parameters = getattr(operator, "parameters", {})
    if name is not None:
        return _registered_operator_to_dict(name, parameters)
    raise TypeError(
        "ModelSpec is not portable: operators must reference a registered plugin by name."
    )


def _registered_operator_to_dict(name, parameters) -> dict[str, Any]:
    from .plugins import describe_plugin

    try:
        info = describe_plugin(str(name))
    except KeyError as exc:
        raise TypeError(
            f"ModelSpec is not portable: operator {name!r} is not a registered plugin. "
            "Import and register it in a trusted Python launcher before saving."
        ) from exc
    return {
        "name": info.name,
        "parameters": _to_jsonable(dict(parameters or {})),
    }


def _operator_from_dict(data: Mapping[str, Any]):
    operator_type = data.get("type")
    if operator_type == "reorientation":
        return ReorientationSpec(
            terms=tuple(_reorientation_term_from_dict(term) for term in data.get("terms", ())),
            sampler=data.get("sampler", "boltzmann"),
            parameters=_from_jsonable(data.get("parameters", {})),
        )
    return {
        "name": data["name"],
        "parameters": _from_jsonable(data.get("parameters", {})),
    }


def _reorientation_term_to_dict(term: ReorientationTermSpec) -> dict[str, Any]:
    return {
        "name": term.name,
        "beta": _to_jsonable(term.beta),
        "parameters": _to_jsonable(dict(term.parameters)),
        "species": _to_jsonable(term.species),
        "trait": _to_jsonable(term.trait),
        "sensed_species": np.asarray(term.sensed_species).tolist(),
    }


def _reorientation_term_from_dict(data: Mapping[str, Any]) -> ReorientationTermSpec:
    return ReorientationTermSpec(
        name=data["name"],
        beta=data.get("beta", 1.0),
        parameters=_from_jsonable(data.get("parameters", {})),
        species=data.get("species"),
        trait=data.get("trait"),
        sensed_species=_from_jsonable(data.get("sensed_species")),
    )


def _observer_to_dict(observer) -> dict[str, Any]:
    supported = {
        "NodeRecorder", "PopulationRecorder", "DensityRecorder",
        "ChannelDensityRecorder", "PerTypeRecorder", "OrderParameterRecorder",
        "FamilyPopulationRecorder", "CSVSnapshotObserver", "ScalarTimeSeriesRecorder", "FieldRecorder",
    }
    observer_type = observer.__class__.__name__
    if observer_type not in supported:
        raise TypeError(
            f"ModelSpec is not portable: {observer_type} is not a built-in observer. "
            "Custom observers must be attached by a trusted Python launcher."
        )
    data = {
        "type": observer_type,
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
        from .simulation import _total_population

        if set(observer.metrics) != {"population"} or observer.metrics["population"] is not _total_population:
            raise TypeError("ScalarTimeSeriesRecorder serialization only supports the default population metric; "
                            "attach recorders of your own metrics in Python, after loading the model")
        if observer.output_path is not None:
            data["output_path"] = str(observer.output_path)
    elif observer_type == "FieldRecorder":
        data["fields"] = list(observer.fields)
    return data


def _observer_from_dict(data: Mapping[str, Any]):
    from .simulation import (
        CSVSnapshotObserver,
        ChannelDensityRecorder,
        DensityRecorder,
        FamilyPopulationRecorder,
        FieldRecorder,
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
    if observer_type == "FieldRecorder":
        return FieldRecorder(data["fields"], schedule=schedule)
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
    if text.lstrip().startswith(("{", "[")) or "\n" in text or "\r" in text:
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
        name = str(operator.get("name", "<missing>"))
    else:
        name = str(getattr(operator, "name", "<missing>"))
    parameters = _operator_parameters(operator)
    _cue_fields(parameters, dependencies)
    if name == "pde":
        production = parameters.get("production")
        if isinstance(production, str):
            dependencies.add(production)
        return name, dependencies
    field = parameters.get("field")
    if field is not None:
        dependencies.add(str(field))
    return name, dependencies


def _cue_fields(value, fields: set[str]) -> None:
    """Add the fields read by the cues (``{"name": "field", "field": ...}``) nested in ``value``."""
    if isinstance(value, Mapping):
        if value.get("name") in ("field", "gradient") and isinstance(value.get("field"), str):
            fields.add(value["field"])
        for item in value.values():
            _cue_fields(item, fields)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _cue_fields(item, fields)


def _operator_parameters(operator) -> Mapping[str, Any]:
    parameters = operator.get("parameters") if isinstance(operator, Mapping) else getattr(operator, "parameters", {})
    return parameters if isinstance(parameters, Mapping) else {}


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
        "FieldRecorder": "fields",
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
    """Built LGCA model plus compiled interaction pipeline.

    ``spec`` is the model's own copy of the specification it runs (see
    :func:`build_model`), and ``pipeline.operators`` are the operators that
    run, also the model's own copies of operator objects in the spec.
    :meth:`reconfigure` changes the dynamics of the running model;
    :attr:`initial_spec` is then the spec it was built from.
    """

    lgca: Any
    spec: ModelSpec
    context: ModelContext
    pipeline: Any
    metadata: dict[str, Any]
    _step: int = 0
    rollback: bool = True
    _failed: str | None = field(default=None, repr=False)
    _transaction: Any = field(default=None, repr=False, compare=False)
    _initial_spec: ModelSpec | None = field(default=None, repr=False, compare=False)

    @property
    def initial_spec(self) -> ModelSpec:
        """The spec the model was built from; :attr:`spec` until :meth:`reconfigure` changes it."""
        return self.spec if self._initial_spec is None else self._initial_spec

    def reconfigure(self, changes: Mapping[str, Any]) -> None:
        """Change the dynamics of the running model from its next step on, e.g. a rate.

        The cells and fields stay as they are; the operators are compiled
        anew from the changed spec, as :func:`build_model` does (a steady
        field is solved again). The change is tried first: the new pipeline
        runs one step on a copy of the model. If that fails, the error is
        raised and the model is unchanged.

        Parameters
        ----------
        changes : mapping
            Path -> new value, as for :func:`lgca.study.vary`, e.g.
            ``{"birth_rate": 0.3}`` or
            ``{"dynamics.operators[0].parameters.birth_rate": 0.3}``. Only
            the dynamics can change; space, state and time are set when the
            model is built.

        Notes
        -----
        ``metadata["reconfigurations"]`` records every change, as
        ``{"from_step": n, "changes": {path: {"old": ..., "new": ...}}}``,
        and the metadata describe the new operators; results of earlier runs
        keep theirs. Building :attr:`initial_spec`, running ``n - 1`` steps
        and reconfiguring with the new values repeats the run.

        Examples
        --------
        >>> model = build_model(spec)                                  # doctest: +SKIP
        >>> first = model.run(showprogress=False)                      # doctest: +SKIP
        >>> model.reconfigure({"birth_rate": 0.3})                     # doctest: +SKIP
        >>> second = model.run(showprogress=False)                     # doctest: +SKIP
        """
        from .study import _get, _tokens, resolve_path, vary

        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("changes must map paths to new values, e.g. {'birth_rate': 0.3}")
        paths = {resolve_path(self.spec, path): value for path, value in changes.items()}
        for path in paths:
            if not path.startswith("dynamics."):
                raise ValueError(f"{path!r} is not part of the dynamics: space, state and time are set when "
                                 "the model is built; build a new model, e.g. from "
                                 "vary(model.initial_spec, changes)")
        paths = {path: deepcopy(value) for path, value in paths.items()}  # the model's own, like its spec
        old = {path: deepcopy(_get(self.spec, _tokens(path))) for path in paths}
        spec = _normalize_and_validate_spec(vary(self.spec, paths))
        _try_step(self, spec)
        metadata = deepcopy(self.metadata)
        metadata["reconfigurations"] = [*metadata.get("reconfigurations", []), {
            "from_step": self._step + 1,
            "changes": {path: {"old": old[path], "new": deepcopy(value)} for path, value in paths.items()}}]
        context = replace(self.context, spec=spec, metadata=metadata)  # earlier results keep their context
        pipeline = _compile_running(context)
        _pipeline_metadata(metadata, spec, pipeline, self.lgca)
        if self._initial_spec is None:
            self._initial_spec = self.spec
        self.context, self.pipeline, self.spec, self.metadata = context, pipeline, spec, metadata
        self.lgca.enable_propagation = spec.dynamics.propagation not in (False, None, "none", "disabled")

    def step(self, **timing):
        """Advance the compiled dynamics once, retaining RNG and model time.

        A step is applied as a whole or not at all: if it raises (also when
        interrupted), the model is put back in the state before the step,
        with the same random numbers to come, and the exception says so in
        a note. The model can then step again, e.g. after a parameter was
        corrected. For this the model keeps a checkpoint, one copy of its
        ``nodes`` and fields; ``model.rollback = False`` saves the memory.
        A step that fails then may have been applied in part, and the model
        refuses further steps until it is rebuilt, e.g. with
        ``build_model(model.spec)``.

        Not rolled back: what operators keep themselves, e.g. the statistics
        and multigrid hierarchies of field solvers, and writes through
        ``TraitArray.values``.
        """
        if self._failed is not None:
            raise RuntimeError(f"step {self._step + 1} of this model failed ({self._failed}) and may have "
                               "been applied in part; rebuild the model to go on, e.g. with "
                               "build_model(model.spec) (Reset in lgca.explore)")
        transaction = self._begin() if self.rollback else None
        try:
            self.pipeline.execute_step(self.context, self._step + 1, **timing)
        except BaseException as exc:
            self._undo(exc, transaction)
            raise
        if transaction is not None:
            transaction.commit()
        self._step += 1

    def _begin(self):
        from .transaction import StepTransaction

        transaction = self._transaction
        if transaction is None or transaction.lgca is not self.lgca:
            transaction = self._transaction = StepTransaction(self.lgca, fields=self.context.fields)
        transaction.begin()
        return transaction

    def _undo(self, exc, transaction):
        """Roll the failed step back, or refuse further steps if it cannot be."""
        step = self._step + 1
        if transaction is not None:
            try:
                transaction.rollback()
            except Exception as error:  # noqa: BLE001 - the step's own error is the one raised
                exc.add_note(f"step {step} could not be rolled back ({type(error).__name__}: {error})")
            else:
                exc.add_note(f"step {step} was rolled back: the model is in the state before it")
                return
        self._failed = f"{type(exc).__name__}: {exc}"

    def run(self, showprogress: bool = True, *, max_recording_bytes=DEFAULT_RECORDING_LIMIT_BYTES):
        """Run with an explicit fixed-buffer recording budget (None disables it)."""
        return _run_compiled_model(self, showprogress=showprogress,
                                   max_recording_bytes=max_recording_bytes)


@dataclass
class ModelRunResult:
    """Result returned by :func:`run_model`.

    Attributes
    ----------
    lgca : LGCA object
        The model in its final state.
    spec : ModelSpec
        The specification that ran, with the seed that was used: the model's
        own copy (see :func:`build_model`), whose operator objects are
        templates that did not run.
    context : ModelContext
        The lattice, specification, fields and metadata the operators saw.
    pipeline : CompiledPipeline
        The operators that ran, ``pipeline.operators[i]``: for an operator
        object in the spec, the model's own copy, which holds what the
        operator stored while it ran.
    data : RunData
        The recorded data by name, e.g. ``result.data["density"]``, with
        ``result.data.steps("density")`` (see :class:`~lgca.simulation.RunData`).
    metadata : dict
        Versions, seed, schedule, timings and output paths of the run.
    """

    lgca: Any
    spec: ModelSpec
    context: ModelContext
    pipeline: Any
    metadata: dict[str, Any]
    data: Any = field(default_factory=lambda: RunData())


def build_model(
    spec: ModelSpec,
    *,
    resource_base: str | Path | None = None,
    trusted_paths: bool = False,
) -> CompiledModel:
    """Build an LGCA instance and compile its interaction pipeline.

    The model is built from its own copy of ``spec``, which it keeps as
    ``model.spec``: later changes to the dicts, lists and arrays of ``spec``
    (operator parameters, initial nodes, traits, the operator list) reach
    neither the model nor ``model.spec``, and the model does not change
    ``model.spec`` as it runs. This copy stays in memory as long as the model:
    initial nodes, traits and fields given as arrays are held twice, by
    ``model.spec`` and by the lattice.

    Operator objects in ``spec.dynamics.operators`` are templates: the model
    runs its own copy of each, ``model.pipeline.operators[i]``, so several
    models built from one spec do not affect each other. Everything an operator
    object holds is copied with it, e.g. a list it records into or the object
    of a bound method; operator objects that refer to each other refer to each
    other's copies. Not copied are the observers in ``spec.analysis``, which
    record the run for the caller (an operator may refer to them), and rules
    and functions.

    Without ``spec.time.seed``, a seed is drawn from the operating system's
    entropy and stored in the returned model's ``spec`` and ``metadata``, so the
    run can be repeated.
    """

    return _build_owned_model(_owned_spec(_normalize_and_validate_spec(spec)), resource_base=resource_base,
                              trusted_paths=trusted_paths)


def _build_owned_model(spec: ModelSpec, *, resource_base=None, trusted_paths=False) -> CompiledModel:
    """:func:`build_model` from a normalized spec that the model may keep as it is, e.g. one that
    :func:`_owned_spec` made (the Explorer's own spec): it is not copied again."""
    seed_drawn = spec.time.seed is None
    if seed_drawn:
        spec = replace(spec, time=replace(spec.time, seed=_draw_seed()))
    from .provenance import environment

    lgca = _build_lgca(spec)
    inputs = {}  # file as given -> hash of what was read
    if spec.state.initializer is not None:
        from .initializers import apply_initializer

        inputs = apply_initializer(
            lgca,
            spec.state.initializer,
            resource_base=resource_base,
            trusted_paths=trusted_paths,
        )
    _validate_field_names(lgca, spec.state.fields)
    metadata = _metadata_from_spec(spec, lgca=lgca)
    metadata["seed_drawn"] = seed_drawn
    metadata["provenance"] = {**environment(), "inputs": inputs}
    context = ModelContext(
        lgca=lgca,
        spec=spec,
        fields=dict(spec.state.fields),
        metadata=metadata,
    )
    _attach_fields(lgca, context.fields)
    traits = _attach_traits(lgca, spec.state.traits)
    pipeline = compile_pipeline(spec.dynamics, context)
    _attach_fields(lgca, context.fields)
    _attach_field_operators(lgca, pipeline)
    for name, values in traits.items():
        if lgca.props.get(name) is not values:
            warn_user(f"an interaction set the cell trait {name!r} when the model was built, so the "
                      f"values from state.traits are not used; set it in one place only")
    _pipeline_metadata(metadata, spec, pipeline, lgca)
    compiled = CompiledModel(
        lgca=lgca,
        spec=spec,
        context=context,
        pipeline=pipeline,
        metadata=metadata,
    )
    lgca._compiled_model = compiled
    lgca.enable_propagation = spec.dynamics.propagation not in (False, None, "none", "disabled")
    return compiled


def _pipeline_metadata(metadata, spec, pipeline, lgca) -> None:
    """Describe the operators of ``pipeline`` in ``metadata`` (at build, and when the model is reconfigured)."""
    metadata["operator_names"] = pipeline.operator_names
    metadata["reorientation_term_names"] = pipeline.reorientation_term_names
    metadata["observer_names"] = _observer_names(spec.analysis)
    metadata["schedule"] = pipeline.describe_schedule()
    metadata["propagation"] = spec.dynamics.propagation
    metadata["channel_capacity"] = lgca.K
    growth_capacities = [
        {"operator_index": index, "name": operator.name, "capacity": operator.capacity}
        for index, operator in enumerate(pipeline.operators)
        if operator.name == "birth_death"
    ]
    metadata["growth_capacities"] = growth_capacities
    metadata["capacity"] = _metadata_from_spec(spec, lgca)["capacity"]
    if len(growth_capacities) == 1:
        metadata["capacity"] = growth_capacities[0]["capacity"]
    from .provenance import pipeline_record

    # the operators with their parameters, defaults included, and the source hashes of their rules
    metadata.setdefault("provenance", {}).update(pipeline_record(pipeline))


def _compile_running(context):
    """The pipeline of ``context.spec`` for a lattice that has run; a steady field is solved again."""
    pipeline = compile_pipeline(context.spec.dynamics, context)
    _attach_field_operators(context.lgca, pipeline)
    return pipeline


def _attach_field_operators(lgca, pipeline):
    """Let the field operators (``pde``) own their fields: their boundary conditions fill the ghost nodes,
    also where rules write the fields; a field without one takes the lattice's."""
    owned = getattr(lgca, "_field_sides", {})
    lgca._field_sides = {}
    for operator in pipeline.operators:
        attach = getattr(operator, "attach_field", None)
        if attach is not None:
            attach(lgca)
    for name in set(owned) - set(lgca._field_sides):  # no longer owned: padded by the lattice
        values = np.asarray(getattr(lgca, name))
        interior = values[lgca.nonborder] if values.shape[:len(lgca.dims)] != tuple(lgca.dims) else values
        setattr(lgca, name, _pad_field(lgca, name, interior))


def _try_step(compiled, spec):
    """Run one step of a copy of the model with ``spec``, so that errors show before a change is made.

    Some values are checked only when the model steps, e.g. a rate that must be a probability.
    """
    try:  # the copy leaves out the running model, whose solvers may not be copied
        lattice = deepcopy(compiled.lgca, {id(compiled): None})
        metadata = deepcopy(compiled.metadata)
    except TypeError as exc:
        warn_user(f"the change could not be tried on a copy of the model ({exc}); it is applied unchecked")
        return
    context = ModelContext(lgca=lattice, spec=spec, fields=dict(compiled.context.fields), metadata=metadata)
    _compile_running(context).execute_step(context, compiled._step + 1)


def _draw_seed() -> int:
    """Draw a fresh seed for a run whose specification sets none."""
    return int(np.random.SeedSequence().generate_state(1, np.uint32)[0])


def run_model(
    spec: ModelSpec,
    showprogress: bool = True,
    *,
    resource_base: str | Path | None = None,
    trusted_paths: bool = False,
) -> ModelRunResult:
    """Build and run a declarative LGCA model.

    The model runs its own copy of ``spec`` (see :func:`build_model`); the
    result's ``spec`` and ``pipeline`` are that copy and the operators that ran.
    """

    compiled = build_model(
        spec, resource_base=resource_base, trusted_paths=trusted_paths
    )
    return compiled.run(showprogress=showprogress)


def _validate_spec(spec: ModelSpec) -> None:
    if isinstance(spec.time.steps, bool) or int(spec.time.steps) != spec.time.steps:
        raise ValueError("model.time.steps must be a non-negative integer")
    if spec.time.steps < 0:
        raise ValueError("model.time.steps must be a non-negative integer")
    _validate_seed(spec.time.seed)
    _validate_non_negative_integer("model.time.timing_trace", spec.time.timing_trace)
    initial_states = [
        spec.state.nodes is not None,
        spec.state.density is not None,
        spec.state.initializer is not None,
    ]
    if sum(initial_states) > 1:
        raise ValueError(
            "model.state.nodes, model.state.density, and model.state.initializer "
            "are mutually exclusive"
        )
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
    if not isinstance(spec.state.traits, Mapping):
        raise ValueError("model.state.traits must be a mapping")
    if spec.state.traits and not spec.state.identity_based:
        raise ValueError("model.state.traits needs an identity-based model (state.identity_based=True); "
                         "classical cells have no individual traits")
    for name in spec.state.traits:
        if not isinstance(name, str) or not name:
            raise ValueError("model.state.traits keys must be non-empty strings")


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
    dims = spec.space.dims
    if dims is not None and np.asarray(dims).ndim != 0:
        dims = tuple(int(value) for value in dims)
        dimension = {"lin": 1, "square": 2, "hex": 2, "cubic": 3, "moore": 3}[_GEOMETRY_ALIASES[geometry.lower()]]
        if len(dims) != dimension:
            raise ValueError(f"model.space.dims must contain {dimension} dimensions for {geometry}")

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
        warn_user("model.state.parameters.capacity is deprecated; use model.state.capacity", DeprecationWarning)

    normalized = replace(
        spec,
        space=replace(
            spec.space,
            geometry=_GEOMETRY_ALIASES[geometry.lower()],
            dims=dims,
            boundary=_BOUNDARY_ALIASES[boundary.lower()],
        ),
        state=replace(spec.state, parameters=parameters, capacity=capacity),
    )
    _validate_spec(normalized)
    return normalized


def _owned_spec(spec: ModelSpec) -> ModelSpec:
    """A copy of ``spec`` that belongs to the model built from it.

    Later changes the caller makes to the dicts, lists and arrays of ``spec`` reach neither the model
    nor ``model.spec``. Operator objects are copied as templates (the pipeline copies them again for
    the model to run); observers stay the caller's objects, which record the run for the caller. The
    spec is copied as a whole, so objects it holds twice (e.g. an operator object that refers to another
    one of the list) are one object in the copy too.
    """
    from .pipeline import _copy_templates

    shared = _shared_objects(spec)
    memo = dict(shared)
    nodes = spec.state.nodes
    if isinstance(nodes, np.ndarray) and nodes.dtype == object:  # lists of labels
        copied = _copied_object_array(nodes)
        if copied is not None:
            memo[id(nodes)] = copied
    try:
        return _copy_templates(spec, memo)
    except Exception as exc:  # noqa: BLE001 - any failure of the copy, e.g. a lock or a generator: named
        raise _copy_error(spec, "model", shared, exc) from None


def _shared_objects(spec: ModelSpec) -> dict[int, Any]:
    """The objects of ``spec`` that its copies share (a ``deepcopy`` memo): the observers."""
    observers = () if spec.analysis is None else spec.analysis.observers
    return {id(observer): observer for observer in observers}


_COPY_LISTS = np.frompyfunc(list.copy, 1, 1)


def _copied_object_array(values: np.ndarray):
    """A copy of an object array of lists of labels with lists of its own, 3 times faster than pickling and
    unpickling it; None if the array holds anything but lists (``deepcopy`` then copies it).

    The labels are numbers, which need no copy: an identity-based lattice takes only lists of integers.
    """
    copied = np.empty(values.shape, dtype=object)
    try:
        _COPY_LISTS(values, out=copied)
    except TypeError:  # "descriptor 'copy' for 'list' objects doesn't apply to a 'tuple' object"
        return None
    return copied


def _copy_error(value, path: str, memo, exc: BaseException) -> ValueError:
    """The error for ``value`` that cannot be copied; it names the innermost part that cannot, e.g.
    ``model.state.parameters.lock``, or the operator object."""
    from .operator_base import InteractionOperator
    from .pipeline import _copy_failure, _template_copy

    where, part = _uncopyable_part(value, path, memo)
    if isinstance(part, InteractionOperator):
        try:
            _template_copy(part, dict(memo))
        except ValueError as error:
            return ValueError(f"{where} {error}")
    return ValueError(f"{where} cannot be copied ({_copy_failure(exc)}); every model keeps its own copy of its "
                      "specification. Give numbers, strings, lists, dicts and arrays, or objects that can be "
                      "copied")


def _uncopyable_part(value, path: str, memo) -> tuple[str, Any]:
    """The path and the innermost part of ``value`` that cannot be copied, e.g. ``model.state.parameters.lock``;
    an operator object counts as one part."""
    from .operator_base import InteractionOperator

    if isinstance(value, InteractionOperator):
        return path, value
    if is_dataclass(value) and not isinstance(value, type):
        parts = [(f"{path}.{item.name}", getattr(value, item.name)) for item in dataclass_fields(value)]
    elif isinstance(value, Mapping):
        parts = [(f"{path}.{key}", item) for key, item in value.items()]
    elif isinstance(value, (list, tuple)):
        parts = [(f"{path}[{index}]", item) for index, item in enumerate(value)]
    else:
        return path, value
    for part_path, part in parts:
        try:
            deepcopy(part, dict(memo))
        except Exception:  # noqa: BLE001 - any failure of the copy
            return _uncopyable_part(part, part_path, memo)
    return path, value


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


def _validate_seed(seed) -> None:
    # NumPy takes only integers as seeds: 2.0 is rejected here, not when the run starts
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError(f"model.time.seed must be a non-negative integer or null, got {seed!r}")


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
        try:
            nodes = np.asarray(spec.state.nodes)
        except (TypeError, ValueError) as exc:
            raise ValueError("model.state.nodes must be a rectangular state array") from exc
        if nodes.dtype == object:  # lists of labels: the lattice's own, so that editing them leaves model.spec
            copied = _copied_object_array(nodes)
            nodes = nodes if copied is None else copied
        kwargs["nodes"] = nodes
    elif spec.state.density is not None:
        kwargs["density"] = spec.state.density
    elif spec.state.initializer is not None:
        kwargs["density"] = 0.0
    if spec.state.capacity is not None and not spec.state.volume_exclusion:
        kwargs["capacity"] = spec.state.capacity
    kwargs.update(dict(spec.state.parameters))
    lgca = get_lgca(
        geometry=spec.space.geometry,
        ib=spec.state.identity_based,
        ve=spec.state.volume_exclusion,
        n_species=spec.state.n_species,
        **kwargs,
    )
    # every LatticeState of the model reads it, e.g. of rules, reorientation terms and field reactions;
    # lgca.capacity of volume-exclusion models stays the channel count (the colour scale of plots)
    lgca._state_capacity = spec.state.capacity
    return lgca


def _metadata_from_spec(spec: ModelSpec, lgca=None) -> dict[str, Any]:
    geometry = getattr(lgca, "geometry", spec.space.geometry)
    boundary = getattr(lgca, "bc", spec.space.boundary)
    dims = tuple(getattr(lgca, "dims", spec.space.dims or ()))
    restchannels = getattr(lgca, "restchannels", spec.state.restchannels)
    capacity = spec.state.capacity  # the crowding scale of all rules (see LatticeState)
    if capacity is None and lgca is not None:
        capacity = spec.state.n_species * lgca.K if spec.state.volume_exclusion else lgca.capacity
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
            "timing_trace": [],
        },
    }


def _attach_traits(lgca, traits: Mapping[str, Any]) -> dict[str, Any]:
    """Give the initial cells their traits, one value per label (``lgca.props``)."""
    from .cells import TraitArray

    if not traits:
        return {}
    interior = lgca.nodes[lgca.nonborder]
    if interior.dtype == object:
        labels = np.fromiter((label for channel in interior.flat for label in channel), dtype=np.int64)
    else:
        labels = interior[interior > 0].astype(np.int64)
    labels = np.sort(labels)
    rows = int(lgca.maxlabel) + 1
    attached = {}
    for name, value in traits.items():
        values = np.asarray(value)
        if values.dtype == object or not np.issubdtype(values.dtype, np.number) and values.dtype != bool:
            raise ValueError(f"model.state.traits.{name} must contain numbers")
        if values.ndim == 0:
            column = np.full(rows, values)
        elif values.ndim == 1:
            if len(values) != len(labels):
                raise ValueError(f"model.state.traits.{name} has {len(values)} values, but there are "
                                 f"{len(labels)} initial cells; give one value per cell or one for all")
            # rows of labels without a cell (e.g. label 0 with volume exclusion) are never read
            column = np.full(rows, values[0] if len(values) else 0, dtype=values.dtype)
            column[labels] = values
        else:
            raise ValueError(f"model.state.traits.{name} must be a number or a sequence of numbers")
        if not np.all(np.isfinite(column.astype(float))):
            raise ValueError(f"model.state.traits.{name} must contain finite numbers")
        lgca.props[name] = attached[name] = TraitArray(column)
    return attached


def _attach_fields(lgca, fields: Mapping[str, Any]) -> None:
    for name, value in fields.items():
        array = np.array(value)  # a copy, also in the padded shape: the caller's later writes do not reach it
        if array.ndim == 0:  # a number: the same value at every node
            array = np.full(tuple(lgca.dims), float(array))
        spatial_ndim = len(lgca.dims)
        spatial_shape = tuple(lgca.dims)
        target_shape = lgca.nodes.shape[:spatial_ndim]
        if array.shape[:spatial_ndim] == spatial_shape and target_shape != spatial_shape:
            array = _pad_field(lgca, name, array)  # a pde that owns the field pads it again by its condition
        setattr(lgca, name, array)


def _observer_names(analysis: AnalysisSpec | None) -> list[str]:
    if analysis is None:
        return []
    return [observer.__class__.__name__ for observer in analysis.observers]


def _run_compiled_model(compiled: CompiledModel, showprogress: bool = True,
                        *, max_recording_bytes=DEFAULT_RECORDING_LIMIT_BYTES) -> ModelRunResult:
    lgca = compiled.lgca
    observers = list(compiled.spec.analysis.observers if compiled.spec.analysis is not None else [])
    operator_timings: dict[tuple[str, str], dict[str, Any]] = {}
    timing_trace: list[dict[str, Any]] = []
    timing_trace_limit = int(compiled.spec.time.timing_trace)

    def execute_pipeline_step(_lgca, step, _runner):
        compiled.step(
            timing=operator_timings,
            timing_trace=timing_trace,
            timing_trace_limit=timing_trace_limit,
        )

    runner = SimulationRunner(
        lgca,
        timesteps=int(compiled.spec.time.steps),
        observers=observers,
        showprogress=showprogress,
        step_function=execute_pipeline_step,
        context=compiled,
        max_recording_bytes=max_recording_bytes,
    )
    def record_runtime():
        compiled.metadata["runtime"] = {
            "start_step": runner.start_step,
            "end_step": runner.end_step,
            "sample_time_origin": "local",
            "estimated_recording_bytes": runner.estimated_recording_bytes,
            "elapsed_seconds": runner.elapsed_seconds,
            "operator_timings": list(operator_timings.values()),
            "timing_trace": timing_trace,
        }
        compiled.metadata["output_paths"] = _collect_output_paths(observers)

    try:
        runner.run()
    except BaseException:
        if getattr(runner, "failed_step", None) is not None:  # it failed in a step, not before the first
            record_runtime()
            runtime = compiled.metadata["runtime"]
            runtime["failed_step"] = runner.start_step + runner.failed_step
            runtime["end_step"] = runner.start_step + max(runner.completed_step or 0, 0)
        raise
    record_runtime()
    return ModelRunResult(
        lgca=lgca,
        spec=compiled.spec,
        context=compiled.context,
        pipeline=compiled.pipeline,
        metadata=deepcopy(compiled.metadata),
        data=RunData.from_run(lgca, observers),
    )


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
