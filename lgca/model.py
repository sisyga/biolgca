"""Declarative model specifications and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from . import get_lgca
from .pipeline import InteractionPipelineSpec, compile_pipeline


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

    _validate_spec(spec)
    lgca = _build_lgca(spec)
    metadata = _metadata_from_spec(spec)
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
    if spec.time.steps < 0:
        raise ValueError("time.steps must be non-negative.")
    if spec.state.nodes is not None and spec.state.density is not None:
        raise ValueError("state.nodes and state.density are mutually exclusive.")
    if spec.state.n_species < 1:
        raise ValueError("state.n_species must be positive.")


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
    kwargs.update(dict(spec.state.parameters))
    return get_lgca(
        geometry=spec.space.geometry,
        ib=spec.state.identity_based,
        ve=spec.state.volume_exclusion,
        n_species=spec.state.n_species,
        **kwargs,
    )


def _metadata_from_spec(spec: ModelSpec) -> dict[str, Any]:
    return {
        "title": spec.description.title,
        "geometry": spec.space.geometry,
        "boundary": spec.space.boundary,
        "steps": spec.time.steps,
        "seed": spec.time.seed,
        "volume_exclusion": spec.state.volume_exclusion,
        "identity_based": spec.state.identity_based,
        "n_species": spec.state.n_species,
        "propagation": spec.dynamics.propagation,
        "operator_names": [],
        "reorientation_term_names": [],
        "observer_names": _observer_names(spec.analysis),
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

    def run(self):
        self.lgca.update_dynamic_fields()
        for observer in self.observers:
            setup = getattr(observer, "setup", None)
            if setup is not None:
                setup(self.lgca, self)

        self._notify_observers(0)
        for step in tqdm(range(1, self.timesteps + 1), disable=not self.showprogress):
            self.compiled.pipeline.execute_step(self.compiled.context, step)
            self._notify_observers(step)

        for observer in self.observers:
            finalize = getattr(observer, "finalize", None)
            if finalize is not None:
                finalize(self.lgca, self)
        return self.lgca

    def _notify_observers(self, step: int) -> None:
        for observer in self.observers:
            schedule = getattr(observer, "schedule", None)
            if schedule is None or schedule.should_run(step):
                observer.on_step(self.lgca, step)
