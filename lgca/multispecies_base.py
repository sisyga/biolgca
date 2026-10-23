"""Multi-species LGCA base classes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._warnings import warn_user
from .base import LGCA_base, _validate_density, channel_sum
from .nove_base import NoVE_LGCA_base, _poisson_channel_populations

_SPATIAL_NDIMS = {
    "lin": 1,
    "square": 2,
    "hex": 2,
    "cubic": 3,
    "moore": 3,
}


class _MultiSpecies:
    """The species axis of the multi-species classes: nodes of shape ``dims + (n_species, K)``.

    The volume-exclusion and the no-volume-exclusion base classes differ in the dtype of the
    nodes (``_node_dtype``) and in how a given initial state is converted (``_initial_nodes``).
    """

    _node_dtype: type = bool

    def __init__(self, *, n_species: int = 1, **kwargs: Any) -> None:
        if isinstance(n_species, bool) or int(n_species) != n_species or n_species < 1:
            raise ValueError("n_species must be a positive integer.")
        self.n_species = int(n_species)
        super().__init__(**kwargs)

    @classmethod
    def _get_valid_kwargs(cls):
        valid = super()._get_valid_kwargs()
        valid.add("n_species")
        return valid

    def _spatial_ndim(self) -> int:
        try:
            return _SPATIAL_NDIMS[self.geometry]
        except KeyError as exc:
            raise ValueError(f"Multi-species LGCA is not implemented for geometry {self.geometry!r}.") from exc

    def _set_spatial_shape(self, shape) -> None:
        shape = tuple(int(dim) for dim in shape)
        if self._spatial_ndim() == 1:
            (self.l,) = shape
            self.dims = (self.l,)
        elif self._spatial_ndim() == 2:
            self.lx, self.ly = shape
            self.dims = (self.lx, self.ly)
        else:
            self.lx, self.ly, self.lz = shape
            self.dims = (self.lx, self.ly, self.lz)

    def _set_dims_from_nodes(self, nodes: np.ndarray, *, allow_multiple_restchannels: bool = True) -> bool:
        if nodes is None:
            return False

        nodes = np.asarray(nodes)
        spatial_ndim = self._spatial_ndim()
        expected_ndim = spatial_ndim + 2
        if nodes.ndim != expected_ndim:
            raise ValueError(
                "Multi-species nodes must have shape dims + (n_species, channels); "
                f"got {nodes.shape}."
            )

        if nodes.shape[-2] != self.n_species:
            raise ValueError(
                f"nodes species axis has length {nodes.shape[-2]}, expected n_species={self.n_species}."
            )

        self._set_spatial_shape(nodes.shape[:spatial_ndim])
        self.K = int(nodes.shape[-1])
        if self.K < self.velocitychannels:
            raise RuntimeError(
                f"Not enough channels specified for the chosen geometry! "
                f"Required: {self.velocitychannels}, provided: {self.K}"
            )
        self.restchannels = self.K - self.velocitychannels
        if not allow_multiple_restchannels and self.restchannels > 1:
            raise RuntimeError(
                f"Only one resting channel allowed, but {self.restchannels} resting channels specified!"
            )
        return True

    def _warn_nodes_shape(self, nodes: np.ndarray | None) -> None:
        if nodes is None:
            return
        expected = self.dims + (self.n_species, self.K)
        if nodes.shape != expected:
            warn_user(f"Provided nodes have shape {nodes.shape}, expected {expected}.")

    def _channel_counts(self, nodes, history=False):
        """Return channel populations summed over species; summed arrays pass through."""
        nodes = np.asarray(nodes)
        if nodes.ndim == len(self.dims) + int(history) + 2:
            return nodes.sum(axis=-2)
        return nodes

    def _initial_nodes(self, nodes: np.ndarray) -> np.ndarray:
        return nodes.astype(self._node_dtype)

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        """Create the padded nodes ``dims + (n_species, K)`` and fill them with ``nodes`` or at random."""
        shape = tuple(dim + 2 * self.r_int for dim in self.dims) + (self.n_species, self.K)
        self.nodes = np.zeros(shape, dtype=self._node_dtype)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = self._initial_nodes(nodes)
            self.apply_boundaries()


class MultiSpeciesLGCA_base(_MultiSpecies, LGCA_base):
    """Classical LGCA supporting multiple species."""

    _node_dtype = bool

    def _initial_nodes(self, nodes: np.ndarray) -> np.ndarray:
        return self._ensure_bool_nodes(nodes)

    def set_dims(self, dims=None, nodes=None, restchannels=0, **kwargs) -> None:
        if self._set_dims_from_nodes(nodes):
            return
        super().set_dims(dims=dims, nodes=None, restchannels=restchannels)

    def random_reset(self, density):
        """Randomly initialize a total density distributed over all species."""
        _validate_density(density, max_density=self.n_species * self.K)
        self.nodes = self.rng.random(self.nodes.shape) < (density / (self.n_species * self.K))
        self.apply_boundaries()
        self.update_dynamic_fields()

    def update_dynamic_fields(self) -> None:
        self.species_density = channel_sum(self.nodes)
        self.cell_density = channel_sum(self.species_density)


class MultiSpeciesNoVE_LGCA_base(_MultiSpecies, NoVE_LGCA_base):
    """No-volume-exclusion LGCA with multiple species."""

    _node_dtype = np.uint

    _LOCAL_ENSEMBLE_INTERACTIONS = NoVE_LGCA_base._LOCAL_ENSEMBLE_INTERACTIONS | {
        "birth",
        "birthdeath",
    }

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None) -> None:
        if self._set_dims_from_nodes(nodes, allow_multiple_restchannels=False):
            if capacity is not None:
                self.capacity = capacity
            elif restchannels is not None and restchannels > 1:
                self.capacity = self.velocitychannels + restchannels
            else:
                self.capacity = self.K
            return
        super().set_dims(dims=dims, nodes=None, restchannels=restchannels, capacity=capacity)

    def random_reset(self, density):
        """Populate a total density distributed over all species."""
        _validate_density(density)
        self.nodes = _poisson_channel_populations(self, density / self.n_species, self.nodes.shape)
        self.apply_boundaries()
        self.update_dynamic_fields()

    def update_dynamic_fields(self) -> None:
        self.channel_pop = self.nodes
        self.species_density = channel_sum(self.channel_pop)
        self.cell_density = channel_sum(self.species_density)

    def timeevo(
        self,
        timesteps: int = 100,
        record: bool = False,
        recordN: bool = False,
        recorddens: bool = True,
        showprogress: bool = True,
        recordpertype: bool = False,
    ) -> None:
        from .simulation import (
            DensityRecorder,
            NodeRecorder,
            PerTypeRecorder,
            PopulationRecorder,
            run_timeevo,
        )

        observers = []
        if record:
            observers.append(NodeRecorder())
        if recordN:
            observers.append(PopulationRecorder())
        if recorddens:
            observers.append(DensityRecorder())
        if recordpertype:
            observers.append(PerTypeRecorder())
        run_timeevo(self, timesteps=timesteps, observers=observers, showprogress=showprogress)
