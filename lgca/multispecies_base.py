"""Multi-species LGCA base classes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .nove_base import NoVE_LGCA_base
from .base import LGCA_base, _validate_density


_SPATIAL_NDIMS = {
    "lin": 1,
    "square": 2,
    "hex": 2,
    "cubic": 3,
    "moore": 3,
}


class MultiSpeciesLGCA_base(LGCA_base):
    """Classical LGCA supporting multiple species."""

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

    def set_dims(self, dims=None, nodes=None, restchannels=0, **kwargs) -> None:
        if self._set_dims_from_nodes(nodes):
            return
        super().set_dims(dims=dims, nodes=None, restchannels=restchannels)

    def _warn_nodes_shape(self, nodes: np.ndarray | None) -> None:
        if nodes is None:
            return
        expected = self.dims + (self.n_species, self.K)
        if nodes.shape != expected:
            warnings.warn(
                f"Provided nodes have shape {nodes.shape}, expected {expected}.",
                UserWarning,
            )

    def random_reset(self, density):
        """Randomly initialize a total density distributed over all species."""
        _validate_density(density, max_density=self.n_species * self.K)
        self.nodes = self.rng.random(self.nodes.shape) < (density / (self.n_species * self.K))
        self.apply_boundaries()
        self.update_dynamic_fields()

    def update_dynamic_fields(self) -> None:
        self.species_density = self.nodes.sum(-1)
        self.cell_density = self.species_density.sum(-1)

    def timeevo(
        self,
        timesteps: int = 100,
        record: bool = False,
        recordN: bool = False,
        recorddens: bool = True,
        showprogress: bool = True,
        recordpertype: bool = False,
    ) -> None:
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros(
                (timesteps + 1,) + self.dims + (self.n_species, self.K),
                dtype=self.nodes.dtype,
            )
            self.nodes_t[0, ...] = self.nodes[self.nonborder]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.dens_t[0, ...] = self.species_density[self.nonborder]
        if recordpertype:
            self.velcells_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.velcells_t[0, ...] = self.nodes[self.nonborder][..., :self.velocitychannels].sum(-1)
            self.restcells_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.restcells_t[0, ...] = self.nodes[self.nonborder][..., self.velocitychannels:].sum(-1)
        for t in tqdm(range(1, timesteps + 1), disable=1 - showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.nonborder]
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.species_density[self.nonborder]
            if recordpertype:
                self.velcells_t[t, ...] = self.nodes[self.nonborder][..., :self.velocitychannels].sum(-1)
                self.restcells_t[t, ...] = self.nodes[self.nonborder][..., self.velocitychannels:].sum(-1)


class MultiSpeciesNoVE_LGCA_base(NoVE_LGCA_base):
    """No-volume-exclusion LGCA with multiple species."""

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

    _spatial_ndim = MultiSpeciesLGCA_base._spatial_ndim
    _set_spatial_shape = MultiSpeciesLGCA_base._set_spatial_shape
    _set_dims_from_nodes = MultiSpeciesLGCA_base._set_dims_from_nodes

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

    def _warn_nodes_shape(self, nodes: np.ndarray | None) -> None:
        if nodes is None:
            return
        expected = self.dims + (self.n_species, self.K)
        if nodes.shape != expected:
            warnings.warn(
                f"Provided nodes have shape {nodes.shape}, expected {expected}.",
                UserWarning,
            )

    def random_reset(self, density):
        """Populate a total density distributed over all species."""
        _validate_density(density)
        density = density / self.n_species
        density = density / self.capacity
        draw1 = self.rng.poisson(lam=density, size=self.nodes.shape)
        if self.capacity > self.K:
            draw2 = self.rng.poisson(lam=density, size=self.nodes.shape[:-1] + ((self.capacity - self.K),))
            draw1[..., -1] += draw2.sum(-1)
        self.nodes = draw1
        self.apply_boundaries()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        if kwargs.get("interaction") == "excitable_medium_ms":
            raise ValueError("excitable_medium_ms requires volume exclusion.")
        super().set_interaction(**kwargs)

    def update_dynamic_fields(self) -> None:
        self.channel_pop = self.nodes
        self.species_density = self.channel_pop.sum(-1)
        self.cell_density = self.species_density.sum(-1)

    def timeevo(
        self,
        timesteps: int = 100,
        record: bool = False,
        recordN: bool = False,
        recorddens: bool = True,
        showprogress: bool = True,
        recordpertype: bool = False,
    ) -> None:
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros(
                (timesteps + 1,) + self.dims + (self.n_species, self.K),
                dtype=self.nodes.dtype,
            )
            self.nodes_t[0, ...] = self.nodes[self.nonborder]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.dens_t[0, ...] = self.species_density[self.nonborder]
        if recordpertype:
            self.velcells_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.velcells_t[0, ...] = self.channel_pop[self.nonborder][..., :self.velocitychannels].sum(-1)
            self.restcells_t = np.zeros((timesteps + 1,) + self.dims + (self.n_species,))
            self.restcells_t[0, ...] = self.channel_pop[self.nonborder][..., self.velocitychannels:].sum(-1)
        for t in tqdm(range(1, timesteps + 1), disable=1 - showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.nonborder]
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.species_density[self.nonborder]
            if recordpertype:
                self.velcells_t[t, ...] = self.channel_pop[self.nonborder][..., :self.velocitychannels].sum(-1)
                self.restcells_t[t, ...] = self.channel_pop[self.nonborder][..., self.velocitychannels:].sum(-1)

