"""Multi-species LGCA base classes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .base_extensions import NoVE_LGCA_base
from .base import LGCA_base


class MultiSpeciesLGCA_base(LGCA_base):
    """Classical LGCA supporting multiple species."""

    def __init__(self, *, n_species: int = 1, **kwargs: Any) -> None:
        self.n_species = n_species
        super().__init__(**kwargs)

    def _warn_nodes_shape(self, nodes: np.ndarray | None) -> None:
        if nodes is None:
            return
        expected = self.dims + (self.n_species, self.K)
        if nodes.shape != expected:
            warnings.warn(
                f"Provided nodes have shape {nodes.shape}, expected {expected}.",
                UserWarning,
            )

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
        self.n_species = n_species
        super().__init__(**kwargs)

    def _warn_nodes_shape(self, nodes: np.ndarray | None) -> None:
        if nodes is None:
            return
        expected = self.dims + (self.n_species, self.K)
        if nodes.shape != expected:
            warnings.warn(
                f"Provided nodes have shape {nodes.shape}, expected {expected}.",
                UserWarning,
            )

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

