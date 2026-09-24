# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Extensions to the base LGCA classes.

Provides abstract classes for identity-based models with or without
volume exclusion.
"""


import logging
from abc import ABC
from copy import copy, deepcopy
from itertools import chain

import numpy as np
from .plot_data import history_steps
from numpy import random as npr
from tqdm.auto import tqdm

from .base import (
    warn_user,
    plt,
    LGCA_base,
    _validate_count_nodes,
    _validate_density,
    _validate_nonnegative_int,
    _validate_positive,
    _validate_positive_int,
)
from .plots import muller_plot


from .ib_base import IBLGCA_base
from .nove_base import NoVE_LGCA_base
from .list_utils import get_arr_of_empty_lists, _copy_arr_of_lists

logger = logging.getLogger(__name__)


_STORE_BOUNDARIES = ("periodic", "reflecting", "absorbing")


def _store_aware(list_method):
    """Wrap a geometry's list propagation: with a current cell table, use the lookup table instead."""

    def propagation(self, *args, **kwargs):
        if self.__dict__.get("_store") is not None:
            return self._store_move("propagation")
        return list_method(self, *args, **kwargs)

    propagation.__doc__ = list_method.__doc__
    propagation.__name__ = list_method.__name__
    propagation.__wrapped__ = list_method
    return propagation


def _flatten_ids(nodes):
    """Return all particle IDs stored in an object array of label lists."""
    return np.fromiter(chain.from_iterable(np.asarray(nodes, dtype=object).flat), dtype=np.intp)


class NoVE_IBLGCA_base(NoVE_LGCA_base, IBLGCA_base, ABC):
    """
    Base class for identity-based LGCA without volume exclusion.

    Explicit object-array channels contain lists of unique non-negative integer
    particle IDs. ID zero denotes a particle in this backend. Uniqueness applies
    to physical sites before boundary copies are added.

    Internally, the cells can also be held as a table (label and padded slot
    of every cell), which rules on the lattice state, the boundary conditions
    and propagation update without building lists. ``nodes`` builds the
    object array from the table when it is read; whichever of the two was
    written last is the state, and both are while neither was written.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "propagation" in cls.__dict__:
            cls.propagation = _store_aware(cls.__dict__["propagation"])

    @property
    def nodes(self):
        """Lists of cell labels per channel, shape ``padded dims + (K,)``."""
        store = self.__dict__.get("_store")
        if store is not None:  # the caller may change the lists in place, so they become the state
            if self.__dict__.get("_nodes_array") is None:
                self.__dict__["_nodes_array"] = self._nodes_from_store(*store)
            self.__dict__["_store"] = None
        try:
            return self.__dict__["_nodes_array"]
        except KeyError:
            raise AttributeError("nodes") from None

    @nodes.setter
    def nodes(self, value):
        self.__dict__["_nodes_array"] = value
        self.__dict__["_store"] = None

    @property
    def nodes_t(self):
        """Recorded lists of labels per channel, shape ``(times,) + dims + (K,)``.

        ``NodeRecorder`` stores compact cell tables (:attr:`cells_t`); the
        lists are built when ``nodes_t`` is first read.
        """
        cached = self.__dict__.get("_nodes_t")
        if cached is None:
            history = self.__dict__.get("cells_t")
            if history is None:
                raise AttributeError("nodes_t")
            cached = self.__dict__["_nodes_t"] = history.to_nodes()
        return cached

    @nodes_t.setter
    def nodes_t(self, value):
        self.__dict__["_nodes_t"] = value

    def _start_cell_history(self, length):
        """Record cell tables from now on (see ``NodeRecorder``); False if the boundary has no table."""
        from .cells import CellHistory

        if self._cell_table() is None:
            return False
        self.cells_t = CellHistory(length, self.dims, self.K)
        self.__dict__["_nodes_t"] = None
        return True

    def _record_cells(self, index):
        labels, slots = self._cell_table()
        inner = self._slot_table()["interior"][slots]
        inside = inner >= 0
        self.cells_t.record(index, labels[inside], inner[inside])
        self.__dict__["_nodes_t"] = None

    def set_bc(self, bc):
        super().set_bc(bc)
        self.__dict__["_slot_maps"] = None
        self._list_boundaries = self.apply_boundaries
        self.apply_boundaries = self._apply_boundaries

    def _apply_boundaries(self):
        if self.__dict__.get("_store") is not None:
            self._store_move("boundary")
        else:
            self._list_boundaries()

    def _cell_table(self):
        """Labels and padded slots of all cells, or None if the boundary has no lookup table.

        Builds the table from ``nodes`` if they are the current state; ghost
        nodes count only where they hold cells in flight (not with periodic
        boundaries, where they copy interior nodes).
        """
        if self.bc not in _STORE_BOUNDARIES:
            return None
        store = self.__dict__.get("_store")
        if store is None:
            flat = self.__dict__["_nodes_array"].reshape(-1)
            lengths = np.fromiter(map(len, flat), dtype=np.int64, count=flat.size)
            if self.bc == "periodic":
                lengths[self._slot_table()["interior"] < 0] = 0
            used = np.flatnonzero(lengths)
            labels = np.fromiter((label for slot in used for label in flat[slot]), dtype=np.int64,
                                 count=int(lengths.sum()))
            store = (labels, np.repeat(used, lengths[used]))
            self.__dict__["_store"] = store  # the lists stay valid until either is written
        return store

    def _set_cell_table(self, labels, slots):
        """Make the table of labels and padded slots the state of the model."""
        self.__dict__["_store"] = (np.asarray(labels, dtype=np.int64), np.asarray(slots, dtype=np.int64))
        self.__dict__["_nodes_array"] = None

    def _slot_table(self):
        maps = self.__dict__.get("_slot_maps")
        if maps is None:
            from .cells import slot_maps

            maps = self.__dict__["_slot_maps"] = slot_maps(self)
        return maps

    def _store_move(self, which):
        labels, slots = self.__dict__["_store"]
        slots = self._slot_table()[which][slots]
        kept = slots >= 0
        self.__dict__["_store"] = (labels[kept], slots[kept])
        self.__dict__["_nodes_array"] = None

    def _store_counts(self, slots):
        """Cells per padded channel, with periodic ghost nodes copying the interior."""
        maps = self._slot_table()
        counts = np.bincount(slots, minlength=int(np.prod(maps["shape"])))
        ghosts = maps["source"] >= 0
        counts[ghosts] = counts[maps["source"][ghosts]]
        return counts.reshape(maps["shape"])

    def _nodes_from_store(self, labels, slots):
        maps = self._slot_table()
        size = int(np.prod(maps["shape"]))
        ends = np.cumsum(np.bincount(slots, minlength=size)).tolist()
        ordered = labels[np.argsort(slots, kind="stable")].tolist()
        flat = np.empty(size, dtype=object)
        start = 0
        for slot, end in enumerate(ends):
            flat[slot] = ordered[start:end]
            start = end
        for ghost in np.flatnonzero(maps["source"] >= 0).tolist():
            flat[ghost] = list(flat[maps["source"][ghost]])
        return flat.reshape(maps["shape"])
    _LOCAL_ENSEMBLE_INTERACTIONS = {
        "birth",
        "birthdeath",
        "birthdeath_cancerdfe",
        "diffusion",
        "go_or_grow",
        "go_or_grow_glioblastoma",
        "go_or_grow_kappa",
        "only_propagation",
        "random_walk",
    }

    def __init__(self, nodes=None, dims=None, density=.1, restchannels=1,
                 bc='periodic', seed=None, propagation=True, **kwargs):
        """
        Initialize class instance.
        :param nodes:
        :param l:
        :param restchannels:
        :param density:
        :param bc:
        :param r_int:
        :param kwargs:
        :param propagation: execute propagation step during a timestep
        """
        self._validate_kwargs(kwargs)
        self.enable_propagation = propagation
        self.r_int = _validate_positive_int(kwargs.pop("r_int", 1), "r_int")
        self.rng = npr.default_rng(seed=seed)
        self.props = {}
        self.length_checker = lambda arr: np.fromiter((len(x) for x in arr.flat), dtype=np.uint,
                                                      count=arr.size).reshape(arr.shape)
        self.set_bc(bc)
        self.interaction_params = {}
        restchannels = _validate_nonnegative_int(restchannels, "restchannels")
        if nodes is not None:
            nodes = np.asarray(nodes)
            if nodes.dtype != object:
                nodes = _validate_count_nodes(nodes)
            else:
                seen = set()
                for channel in nodes.flat:
                    if not isinstance(channel, list):
                        raise ValueError("nodes identity channels must contain lists of integer IDs")
                    for label in channel:
                        if (isinstance(label, (bool, np.bool_))
                                or not isinstance(label, (int, np.integer))
                                or not 0 <= label < np.iinfo(np.intp).max):
                            raise ValueError("nodes particle IDs must be non-negative indexable integers")
                        if int(label) in seen:
                            raise ValueError("nodes particle IDs must be unique on the physical lattice")
                        seen.add(int(label))
        if restchannels != 1:
            restchannels = 1
            warn_user("There can only be one rest channel in this LGCA class. Setting to 1 to prevent issues")
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes, capacity=kwargs.get("capacity"))
        self._validate_model_setup(nodes=nodes)
        if nodes is None:
            _validate_density(density)
        self.init_coords()
        self.init_nodes(density, nodes=nodes)
        self.calc_max_label()
        self.update_dynamic_fields()
        self.mean_prop_t = {}
        self.set_interaction(**kwargs)


    def update_dynamic_fields(self):
        store = self.__dict__.get("_store")
        if store is not None:
            self.channel_pop = self._store_counts(store[1]).astype(np.uint)
        else:
            self.channel_pop = self.length_checker(self.nodes)  # population of a channel
        self.cell_density = self.channel_pop.sum(-1)  # population of a node

    def _channel_counts(self, nodes, history=False):
        """Return channel populations for label lists; count arrays pass through."""
        nodes = np.asarray(nodes)
        if nodes.dtype == object:
            return self.length_checker(nodes)
        return nodes

    def convert_int_to_ib(self, occ):
        """
        Convert an array of integers representing the occupation numbers of an lgca to an array consisting of lists of
        individual cell labels, starting at 0. The length of each list corresponds to the entry in 'occ'.
        :param occ: array of occupation numbers. must match the lgca dimensions
        :return: array, where each entry is a list of individual cell labels.
        """
        ntot = occ.sum()
        labels = list(range(ntot))
        tempnodes = np.empty(occ.shape, dtype=object)
        counter = 0
        for ind, dens in np.ndenumerate(occ):
            tempnodes[ind] = labels[counter:counter+dens]
            counter += dens

        return tempnodes

    def random_reset(self, density):
        """Populate the lattice from a Poisson distribution with mean ``density`` per node."""
        _validate_density(density)
        lam = density / self.K
        numbers = self.rng.poisson(lam=lam, size=self.dims + (self.K,))
        tempnodes = self.convert_int_to_ib(numbers)
        self.nodes[self.nonborder] = tempnodes
        self.maxlabel = numbers.sum()
        self.update_dynamic_fields()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, recordchanneldens=False,
                showprogress=True, recordfampop=False):
        from .simulation import (
            ChannelDensityRecorder,
            DensityRecorder,
            FamilyPopulationRecorder,
            NodeRecorder,
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
        if recordchanneldens:
            observers.append(ChannelDensityRecorder())
        if recordfampop:
            observers.append(FamilyPopulationRecorder())
        run_timeevo(self, timesteps=timesteps, observers=observers, showprogress=showprogress)

    def init_families(self, type='homogeneous', mutation=True):
        """Initialize family tracking; see :meth:`IBLGCA_base.init_families`.

        Labels start at 0 in this model, so every label, including 0, is a
        cell: family 0 stays the root of the tree, and cell ``i`` of a
        heterogeneous population founds family ``i + 1``.
        """
        cells = int(self.maxlabel) + 1
        if type == 'homogeneous':
            self.props.update(family=[1] * cells)
            if mutation:
                self.family_props = {'ancestor': [0, 0], 'descendants': [[1], []]}
                self.maxfamily = 1
        elif type == 'heterogeneous':
            families = list(range(1, cells + 1))
            self.props.update(family=families)
            if mutation:
                self.family_props = {'ancestor': [0] * (cells + 1),
                                     'descendants': [families] + [[] for _ in range(cells)]}
                self.maxfamily = cells

    def calc_max_label(self):
        cells = _flatten_ids(self.nodes)
        self.maxlabel = int(cells.max()) if cells.size else 0

    def get_prop(self, nodes=None, props=None, propname=None):
        """
        Return array of property "propname" on every node, given the lattice configuration "nodes" and the property
        dictionary "props".
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        prop = np.array(props[propname])
        proparray = prop[_flatten_ids(nodes)]
        return proparray

    def calc_prop_mean(self, nodes=None, props=None, propname=None):
        """
        Calculate the mean value of a property "propname" on every node, given the node configuration "nodes" and
        the properties "props"
        """
        cells = nodes.sum(-1)
        mean_prop = np.ma.masked_all(cells.shape)
        proparray = np.array(props[propname])
        for ind, loccells in np.ndenumerate(cells):
            if loccells:
                nodeprops = proparray[loccells]
                mean_prop[ind] = nodeprops.mean()
                mean_prop.mask[ind] = 0
        return mean_prop

    def calc_prop_mean_spatiotemp(self, nodes_t=None, props=None):
        """
        Given the lgca states nodes_t, calculate the mean of individual cell properties at each node for each time point.
        :param nodes_t: Recorded node states. If omitted, use recorded state of latest timeevo.
        :param props: dictionary of individual cell properties. If omitted, props dictionary of current instance is used
        :return:
        """
        if nodes_t is None:
            nodes_t = self.nodes_t
        if props is None:
            props = self.props
        tmax = nodes_t.shape[0]
        for key in props:
            self.mean_prop_t[key] = np.ma.masked_all((tmax, *self.dims))
            # self.mean_prop_vel_t[key] = np.zeros([tmax,l])
            # self.mean_prop_rest_t[key] = np.zeros([tmax,l])
            self.mean_prop_t[key] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t)
            # self.mean_prop_vel_t[key][t] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t[t][...,0:(self.K-1)])
            # self.mean_prop_rest_t[key][t] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t[t][...,(self.K-1):])

        return self.mean_prop_t


    def plot_prop_timecourse(self, nodes_t=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot the time evolution of the cell property 'propname'
        :param nodes_t:
        :param props:
        :param propname:
        :param figindex:
        :param figsize:
        :param kwargs: keyword arguments for the matplotlib.plot command
        :return:
        """

        implicit = nodes_t is None
        steps = kwargs.pop("steps", None)
        if nodes_t is None:
            nodes_t = self.nodes_t

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        proparray = np.array(props[propname])
        prop_t = [proparray[_flatten_ids(nodes)] for nodes in nodes_t]
        mean_prop_t = np.array([np.mean(prop) if len(prop) > 0 else np.nan for prop in prop_t])
        std_mean_prop_t = np.array \
            ([np.std(prop, ddof=1) / np.sqrt(len(prop)) if len(prop) > 0 else np.nan for prop in prop_t])
        if figindex is None:
            fig = plt.gcf()

        else:
            fig = plt.figure(num=figindex)

        if figsize is not None:
            fig.set_size_inches(figsize)

        tmax = nodes_t.shape[0]
        yerr = std_mean_prop_t
        x = history_steps(self, tmax, "nodes_steps", steps, implicit=implicit)
        y = mean_prop_t

        plt.xlabel('$t$')
        plt.ylabel('${}$'.format(propname))
        plt.title('Time course of the cell property')
        line = plt.plot(x, y, **kwargs)
        errors = plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True, interpolate=True)
        return line, errors

    def plot_prop_hist(self, nodes=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot histogram of cell property 'propname' of cells in 'nodes'. Per default, the current lgca state and the
        first property is shown.
        :param nodes:
        :param props:
        :param propname:
        :param figindex:
        :param figsize:
        :param kwargs:
        :return:
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        propvals = np.asarray(props[propname])[_flatten_ids(nodes)]
        plt.figure(num=figindex, figsize=figsize)
        plt.hist(propvals, **kwargs)
        plt.xlabel('{}'.format(propname))
        plt.ylabel('Count')

    def plot_prop_2dhist(self, nodes=None, props=None, propnames=None, figindex=None, figsize=None, bins=20,
                         **kwargs):
        """
        Plot a 2D histogram of two cell properties with marginal histograms.

        Parameters
        ----------
        nodes : numpy.ndarray, optional
            Object array of cell-ID lists. Defaults to the current physical lattice.
        props : dict, optional
            Property dictionary. Defaults to ``self.props``.
        propnames : sequence of two str, optional
            Properties shown on the x and y axes. Defaults to the first two properties.
        figindex : int or str, optional
            Figure identifier passed to :func:`matplotlib.pyplot.figure`.
        figsize : tuple of float, optional
            Figure size in inches.
        bins : int or sequence, default=20
            Bins for the joint and the marginal histograms.
        **kwargs
            Further arguments for :meth:`matplotlib.axes.Axes.hist2d`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            Joint, top-marginal and right-marginal axes.
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propnames is None:
            propnames = list(props)[:2]
        if len(propnames) != 2:
            raise ValueError("plot_prop_2dhist requires exactly two property names.")
        propname1, propname2 = propnames

        ids = _flatten_ids(nodes)
        propvals1 = np.asarray(props[propname1])[ids]
        propvals2 = np.asarray(props[propname2])[ids]

        fig = plt.figure(num=figindex, figsize=figsize)
        grid = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(grid[1, 0])
        ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
        ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
        ax.hist2d(propvals1, propvals2, bins=bins, **kwargs)
        ax_top.hist(propvals1, bins=bins)
        ax_right.hist(propvals2, bins=bins, orientation='horizontal')
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
        ax.set_xlabel(str(propname1))
        ax.set_ylabel(str(propname2))
        return fig, (ax, ax_top, ax_right)

    def calc_family_pop_alive(self):
        """
        Calculate how many cells of each family are alive.
        :returns: np.ndarray fam_pop_array - array of family population counts indexed by family ID
        """
        if 'family' not in self.props:
            raise RuntimeError("Family properties are not recorded by the LGCA, choose suitable interaction.")

        cells_alive_list = []
        for site_list_collection in self.nodes[self.nonborder].flat:
            cells_alive_list.extend(site_list_collection)

        cells_alive = np.array(cells_alive_list,
                               dtype=np.intp)  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive]  # filter family array for families of live cells
        fam_alive, fam_pop = np.unique(cell_fam_alive, return_counts=True)  # count number of cells for each family
        # transform into array with population entry for all families that ever existed
        fam_pop_array = np.zeros(self.maxfamily+1, dtype=int)
        fam_pop_array[fam_alive] = fam_pop
        return fam_pop_array
        # alternative: look up family for each live cell, then do unique on those altered nodes

    def list_families_alive(self):
        """
        Calculate which families are alive.
        :returns: np.ndarray - array of family IDs in ascending order
        """
        cells_alive = _flatten_ids(self.nodes[self.nonborder])  # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive]  # filter family array for families of live cells
        return np.unique(cell_fam_alive) # remove duplicate entries


__all__ = ["NoVE_IBLGCA_base"]
