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


def _flatten_ids(nodes):
    """Return all particle IDs stored in an object array of label lists."""
    return np.fromiter(chain.from_iterable(np.asarray(nodes, dtype=object).flat), dtype=np.intp)


class NoVE_IBLGCA_base(NoVE_LGCA_base, IBLGCA_base, ABC):
    """
    Base class for identity-based LGCA without volume exclusion.

    Explicit object-array channels contain lists of unique non-negative integer
    particle IDs. ID zero denotes a particle in this backend. Uniqueness applies
    to physical sites before boundary copies are added.
    """
    interactions = [
        'go_or_grow',
        'go_or_grow_kappa',
        'go_or_grow_glioblastoma',
        'birth',
        'birthdeath',
        'birthdeath_cancerdfe',
        'random_walk',
        'randomwalk',
        'steric_evolution',
    ]
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

    def set_interaction(self, **kwargs):
        if self._set_callable_interaction(kwargs):
            return
        from lgca.nove_ib_interactions import random_walk, birth, birthdeath, birthdeath_cancerdfe, go_or_grow, \
            evo_steric, go_or_grow_kappa, go_or_grow_glioblastoma
        from lgca.interactions import only_propagation
        if 'interaction' in kwargs:
            interaction = kwargs['interaction'].replace(" ", "_")
            if interaction in ('random_walk', 'diffusion'):
                self.interaction = random_walk
            elif interaction == 'only_propagation':
                self.interaction = only_propagation

            elif interaction in ('birth', 'birthdeath'):
                self.interaction = birthdeath if interaction == 'birthdeath' else birth
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    logger.info('capacity of channel set to %s', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                    if interaction == 'birth':
                        warn_user("Death rate defined but not used in birth interaction.")
                else:
                    if interaction == 'birthdeath':
                        self.interaction_params['r_d'] = 0.02
                        logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

                if 'std' in kwargs:
                    self.interaction_params['std'] = kwargs['std']
                else:
                    self.interaction_params['std'] = 0.01
                    logger.info('standard deviation set to = %s', self.interaction_params['std'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    logger.info('Max. birth rate set to a_max = %s', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    logger.info('Rest channel weight set to gamma = %s', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'birthdeath_cancerdfe':
                self.interaction = birthdeath_cancerdfe
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    logger.info('capacity of channel set to %s', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

                if 'p_d' in kwargs:
                    self.interaction_params['p_d'] = kwargs['p_d']
                else:
                    self.interaction_params['p_d'] = 1.4e-5  # from macfarlane 2014
                    logger.info('probability of drivers set to = %s', self.interaction_params['p_d'])

                if 'p_p' in kwargs:
                    self.interaction_params['p_p'] = kwargs['p_p']
                else:
                    self.interaction_params['p_p'] = 0.1  # from macfarlane 2014
                    logger.info('probability of passengers set to = %s', self.interaction_params['p_p'])

                if 's_d' in kwargs:
                    self.interaction_params['s_d'] = kwargs['s_d']
                else:
                    self.interaction_params['s_d'] = .1 * self.interaction_params['r_b']  # from macfarlane 2014
                    logger.info('driver strength set to = %s', self.interaction_params['s_d'])

                if 's_p' in kwargs:
                    self.interaction_params['s_p'] = kwargs['s_p']
                else:
                    self.interaction_params['s_p'] = .001 * self.interaction_params['r_b']  # from macfarlane 2014
                    logger.info('passenger strength set to = %s', self.interaction_params['s_p'])

                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    logger.info('Max. birth rate set to a_max = %s', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    logger.info('Rest channel weight set to gamma = %s', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'go_or_grow':
                self.interaction = go_or_grow
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    warn_user('This interaction requires a rest channel.')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    logger.info('node capacity set to %s', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    logger.info('std of kappa set to %s', self.interaction_params['kappa_std'])

                if 'theta_std' in kwargs:
                    self.interaction_params['theta_std'] = kwargs['theta_std']
                else:
                    self.interaction_params['theta_std'] = 0.05
                    logger.info('std of theta set to %s', self.interaction_params['theta_std'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])

                if 'kappa' in kwargs:
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    if hasattr(theta, '__iter__'):
                        self.interaction_params['theta'] = list(theta)
                    else:
                        self.interaction_params['theta'] = [theta] * (self.maxlabel + 1)
                else:
                    self.interaction_params['theta'] = [0.5] * (self.maxlabel + 1)
                    logger.info('switch threshold set to theta = %s', self.interaction_params['theta'][0])
                self.props.update(theta=np.array(self.interaction_params['theta']))

            elif interaction == 'go_or_grow_kappa':
                self.interaction = go_or_grow_kappa
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    warn_user('This interaction requires a rest channel.')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    logger.info('node capacity set to %s', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    logger.info('std of kappa set to %s', self.interaction_params['kappa_std'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])

                if 'kappa' in kwargs:
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    self.interaction_params['theta'] = theta
                else:
                    self.interaction_params['theta'] = 0.5
                    logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])

            elif interaction == 'steric_evolution':
                self.interaction = evo_steric
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.1
                    logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])
                if 'r_m' in kwargs:
                    self.interaction_params['r_m'] = kwargs['r_m']
                else:
                    self.interaction_params['r_m'] = 1e-3
                    logger.info('mutation rate set to r_m = %s', self.interaction_params['r_m'])
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = .98 * self.interaction_params['r_b']
                    logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])
                if 'alpha' in kwargs:
                    self.interaction_params['alpha'] = kwargs['alpha']
                else:
                    self.interaction_params['alpha'] = 2.0
                    logger.info('steric interaction strength set to alpha = %s', self.interaction_params['alpha'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 3.0
                    logger.info('rest channel weight set to gamma = %s', self.interaction_params['gamma'])
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 512
                    logger.info('deme capacity set to capacity = %s', self.interaction_params['capacity'])
                self.init_families(type='homogeneous', mutation=True)
                self.props['family'][0] = 1  # there is no 'void' cell, so the cell w/ id = 0 also belongs to fam. 1
                self.family_props.update(r_b=[0] + [self.interaction_params['r_b']] * self.maxfamily)
                if 'fitness_increase' in kwargs:
                    self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                else:
                    self.interaction_params['fitness_increase'] = 1.1
                    logger.info('fitness increase for driver mutations set to %s', self.interaction_params['fitness_increase'])

            elif interaction == 'go_or_grow_glioblastoma':
                self.interaction = go_or_grow_glioblastoma
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    warn_user('This interaction requires a rest channel.')

                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    logger.info('node capacity set to %s', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    logger.info('std of kappa set to %s', self.interaction_params['kappa_std'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

                if 'r_m' in kwargs:
                    self.interaction_params['r_m'] = kwargs['r_m']
                else:
                    self.interaction_params['r_m'] = 1e-3
                    logger.info('mutation rate set to r_m = %s', self.interaction_params['r_m'])

                if 'fitness_increase' in kwargs:
                    self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                else:
                    self.interaction_params['fitness_increase'] = 1.1
                    logger.info('fitness increase for driver mutations set to %s', self.interaction_params['fitness_increase'])

                if 'theta' in kwargs:
                    self.interaction_params['theta'] = kwargs['theta']
                else:
                    self.interaction_params['theta'] = 0.5
                    logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])

                if 'r_b' in kwargs:
                    initial_r_b = kwargs['r_b']
                else:
                    initial_r_b = 0.2
                    logger.info('initial family birth rate set to r_b = %s', initial_r_b)

                if 'kappa' in kwargs:
                    initial_kappa = kwargs['kappa']
                else:
                    initial_kappa = 5.0
                    logger.info('initial family switch rate set to kappa = %s', initial_kappa)

                self.init_families(type='homogeneous', mutation=True)
                if self.props.get('family'):
                    self.props['family'][0] = 1
                self.family_props.update(r_b=[0.0] + [initial_r_b] * self.maxfamily)
                self.family_props.update(kappa=[0.0] + [initial_kappa] * self.maxfamily)
            else:
                raise ValueError(
                    "Unknown interaction {!r}. Implemented interactions: {}".format(
                        kwargs["interaction"], self.interactions
                    )
                )

        else:
            logger.info('Random walk interaction is used.')
            interaction = 'random_walk'
            self.interaction = random_walk
        self._validate_interaction_params()
        self._warn_if_nonlocal_ensemble_interaction(interaction)

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
