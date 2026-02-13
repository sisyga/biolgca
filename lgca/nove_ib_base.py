# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2024 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Identity-based LGCA without volume exclusion base class module.
"""

from abc import ABC, abstractmethod
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from matplotlib import cm
from numpy import random as npr
from sympy.utilities.iterables import multiset_permutations
from copy import copy, deepcopy
from lgca.plots import muller_plot
import warnings
from tqdm.auto import tqdm
from lgca.nove_base import NoVE_LGCA_base
from lgca.ib_base import IBLGCA_base

# configure matplotlib style
plt.style.use('default')


class NoVE_IBLGCA_base(NoVE_LGCA_base, IBLGCA_base, ABC):
    """
    Base class for identity-based LGCA without volume exclusion.
    """
    interactions = ['go_or_grow', 'birthdeath', 'randomwalk', 'steric_evolution']

    def __init__(self, nodes=None, dims=None, density=.1, restchannels=1, bc='periodic', seed=None, **kwargs):
        """
        Initialize class instance.
        :param nodes:
        :param l:
        :param restchannels:
        :param density:
        :param bc:
        :param r_int:
        :param kwargs:
        """
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.rng = npr.default_rng(seed=seed)
        self.props = {}
        self.length_checker = np.vectorize(len)
        self.set_bc(bc)
        self.interaction_params = {}
        if restchannels != 1:
            restchannels = 1
            warnings.warn("There can only be one rest channel in this LGCA class. Setting to 1 to prevent issues")
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self.init_coords()
        self.init_nodes(density, nodes=nodes)
        self.calc_max_label()
        self.update_dynamic_fields()
        self.mean_prop_t = {}
        self.set_interaction(**kwargs)


    def update_dynamic_fields(self):
        self.channel_pop = self.length_checker(self.nodes)  # population of a channel
        self.cell_density = self.channel_pop.sum(-1)  # population of a node

    def convert_int_to_ib(self, occ):
        """
        Convert an array of integers representing the occupation numbers of an lgca to an array consisting of lists of
        individual cell labels, starting at 'starting_id'. The length of each list corresponds to the entry in 'occ'.
        :param occ: array of occupation numbers. must match the lgca dimensions
        : param starting_id: int > 0
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
        """
        Distribute particles in the lattice according to a given density; can yield different cell numbers per lattice site
        :param density: particle density in the lattice: average number of particles per channel
        """
        density = npr.poisson(lam=density, size=self.dims + (self.K,))
        tempnodes = self.convert_int_to_ib(density)
        self.nodes[self.nonborder] = tempnodes
        self.maxlabel = density.sum()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        from lgca.nove_ib_interactions import randomwalk, birth, birthdeath, birthdeath_cancerdfe, go_or_grow, \
            evo_steric, go_or_grow_kappa
        from lgca.interactions import only_propagation
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction in ('random walk', 'random_walk', 'diffusion'):
                self.interaction = randomwalk
            elif interaction == 'only_propagation':
                self.interaction = only_propagation

            elif interaction in ('birth', 'birthdeath'):
                self.interaction = birthdeath if interaction == 'birthdeath' else birth
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('capacity of channel set to ', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                    if interaction == 'birth':
                        warnings.warn("Death rate defined but not used in birth interaction.")
                else:
                    if interaction == 'birthdeath':
                        self.interaction_params['r_d'] = 0.02
                        print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'std' in kwargs:
                    self.interaction_params['std'] = kwargs['std']
                else:
                    self.interaction_params['std'] = 0.01
                    print('standard deviation set to = ', self.interaction_params['std'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    print('Rest channel weight set to gamma =', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'birthdeath_cancerdfe':
                self.interaction = birthdeath_cancerdfe
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('capacity of channel set to ', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'p_d' in kwargs:
                    self.interaction_params['p_d'] = kwargs['p_d']
                else:
                    self.interaction_params['p_d'] = 1.4e-5  # from macfarlane 2014
                    print('probability of drivers set to = ', self.interaction_params['p_d'])

                if 'p_p' in kwargs:
                    self.interaction_params['p_p'] = kwargs['p_p']
                else:
                    self.interaction_params['p_p'] = 0.1  # from macfarlane 2014
                    print('probability of passengers set to = ', self.interaction_params['p_p'])

                if 's_d' in kwargs:
                    self.interaction_params['s_d'] = kwargs['s_d']
                else:
                    self.interaction_params['s_d'] = .1 * self.interaction_params['r_b']  # from macfarlane 2014
                    print('driver strength set to = ', self.interaction_params['s_d'])

                if 's_p' in kwargs:
                    self.interaction_params['s_p'] = kwargs['s_p']
                else:
                    self.interaction_params['s_p'] = .001 * self.interaction_params['r_b']  # from macfarlane 2014
                    print('passenger strength set to = ', self.interaction_params['s_p'])

                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    print('Rest channel weight set to gamma =', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'go_or_grow':
                self.interaction = go_or_grow
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    print('There must be exactly one rest channel for this interaction to work!')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('node capacity set to ', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    print('std of kappa set to', self.interaction_params['kappa_std'])

                if 'theta_std' in kwargs:
                    self.interaction_params['theta_std'] = kwargs['theta_std']
                else:
                    self.interaction_params['theta_std'] = 0.05
                    print('std of theta set to', self.interaction_params['theta_std'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])

                if 'kappa' in kwargs:
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    print('switch rate set to kappa = ', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    if hasattr(theta, '__iter__'):
                        self.interaction_params['theta'] = list(theta)
                    else:
                        self.interaction_params['theta'] = [theta] * (self.maxlabel + 1)
                else:
                    self.interaction_params['theta'] = [0.5] * (self.maxlabel + 1)
                    print('switch threshold set to theta = ', self.interaction_params['theta'][0])
                self.props.update(theta=np.array(self.interaction_params['theta']))

            elif interaction == 'go_or_grow_kappa':
                self.interaction = go_or_grow_kappa
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    print('There must be exactly one rest channel for this interaction to work!')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('node capacity set to ', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    print('std of kappa set to', self.interaction_params['kappa_std'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])

                if 'kappa' in kwargs:
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    print('switch rate set to kappa = ', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    self.interaction_params['theta'] = theta
                else:
                    self.interaction_params['theta'] = 0.5
                    print('switch threshold set to theta = ', self.interaction_params['theta'])

            elif interaction == 'steric_evolution':
                self.interaction = evo_steric
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.1
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                if 'r_m' in kwargs:
                    self.interaction_params['r_m'] = kwargs['r_m']
                else:
                    self.interaction_params['r_m'] = 1e-3
                    print('mutation rate set to r_m = ', self.interaction_params['r_m'])
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = .98 * self.interaction_params['r_b']
                    print('death rate set to r_d = ', self.interaction_params['r_d'])
                if 'alpha' in kwargs:
                    self.interaction_params['alpha'] = kwargs['alpha']
                else:
                    self.interaction_params['alpha'] = 2.0
                    print('steric interaction strength set to alpha = ', self.interaction_params['alpha'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 3.0
                    print('rest channel weight set to gamma = ', self.interaction_params['gamma'])
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 512
                    print('deme capacity set to capacity = ', self.interaction_params['capacity'])
                self.init_families(type='homogeneous', mutation=True)
                self.props['family'][0] = 1  # there is no 'void' cell, so the cell w/ id = 0 also belongs to fam. 1
                self.family_props.update(r_b=[0] + [self.interaction_params['r_b']] * self.maxfamily)
                if 'fitness_increase' in kwargs:
                    self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                else:
                    self.interaction_params['fitness_increase'] = 1.1
                    print('fitness increase for driver mutations set to ',
                          self.interaction_params['fitness_increase'])
            else:
                print('interaction', kwargs['interaction'], 'is not defined! Random walk used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = randomwalk

        else:
            print('Random walk interaction is used.')
            self.interaction = randomwalk

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, recordchanneldens=False,
                showprogress=True, recordfampop=False):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = get_arr_of_empty_lists((timesteps +1,) + self.dims + (self.K,))
            self.nodes_t[0, ...] = copy(self.nodes[self.nonborder])
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims, dtype=np.uint)
            self.dens_t[0, ...] = self.cell_density[self.nonborder]
        if recordchanneldens:
            self.channel_pop_t = np.zeros((timesteps + 1,) + self.dims + (self.K,), dtype=np.uint)
            self.channel_pop_t[0, ...] = self.channel_pop[self.nonborder]
        if recordfampop:
            from lgca.nove_ib_interactions import evo_steric
            # this needs to include all interactions that can increase the number of recorded families!
            if self.interaction in [evo_steric]:
                # if mutations are allowed, this is a list because it will be ragged due to increasing family numbers
                self.fam_pop_t = [self.calc_family_pop_alive()]
                is_mutating = True
            else:
                if 'family' not in self.props:
                    raise RuntimeError("Interaction does not deal with families, "
                                       "family population can therefore not be recorded.")
                # otherwise standard procedure
                self.fam_pop_t = np.zeros((timesteps + 1, self.maxfamily + 1))
                self.fam_pop_t[0, ...] = self.calc_family_pop_alive()
                is_mutating = False

        for t in tqdm(iterable=range(1, timesteps + 1), disable=1-showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = copy(self.nodes[self.nonborder])
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.nonborder]
            if recordfampop:
                if is_mutating:
                    # append to the ragged nested list
                    self.fam_pop_t.append(self.calc_family_pop_alive())
                else:
                    # standard procedure
                    try:
                        self.fam_pop_t[t, ...] = self.calc_family_pop_alive()
                    except ValueError as e:
                        raise ValueError("Number of families has increased, interaction must be included in the case " +
                                         "distinction for the recordfampop keyword in the IBLGCA base timeevo function!") from e
        if recordfampop and is_mutating:
            self.straighten_family_populations()

    def calc_max_label(self):
        cells = self.nodes.sum()
        if len(cells) == 0:
            self.maxlabel = None

        else: self.maxlabel = max(cells)

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
        proparray = prop[nodes.sum()]
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
        for key in self.props:
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

        if nodes_t is None:
            nodes_t = self.nodes_t

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        proparray = np.array(props[propname])
        prop_t = [proparray[nodes.sum()] for nodes in nodes_t]
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
        x = np.arange(tmax)
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

        propvals = [props[propname][id] for id in nodes.sum()]
        plt.figure(num=figindex, figsize=figsize)
        plt.hist(propvals, **kwargs)
        plt.xlabel('{}'.format(propname))
        plt.ylabel('Count')

    def plot_prop_2dhist(self, nodes=None, props=None, propnames=None, figindex=None, figsize=None, **kwargs):
        """
        Plot a 2d-histogram of two cell properties given by 'propnames' of all cells in 'nodes'. By default, the current
        lgca state is used and the first two properties are shown.
        :param nodes:
        :param props:
        :param propnames:
        :param figindex:
        :param figsize:
        :param kwargs:
        :return:
        """
        import seaborn as sns
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propnames is None:
            names = iter(props)
            propname1 = next(names)
            propname2 = next(names)

        ids = [id for id in nodes.sum()]
        propvals1, propvals2 = [props[propname1][id] for id in ids], [props[propname2][id] for id in ids]
        # plt.figure(num=figindex, figsize=figsize)
        sns.jointplot(x=propvals1, y=propvals2, marginal_ticks=True, kind='hist', **kwargs)
        plt.xlabel('{}'.format(propname1))
        plt.ylabel('{}'.format(propname2))

    def calc_family_pop_alive(self):
        """
        Calculate how many cells of each family are alive.
        :returns: np.ndarray fam_pop_array - array of family population counts indexed by family ID
        """
        if 'family' not in self.props:
            raise RuntimeError("Family properties are not recorded by the LGCA, choose suitable interaction.")

        cells_alive = np.array(self.nodes[self.nonborder].sum()) # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
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
        cells_alive = np.array(self.nodes[self.nonborder].sum(-1))  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        return np.unique(cell_fam_alive) # remove duplicate entries
