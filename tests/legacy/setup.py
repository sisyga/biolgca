"""The ``set_interaction`` methods of the legacy model classes, as functions of the model.

Copied from biolgca before the legacy interactions became stacks of rules
(``lgca.legacy_names``). They set ``lgca.interaction`` to a function of
``tests.legacy`` and fill ``lgca.interaction_params`` as before, so that
tests can compare the rules with the legacy interactions.
"""
# ruff: noqa

import logging

import numpy as np

from lgca._warnings import warn_user
from lgca.base import _validate_vector_field_shape, calc_nematic_tensor

logger = logging.getLogger("lgca")


_MULTISPECIES = {True: {"random_walk", "only_propagation", "excitable_medium_ms"},
                 False: {"birth", "birthdeath", "go_or_grow", "only_propagation"}}


def _require_legacy_interaction(lgca, interaction) -> None:
    from lgca.nove_base import NoVE_LGCA_base

    name = None if interaction is None else str(interaction).replace(" ", "_")
    if name in _MULTISPECIES[not isinstance(lgca, NoVE_LGCA_base)]:
        return
    raise ValueError(f"Interaction {name!r} is not supported for this legacy multispecies model")



def set_classical(self, **kwargs):
    """
    Set the interaction rule and respective needed parameters.

    Set :py:attr:`self.interaction` and possibly add entries in :py:attr:`self.interaction_params`.

    Parameters
    ----------
    kwargs['interaction'] : str or callable, default='random_walk'
        Name of the predefined interaction in :py:mod:`lgca.interactions`, or a
        function ``f(lgca)`` that updates ``lgca.nodes`` in place of the
        interaction step. A function reads its parameters from
        ``lgca.interaction_params``, which receives all other keyword arguments.
    **kwargs
        Interaction parameters.

    """
    from .interactions import go_or_grow, go_or_rest, birth, alignment, persistent_walk, chemotaxis, \
            contact_guidance, nematic, aggregation, random_walk, birthdeath, excitable_medium, \
            only_propagation
    from .ms_interactions import excitable_medium_ms
    if 'interaction' in kwargs:
        interaction = kwargs['interaction'].replace(" ", "_")
        if interaction == 'go_or_grow':
            self.interaction = go_or_grow
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
                self.interaction_params['kappa'] = kwargs['kappa']
            else:
                self.interaction_params['kappa'] = 5.
                logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'])
            if 'theta' in kwargs:
                self.interaction_params['theta'] = kwargs['theta']
            else:
                self.interaction_params['theta'] = 0.75
                logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])
            if self.restchannels < 2:
                warn_user('Not enough rest channels - system will die out.')

        elif interaction == 'go_or_rest':
            self.interaction = go_or_rest
            if 'kappa' in kwargs:
                self.interaction_params['kappa'] = kwargs['kappa']
            else:
                self.interaction_params['kappa'] = 5.
                logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'])
            if 'theta' in kwargs:
                self.interaction_params['theta'] = kwargs['theta']
            else:
                self.interaction_params['theta'] = 0.75
                logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])
            if self.restchannels < 2:
                warn_user('Not enough rest channels - system will die out.')

        elif interaction == 'go_and_grow':
            self.interaction = birth
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])

        elif interaction == 'alignment':
            self.interaction = alignment
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

        elif interaction == 'persistent_motion':
            self.interaction = persistent_walk
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

        elif interaction == 'chemotaxis':
            self.interaction = chemotaxis
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

            if 'gradient' in kwargs:
                self.interaction_params['gradient_field'] = _validate_vector_field_shape(
                    kwargs['gradient'],
                    self.nodes.shape[:-1] + (self.c.shape[0],),
                    "gradient",
                )
            else:
                if len(self.dims) == 2:
                    x_source = self.xcoords.mean()
                    y_source = self.ycoords.mean()
                    rx = self.xcoords - x_source
                    ry = self.ycoords - y_source
                    r = np.sqrt(rx ** 2 + ry ** 2)
                    self.concentration = np.exp(-2 * r / self.ly)
                    self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                     'reflect'))
                elif len(self.dims) == 1:
                    source = self.l / 2
                    r = abs(self.xcoords - source)
                    self.concentration = np.exp(-2 * r / self.l)
                    self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                     'reflect'))
                    self.interaction_params['gradient_field'] /= self.interaction_params['gradient_field'].max()

                elif len(self.dims) == 3:
                    x_source = self.xcoords.mean()
                    y_source = self.ycoords.mean()
                    z_source = self.zcoords.mean()
                    rx = self.xcoords - x_source
                    ry = self.ycoords - y_source
                    rz = self.zcoords - z_source
                    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
                    self.concentration = np.exp(-2 * r / self.ly)
                    self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                     'reflect'))


        elif interaction == 'contact_guidance':
            if len(self.dims) != 2:
                raise ValueError("contact_guidance is not supported for this geometry.")
            self.interaction = contact_guidance
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

            if 'director' in kwargs:
                self.interaction_params['gradient_field'] = _validate_vector_field_shape(
                    kwargs['director'],
                    self.nodes.shape[:-1] + (2,),
                    "director",
                )
            else:
                self.interaction_params['gradient_field'] = np.zeros((self.lx + 2 * self.r_int,
                                                                      self.ly + 2 * self.r_int, 2))
                self.interaction_params['gradient_field'][..., 0] = 1
            self.guiding_tensor = calc_nematic_tensor(self.interaction_params['gradient_field'])
            if self.velocitychannels < 4:
                warn_user('Nematic interaction undefined in 1D.')

        elif interaction == 'nematic':
            self.interaction = nematic
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

        elif interaction == 'aggregation':
            self.interaction = aggregation
            self.calc_permutations()

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])

        elif interaction == 'random_walk':
            self.interaction = random_walk

        elif interaction == 'birth':
            self.interaction = birth
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])

        elif interaction == 'birthdeath':
            self.interaction = birthdeath
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])

            if 'r_d' in kwargs:
                self.interaction_params['r_d'] = kwargs['r_d']
            else:
                self.interaction_params['r_d'] = 0.05
                logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])

        elif interaction == 'excitable_medium':
            self.interaction = excitable_medium
            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']

            else:
                self.interaction_params['beta'] = .05
                logger.info('alignment sensitivity set to beta = %s', self.interaction_params['beta'])

            if 'alpha' in kwargs:
                self.interaction_params['alpha'] = kwargs['alpha']
            else:
                self.interaction_params['alpha'] = 1.
                logger.info('aggregation sensitivity set to alpha = %s', self.interaction_params['alpha'])

            if 'N' in kwargs:
                self.interaction_params['N'] = kwargs['N']
            else:
                self.interaction_params['N'] = 50
                logger.info('repetition of fast reaction set to N = %s', self.interaction_params['N'])

        elif interaction == 'excitable_medium_ms':
            if getattr(self, "n_species", 1) != 2:
                raise ValueError("excitable_medium_ms requires a multi-species LGCA with exactly two species.")
            if self.restchannels < 1:
                raise ValueError("excitable_medium_ms requires at least one rest channel.")
            self.interaction = excitable_medium_ms
            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = .05
                logger.info('alignment sensitivity set to beta = %s', self.interaction_params['beta'])

            if 'alpha' in kwargs:
                self.interaction_params['alpha'] = kwargs['alpha']
            else:
                self.interaction_params['alpha'] = 1.
                logger.info('aggregation sensitivity set to alpha = %s', self.interaction_params['alpha'])

            if 'N' in kwargs:
                self.interaction_params['N'] = kwargs['N']
            else:
                self.interaction_params['N'] = 50
                logger.info('repetition of fast reaction set to N = %s', self.interaction_params['N'])

        elif interaction == 'only_propagation':
            self.interaction = only_propagation

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



def set_nove(self, **kwargs):
    from .nove_interactions import dd_alignment, di_alignment, go_or_grow, go_or_rest, random_walk
    from .interactions import only_propagation
    # configure interaction
    if 'interaction' in kwargs:
        interaction = kwargs['interaction']
        if interaction == 'random_walk':
            self.interaction = random_walk
        # density-dependent interaction rule
        elif interaction == 'dd_alignment':
            if self.restchannels > 0:
                raise RuntimeError("Rest channels ({:d}) defined, interaction will crash! Set number of"
                                   " rest channels to 0 with restchannels keyword.".format(self.restchannels))
            self.interaction = dd_alignment

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])
            if 'include_center' in kwargs:
                self.interaction_params['nb_include_center'] = kwargs['include_center']
            else:
                self.interaction_params['nb_include_center'] = False
                logger.info('neighbourhood set to exclude the central node')
        # density-independent alignment rule
        elif interaction == 'di_alignment':
            if self.restchannels > 0:
                raise RuntimeError("Rest channels ({:d}) defined, interaction will crash! Set number of"
                                   " rest channels to 0 with restchannels keyword.".format(self.restchannels))
            self.interaction = di_alignment
            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])
            if 'include_center' in kwargs:
                self.interaction_params['nb_include_center'] = kwargs['include_center']
            else:
                self.interaction_params['nb_include_center'] = False
                logger.info('neighbourhood set to exclude the central node')
        elif interaction == 'go_or_grow':
            if self.restchannels < 1:
                raise RuntimeError("No rest channels ({:d}) defined, interaction cannot be performed! Set number of"
                                   " rest channels with restchannels keyword.".format(self.restchannels))
            self.interaction = go_or_grow
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
                self.interaction_params['kappa'] = kwargs['kappa']
            else:
                self.interaction_params['kappa'] = 5.
                logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'])
            if 'theta' in kwargs:
                self.interaction_params['theta'] = kwargs['theta']
            else:
                self.interaction_params['theta'] = 0.75
                logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])
        elif interaction == 'go_or_rest':
            if self.restchannels < 1:
                raise RuntimeError(
                    "No rest channels ({:d}) defined, interaction cannot be performed! Set number of rest "
                    "channels with restchannels keyword.".format(
                        self.restchannels))

            self.interaction = go_or_rest
            if 'kappa' in kwargs:
                self.interaction_params['kappa'] = kwargs['kappa']
            else:
                self.interaction_params['kappa'] = 5.
                logger.info('switch rate set to kappa = %s', self.interaction_params['kappa'])
            if 'theta' in kwargs:
                self.interaction_params['theta'] = kwargs['theta']
            else:
                self.interaction_params['theta'] = 0.75
                logger.info('switch threshold set to theta = %s', self.interaction_params['theta'])

        elif interaction == 'only_propagation':
            self.interaction = only_propagation

        else:
            raise ValueError(
                "Unknown interaction {!r}. Implemented interactions: {}".format(
                    kwargs["interaction"], self.interactions
                )
            )

    # if nothing is specified, use density-dependent interaction rule
    else:
        logger.info('Density-dependent alignment interaction is used.')
        interaction = 'dd_alignment'
        self.interaction = dd_alignment

        if self.restchannels > 0:
            raise RuntimeError("Rest channels ({:d}) defined, interaction will crash! Set number of"
                               " rest channels to 0 with restchannels keyword.".format(self.restchannels))

        if 'beta' in kwargs:
            self.interaction_params['beta'] = kwargs['beta']
        else:
            self.interaction_params['beta'] = 2.
            logger.info('sensitivity set to beta = %s', self.interaction_params['beta'])
        if 'include_center' in kwargs:
            self.interaction_params['nb_include_center'] = kwargs['include_center']
        else:
            self.interaction_params['nb_include_center'] = False
            logger.info('neighbourhood set to exclude the central node')
    self._validate_interaction_params()
    self._warn_if_nonlocal_ensemble_interaction(interaction)



def set_ib(self, **kwargs):
    """
    Set the interaction rule and respective needed parameters.

    Set :py:attr:`self.interaction` and possibly add entries in :py:attr:`self.interaction_params` and
    :py:attr:`self.props`.
    If inheritance is involved, initialize :py:attr:`self.family_props` and :py:attr:`self.maxfamily`.
    Do not use this to specify a custom interaction. In order to do this (as of now), :py:attr:`self.interaction`
    and :py:attr:`self.interaction_params` must be manipulated directly from an external script.

    Parameters
    ----------
    kwargs['interaction'] : str, default='random_walk'
        Name of the predefined interaction in :py:mod:`lgca.interactions`.
    **kwargs
        Interaction parameters.

    """
    from .ib_interactions import random_walk, birth, birthdeath, birthdeath_discrete, go_or_grow, \
        go_and_grow_mutations
    from .interactions import only_propagation
    if 'interaction' in kwargs:
        interaction = kwargs['interaction']
        if interaction == 'birth':
            self.interaction = birth
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('Birth rate set to r_b = %s', self.interaction_params['r_b'])
            self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
            if 'std' in kwargs:
                self.interaction_params['std'] = kwargs['std']
            else:
                self.interaction_params['std'] = 0.01
                logger.info('Standard deviation set to std = %s', self.interaction_params['std'])
            if 'a_max' in kwargs:
                self.interaction_params['a_max'] = kwargs['a_max']
            else:
                self.interaction_params['a_max'] = 1.
                logger.info('Max. birth rate set to a_max = %s', self.interaction_params['a_max'])

        elif interaction == 'birthdeath' or interaction == 'go_and_grow':
            self.interaction = birthdeath
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])
            self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
            if 'r_d' in kwargs:
                self.interaction_params['r_d'] = kwargs['r_d']
            else:
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

            if 'track_inheritance' in kwargs:
                self.interaction_params['track_inheritance'] = kwargs['track_inheritance']
            else:
                self.interaction_params['track_inheritance'] = False
                logger.info('Family relationships not tracked.')
            if self.interaction_params['track_inheritance']:
                self.init_families(type='heterogeneous', mutation=False)

        elif interaction == 'birthdeath_discrete':
            self.interaction = birthdeath_discrete
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.2
                logger.info('Birth rate set to r_b = %s', self.interaction_params['r_b'])

            self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
            if 'r_d' in kwargs:
                self.interaction_params['r_d'] = kwargs['r_d']
            else:
                self.interaction_params['r_d'] = 0.02
                logger.info('Death rate set to r_d = %s', self.interaction_params['r_d'])

            if 'drb' in kwargs:
                self.interaction_params['drb'] = kwargs['drb']
            else:
                self.interaction_params['drb'] = 0.01
                logger.info('Delta r_b set to = %s', self.interaction_params['drb'])
            if 'a_max' in kwargs:
                self.interaction_params['a_max'] = kwargs['a_max']
            else:
                self.interaction_params['a_max'] = 1.
                logger.info('Max. birth rate set to a_max = %s', self.interaction_params['a_max'])

            if 'pmut' in kwargs:
                self.interaction_params['pmut'] = kwargs['pmut']
            else:
                self.interaction_params['pmut'] = 0.1
                logger.info('Mutation probability set to p_mut = %s', self.interaction_params['pmut'])

        elif interaction == 'go_or_grow':
            self.interaction = go_or_grow
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
                try:
                    self.interaction_params['kappa'] = list(kappa)
                except TypeError:
                    self.interaction_params['kappa'] = [kappa] * self.maxlabel
            else:
                self.interaction_params['kappa'] = [5.] * self.maxlabel
                logger.info('switch rate set to kappa = %s', 5.0)
            self.props.update(kappa=[0.] + self.interaction_params['kappa'])
            if 'theta' in kwargs:
                theta = kwargs['theta']
                try:
                    self.interaction_params['theta'] = list(theta)
                except TypeError:
                    self.interaction_params['theta'] = [theta] * self.maxlabel
            else:
                self.interaction_params['theta'] = [0.75] * self.maxlabel
                logger.info('switch threshold set to theta = %s', 0.75)
            self.props.update(theta=[0.] + self.interaction_params['theta'])
            if 'kappa_std' in kwargs:
                self.interaction_params['kappa_std'] = kwargs['kappa_std']
            else:
                self.interaction_params['kappa_std'] = 0.2
                logger.info('Standard deviation for kappa mutation set to %s', self.interaction_params['kappa_std'])
            if 'theta_std' in kwargs:
                self.interaction_params['theta_std'] = kwargs['theta_std']
            else:
                self.interaction_params['theta_std'] = 0.05
                logger.info('Standard deviation for theta mutation set to %s', self.interaction_params['theta_std'])

            if self.restchannels < 2:
                warn_user('Not enough rest channels - system will die out.')

        elif interaction == 'random_walk':
            self.interaction = random_walk

        elif interaction == 'only_propagation':
            self.interaction = only_propagation

        elif interaction == 'go_and_grow_mutations':
            self.interaction = go_and_grow_mutations
            if 'effect' in kwargs:
                self.interaction_params['effect'] = kwargs['effect']
            else:
                self.interaction_params['effect'] = 'passenger_mutation'
                logger.info('fitness effect set to passenger mutation, rb=const.')
            if 'r_int' in kwargs:
                self.set_r_int(kwargs['r_int'])
            if 'r_b' in kwargs:
                self.interaction_params['r_b'] = kwargs['r_b']
            else:
                self.interaction_params['r_b'] = 0.5
                logger.info('birth rate set to r_b = %s', self.interaction_params['r_b'])
            if 'r_m' in kwargs:
                self.interaction_params['r_m'] = kwargs['r_m']
            else:
                self.interaction_params['r_m'] = 0.001
                logger.info('mutation rate set to r_m = %s', self.interaction_params['r_m'])
            if 'r_d' in kwargs:
                self.interaction_params['r_d'] = kwargs['r_d']
            else:
                self.interaction_params['r_d'] = 0.02
                logger.info('death rate set to r_d = %s', self.interaction_params['r_d'])
            self.init_families(type='homogeneous', mutation=True)
            if self.interaction_params['effect'] == 'driver_mutation':
                self.family_props.update(r_b=[0] + [self.interaction_params['r_b']] * self.maxfamily)
                if 'fitness_increase' in kwargs:
                    self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                else:
                    self.interaction_params['fitness_increase'] = 1.1
                    logger.info('fitness increase for driver mutations set to %s', self.interaction_params['fitness_increase'])
        else:
            raise ValueError(
                "Unknown interaction {!r}. Implemented interactions: {}".format(
                    kwargs["interaction"], self.interactions
                )
            )
    else:
        interaction = 'random_walk'
        self.interaction = random_walk
    self._validate_interaction_params()
    self._warn_if_nonlocal_ensemble_interaction(interaction)



def set_nove_ib(self, **kwargs):
    from .nove_ib_interactions import random_walk, birth, birthdeath, birthdeath_cancerdfe, go_or_grow, \
        evo_steric, go_or_grow_kappa, go_or_grow_glioblastoma
    from .interactions import only_propagation
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



def set_multispecies(self, **kwargs):
    _require_legacy_interaction(self, kwargs.get("interaction", "random_walk"))
    set_classical(self, **kwargs)



def set_multispecies_nove(self, **kwargs):
    if kwargs.get("interaction") == "excitable_medium_ms":
        raise ValueError("excitable_medium_ms requires volume exclusion.")
    _require_legacy_interaction(self, kwargs.get("interaction"))
    interaction = kwargs.get("interaction", "").replace(" ", "_")
    if interaction in {"birth", "birthdeath", "go_or_grow"}:
        from .ms_interactions import (
            _resolve_mutation_matrix,
            _validate_mutation_matrix,
            _validate_species_vector,
            birth,
            birthdeath,
            go_or_grow,
        )

        self.interaction_params["capacity"] = self.capacity
        if "mutation_matrix" in kwargs:
            self.interaction_params["mutation_matrix"] = _validate_mutation_matrix(
                self, kwargs["mutation_matrix"]
            )

        if interaction in {"birth", "birthdeath"}:
            self.interaction = birthdeath if interaction == "birthdeath" else birth
            r_b = kwargs.get("r_b", 0.2)
            self.interaction_params["r_b"] = _validate_species_vector(self, "r_b", r_b)
            if "std" in kwargs:
                self.interaction_params["std"] = kwargs["std"]
            self.interaction_params["mutation_matrix"] = _validate_mutation_matrix(
                self, _resolve_mutation_matrix(self, trait_name="r_b", std_name="std")
            )
            if interaction == "birthdeath":
                self.interaction_params["r_d"] = kwargs.get("r_d", 0.02)
            elif "r_d" in kwargs:
                warn_user("Death rate defined but not used in birth interaction.")

            gamma = kwargs.get("gamma", 0.0)
            self.interaction_params["gamma"] = gamma
            z = self.velocitychannels + np.exp(gamma) * self.restchannels
            self.channel_weights = np.array(
                [1.0 / z] * self.velocitychannels
                + [np.exp(gamma) / z] * self.restchannels
            )

        else:
            if self.restchannels != 1:
                raise ValueError("go_or_grow requires exactly one rest channel.")
            self.interaction = go_or_grow
            self.interaction_params["r_d"] = kwargs.get("r_d", 0.01)
            self.interaction_params["r_b"] = kwargs.get("r_b", 0.2)
            self.interaction_params["kappa"] = _validate_species_vector(
                self, "kappa", kwargs.get("kappa", 5.0)
            )
            self.interaction_params["theta"] = kwargs.get("theta", 0.5)
            if "kappa_std" in kwargs:
                self.interaction_params["kappa_std"] = kwargs["kappa_std"]
            self.interaction_params["mutation_matrix"] = _validate_mutation_matrix(
                self, _resolve_mutation_matrix(self, trait_name="kappa", std_name="kappa_std")
            )

        self._validate_interaction_params()
        self._warn_if_nonlocal_ensemble_interaction(interaction)
        return
    set_nove(self, **kwargs)



def set_legacy_interaction(lgca, **kwargs):
    """Give ``lgca`` the legacy interaction ``kwargs["interaction"]`` with its legacy setup."""
    from lgca.ib_base import IBLGCA_base
    from lgca.multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base
    from lgca.nove_base import NoVE_LGCA_base
    from lgca.nove_ib_base import NoVE_IBLGCA_base

    lgca.interaction_params = {}
    for base, setter in ((NoVE_IBLGCA_base, set_nove_ib), (IBLGCA_base, set_ib),
                         (MultiSpeciesNoVE_LGCA_base, set_multispecies_nove), (MultiSpeciesLGCA_base, set_multispecies),
                         (NoVE_LGCA_base, set_nove)):
        if isinstance(lgca, base):
            return setter(lgca, **kwargs)
    return set_classical(lgca, **kwargs)
