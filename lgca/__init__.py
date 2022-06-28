# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
biolgca is a Python package for simulating different types of lattice-gas
cellular automata (LGCA) in the biological context. It is under active development.

It is tightly coupled to the theoretical method presented by Deutsch et al. [1]_.
The get_lgca function returns an LGCA object of the requested type with the
given initial conditions. It can simulate for a given number of timesteps and has
different plotting functions for analysis, depending on the geometry. The
interaction function can be chosen from built-in ones or defined by the user.
Currently, LGCA with and without volume exclusion and identity-based LGCA are
supported on 1D, 2D square and 2D hexagonal lattices.

References
----------
.. [1] Deutsch A, Nava-Sedeño JM, Syga S, Hatzikirou H (2021) BIO-LGCA: A cellular
    automaton modelling class for analysing collective cell migration.
    PLoS Comput Biol 17(6): e1009066. https://doi.org/10.1371/journal.pcbi.1009066

"""


def get_lgca(geometry: str='hex', ib: bool=False, ve: bool=True, **kwargs):
    """
    Build an LGCA with the specified geometry and initial conditions. Choose the correct LGCA subclass
    from the package and pass remaining keyword parameters on to it for initialization.

    Parameters
    ----------
    geometry : {'hex', 'square', 'lin'}, default='hex'
        Lattice geometry. Supported are 1D, 2D square and 2D hexagonal lattices.

        Aliases: 1D: ``'1D', '1d', 'linear'``; 2D square: ``'sq', 'rect', 'rectangular'``;
        2D hexagonal: ``'hexagonal', 'hx'``.
    ib : bool, default=False
        If the LGCA should be identity-based (every particle can have individual properties).
    ve : bool, default=True
        If the LGCA should comply with the volume exclusion principle (only one particle per channel).
    **kwargs : dict
        Keyword arguments for dimensions, initial conditions and interaction. Used by the constructor of the LGCA subclass.

    Returns
    -------
    lgca : subclass of :py:class:`base.LGCA_base` instance
        LGCA simulator object.

    See Also
    --------
    base.LGCA_base.set_bc : Set boundary conditions. Processes `**kwargs`.
    base.LGCA_base.set_dims : Set the lattice geometry. Processes `**kwargs`.
    base.LGCA_base.init_nodes : Initialize the lattice.  Processes `**kwargs`.
    base.LGCA_base.set_interaction : Set the interaction and corresponding parameters. Processes `**kwargs`.

    Notes
    -----
    The function mediates between user and package. It picks the correct LGCA subclass
    and initializes an instance of it. Allowed types and values of keyword arguments
    for initialization may vary. Details can be found in the documentation of the subclasses.

    How to navigate: Subclasses are structured as follows (omitting geometry inheritance):

    - :py:class:`lgca.base.LGCA_base`: classical LGCA
        - :py:class:`lgca.lgca_1d.LGCA_1D`
        - :py:class:`lgca.lgca_square.LGCA_Square`
            - :py:class:`lgca.lgca_hex.LGCA_Hex`
        - :py:class:`lgca.base.IBLGCA_base`: identity-based LGCA
            - :py:class:`lgca.lgca_1d.IBLGCA_1D`
            - :py:class:`lgca.lgca_square.IBLGCA_Square`
                - :py:class:`lgca.lgca_hex.IBLGCA_Hex`
        - :py:class:`lgca.base.NoVE_LGCA_base`: LGCA without volume exclusion
            - :py:class:`lgca.lgca_1d.NoVE_LGCA_1D`
            - :py:class:`lgca.lgca_square.NoVE_LGCA_Square`
                - :py:class:`lgca.lgca_hex.NoVE_LGCA_Hex`

    Examples
    --------
    Request a classical LGCA with a hexagonal lattice and a random walk interaction.

    >>> from lgca import get_lgca
    >>> lgca = get_lgca(test='unused')
    Random walk interaction is used.
    {'test': 'unused'}

    Used default values for interactions are printed to the terminal.
    Unused keywords are printed as a dictionary below that.

    Request an identity-based LGCA in a linear geometry with a birth interaction.

    >>> lgca = get_lgca(ib=True, geometry='1d', interaction='birth')
    Birth rate set to r_b = 0.2
    Standard deviation set to std = 0.01
    Max. birth rate set to a_max = 1.0

    The returned LGCA object can then be used to simulate.

    >>> # simulate for 50 timesteps
    >>> lgca.timeevo(timesteps=50)
    Progress: [####################] 100% Done...

    """
    if not ve and not ib:
        if geometry in ['1d', '1D', 'lin', 'linear']:
            from lgca.lgca_1d import NoVE_LGCA_1D
            return NoVE_LGCA_1D(**kwargs, ve=ve)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import NoVE_LGCA_Square
            return NoVE_LGCA_Square(**kwargs, ve=ve)

        elif geometry in ['hex', 'hx', 'hexagonal']:
            from lgca.lgca_hex import NoVE_LGCA_Hex
            return NoVE_LGCA_Hex(**kwargs)

        else:
            raise ValueError("Geometry specification is unknown. Try: '1d', '1D', 'lin', "
                             "'linear', 'square', 'sq', 'rect', 'rectangular', 'hex', 'hx' or 'hexagonal'.")

    if ib and ve:
        if geometry in ['1d', '1D', 'lin', 'linear']:
            from lgca.lgca_1d import IBLGCA_1D
            return IBLGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import IBLGCA_Square
            return IBLGCA_Square(**kwargs)

        elif geometry in ['hex', 'hx', 'hexagonal']:
            from lgca.lgca_hex import IBLGCA_Hex
            return IBLGCA_Hex(**kwargs)

        else:
            raise ValueError("Geometry specification is unknown. Try: '1d', 'lin', "
                             "'linear', 'square', 'sq', 'rect', 'rectangular', 'hex', 'hx' or 'hexagonal'.")

    if not ve and ib:
        if geometry in ['1d', '1D', 'lin', 'linear']:
            from lgca.lgca_1d import NoVE_IBLGCA_1D
            return NoVE_IBLGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import NoVE_IBLGCA_Square
            return NoVE_IBLGCA_Square(**kwargs)

        elif geometry in ['hex', 'hx', 'hexagonal']:
            from lgca.lgca_hex import NoVE_IBLGCA_Hex
            return NoVE_IBLGCA_Hex(**kwargs)

        else:
            raise ValueError("Geometry specification is unknown. Try: '1d', 'lin', "
                             "'linear', 'square', 'sq', 'rect', 'rectangular', 'hex', 'hx' or 'hexagonal'.")

    else:
        if geometry in ['1d', '1D', 'lin', 'linear']:
            from lgca.lgca_1d import LGCA_1D
            return LGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import LGCA_Square
            return LGCA_Square(**kwargs)

        elif geometry in ['hex', 'hx', 'hexagonal']:
            from lgca.lgca_hex import LGCA_Hex
            return LGCA_Hex(**kwargs)

        else:
            raise ValueError("Geometry specification is unknown. Try: '1d', 'lin', "
                             "'linear', 'square', 'sq', 'rect', 'rectangular', 'hex', 'hx' or 'hexagonal'.")
