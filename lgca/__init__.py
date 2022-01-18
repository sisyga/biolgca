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
biolgca docstring
"""


def get_lgca(geometry='hex', ib=False, ve=True, **kwargs):
    if not ve:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import NoVE_LGCA_1D
            return NoVE_LGCA_1D(**kwargs, ve=ve)
        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import NoVE_LGCA_Square
            return NoVE_LGCA_Square (**kwargs, ve=ve)
        elif geometry in ['hex']:
            from lgca.lgca_hex import NoVE_LGCA_Hex
            return NoVE_LGCA_Hex (**kwargs)


    if ib:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import IBLGCA_1D
            return IBLGCA_1D(**kwargs)
        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import IBLGCA_Square
            return IBLGCA_Square(**kwargs)

        else:
            from lgca.lgca_hex import IBLGCA_Hex
            return IBLGCA_Hex(**kwargs)

    else:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import LGCA_1D
            return LGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import LGCA_Square
            return LGCA_Square(**kwargs)

        else:
            from lgca.lgca_hex import LGCA_Hex
            return LGCA_Hex(**kwargs)
