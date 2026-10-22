"""Compatibility reexports for base classes moved to dedicated modules.

Deprecated: import the classes from :mod:`lgca.ib_base`, :mod:`lgca.nove_base` and :mod:`lgca.nove_ib_base`.
"""

from ._warnings import warn_user
from .ib_base import IBLGCA_base
from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists
from .nove_base import NoVE_LGCA_base
from .nove_ib_base import NoVE_IBLGCA_base

warn_user(
    "lgca.base_extensions is deprecated and will be removed; import IBLGCA_base from lgca.ib_base, "
    "NoVE_LGCA_base from lgca.nove_base and NoVE_IBLGCA_base from lgca.nove_ib_base",
    DeprecationWarning,
)

__all__ = [
    "IBLGCA_base",
    "NoVE_LGCA_base",
    "NoVE_IBLGCA_base",
    "get_arr_of_empty_lists",
    "_copy_arr_of_lists",
]
