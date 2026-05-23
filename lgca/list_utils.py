"""Helpers for object arrays containing per-channel cell-label lists."""

import numpy as np


ufunclist = np.frompyfunc(list, 0, 1)
_copylist = np.frompyfunc(list.copy, 1, 1)


def get_arr_of_empty_lists(dims):
    """
    Create a numpy array of dimensions 'dims' that is filled with empty lists.

    Parameters
    ----------
    dims: tuple, corresponding to shape of the array.

    Returns
    -------
    :py:class:`numpy.ndarray`

    """
    return ufunclist(np.empty(dims, dtype=object))


def _copy_arr_of_lists(arr):
    """Copy each list stored in an object array."""
    return _copylist(arr)


__all__ = ["get_arr_of_empty_lists", "_copy_arr_of_lists"]
