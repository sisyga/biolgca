"""Warnings that point at the user's code instead of library internals."""

from __future__ import annotations

import os
import sys
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def warn_user(message, category=UserWarning):
    """Warn and point at the first caller outside the lgca package."""
    if sys.version_info >= (3, 12):
        warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_DIR,))
    else:
        warnings.warn(message, category, stacklevel=3)
