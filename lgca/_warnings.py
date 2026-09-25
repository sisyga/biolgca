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
        return
    # Python 3.11 has no skip_file_prefixes: count the frames up to the first one outside the package.
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(message, category, stacklevel=stacklevel)
