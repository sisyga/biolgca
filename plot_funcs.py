import string
from itertools import cycle
def label_axes(axes, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`
    Parameters
    ----------
    axes : iterable of axes to label
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (0., 1.1)
    for ax, lab in zip(axes, labels):
        ax.annotate(lab, xy=loc, ha='right', weight='bold', xycoords='axes fraction', **kwargs)