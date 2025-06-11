# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.

"""Utilities for plotting properties and outputs of all LGCA types."""

import numpy as np
import random
from itertools import cycle
try:  # optional plotting dependencies
    import matplotlib.colors as mplcolors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import FuncFormatter
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover - handled at runtime
    from lgca.base import _MissingPlotLib

    mplcolors = plt = ticker = FuncFormatter = cm = make_axes_locatable = _MissingPlotLib(
        "matplotlib"
    )


class IdentityColourMapper:
    """
    Maps from the family ID to a matplotlib colour that encodes the family identity.
    Prevents families from sharing a colour with their children, siblings and parent
    """

    def __init__(self, cmap, children_nlist, parent_list):
        """
        Initialise an instance of an identity colour mapper.
        :param cmap: string, list or matplotlib.colors.ListedColormap - optionally ordered colormap to choose the colors from.
                     cmap can be 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                     'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
                     'tab20b', 'tab20c' or 'turbo', but the fine grained colormaps make families hard to distinguish
        :param children_nlist: family tree as nested list of child family IDs, index=parent of the children
        :param parent_list: list of parent family IDs, index=child of this parent
        """
        # dictionary to save already requested family colours
        self.fam_colour = {}

        # colours to choose from
        if cmap is not None:
            if type(cmap) == str:
                if not isinstance(plt.cm.get_cmap(cmap), mplcolors.ListedColormap):
                    raise TypeError("Requested colormap must be a matplotlib.colors.ListedColormap!")
                    # it must be a ListedColormap and not just a Colormap as of now to construct the cycler
                self.cmap = plt.cm.get_cmap(cmap)
            else:
                if not isinstance(cmap, mplcolors.ListedColormap) and not isinstance(cmap, list):
                    raise TypeError("Colormap must be string (colormap name), list or matplotlib.colors.ListedColormap!")
                    # it must be a ListedColormap and not just a Colormap as of now to construct the cycler
                self.cmap = cmap
        else:
            self.cmap = plt.cm.get_cmap('tab20')
            # deprecated: shuffle colours because similar colours for non-closely related families can be confusing
            #cols = list(self.cmap.colors)
            #random.shuffle(cols)
            #self.cmap.colors = tuple(cols)

        # cycler to iterate through colours
        if type(self.cmap) == list:
            self.col_cycler = cycle(self.cmap)
            self.cycle_len = len(self.cmap)
        elif isinstance(self.cmap, mplcolors.ListedColormap):
            self.col_cycler = cycle(self.cmap.colors)
            self.cycle_len = len(self.cmap.colors)

        # be able to navigate the family tree
        all_lists = True
        for el in children_nlist:
            if not isinstance(el, list):
                all_lists = False
                break
        if not isinstance(children_nlist, list) and not all_lists:
            raise TypeError("Children must be a nested list!")
        self.children_nlist = children_nlist
        if not isinstance(parent_list, list):
            raise TypeError("Parents must be a list!")
        self.parent_list = parent_list

    def get_colour(self, fam_ID):
        """
        Pick a colour for a family, if it has been requested before return the old colour.
        :param fam_ID: some hashable identifier for each family
        :returns: 3-tuple of RGB-values or name of a colour
        """
        # if it has not been requested before, get a new colour and save it
        if fam_ID not in self.fam_colour.keys():
            # fetch parent's and siblings' colours
            cols = []
            parent = self.parent_list[fam_ID]
            for fam in self.children_nlist[parent] + [parent]:
                if fam in self.fam_colour.keys():
                    cols += [self.fam_colour[parent]]

            # initialise loop, terminates if there is no match
            trials = 0
            eq_col = True
            while eq_col:
                # use a random colour if siblings and parent have already used the whole cycle
                if trials >= self.cycle_len:
                    col_candidate = tuple([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
                    break
                # otherwise: fetch a new candidate
                col_candidate = next(self.col_cycler)
                trials += 1
                # compare to parent's and siblings' colours
                eq_col = False
                for col in cols:
                    if col_candidate == col:
                        eq_col = True
                        continue  # enters the next iteration as soon as the first match is detected

            # save result
            self.fam_colour[fam_ID] = col_candidate
        return self.fam_colour[fam_ID]


class PropertyColourMapper:
    """
    Maps from the family ID to a matplotlib colour that encodes the value of
    a family property, e.g. the birth rate
    """

    def __init__(self, prop_list, cmap, norm):
        """
        Initialise an instance of a property colour mapper.
        :param prop_list: list with the family property, indexed by family ID
        :param cmap: string (name of a matplotlib colour) or matplotlib.colors.Colormap
        :param norm: matplotlib.colors.Normalize that maps property values in prop_list to a range between 0 and 1
        """
        # property list for lookup
        if type(prop_list) != list:
            raise TypeError("Properties must be passed as a list!")
        self.proplist = prop_list

        # colour map to encode the values between 0 and 1
        if cmap is not None:
            if type(cmap) == str:
                self.cmap = plt.cm.get_cmap(cmap)
            else:
                if not isinstance(cmap, mplcolors.Colormap):
                    raise TypeError("Colormap must be string (colormap name) or matplotlib.colors.Colormap!")
                self.cmap = cmap
        else:
            self.cmap = plt.cm.get_cmap('jet')

        # function to map from property values to the range [0,1]
        if norm is not None:
            if isinstance(norm, mplcolors.Normalize):
                self.norm = norm
            else:
                raise ValueError("Provided 'norm' keyword must be an instance of matplotlib.colors.Normalize "
                                 "or its subclasses.")
        else:
            self.norm = mplcolors.Normalize(vmin=0, vmax=max(self.proplist))

    def get_colour(self, fam_ID):
        """
        Return the colour that encodes the property value of a family.
        :param fam_ID: int, family identifier that indexes the property list passed to the constructor
        :returns: 3-tuple of RGB-values
        """
        # obtain value
        val = self.proplist[fam_ID]
        # scale it for the colourmap and encode into colour
        return self.cmap(self.norm(val))

    def get_colour_scale(self):
        """
        Return how the instance translates between property values and colours.
        """
        return self.cmap, self.norm


def clip_frequencies(y):
    """
    Clip an array of frequencies to the first non-zero and the last non-zero values.
    Utility for draw_wedges of the Muller plot.
    Adapted from https://phylo-baltic.github.io/baltic-gallery/advanced-muller-plots-raw/
    :returns: (np.ndarray x_clipped, np.ndarray x) -
              (indices of the clipped array, indices of the full array)
    """
    # obtain indices and filter them according to y's content
    x = np.arange(len(y), dtype=int)
    relevant = x[np.where(y > 0)[0]]
    start = np.min(relevant)
    end = np.max(relevant)

    # use start-1 and end+1 for nice fade in and out in the population plot
    if start == 0:
        start = 1

    return x[start-1:end+2], x


# this Muller plot can't deal with >1 starter family, in Bio-LGCA that starter
# family is 0, effectively (holds the whole population)
def draw_wedges(ax, family, cum_pop_t, children_nlist, timeline, facecolour_map,
                edgecolour, label_map=None, lw=0.7, bottom=None, level=None, rel_freq=None,
                clipped_x=None):
    """
    Plot a Muller plot on ax along a timeline grid starting from one family.
    Adapted from https://phylo-baltic.github.io/baltic-gallery/advanced-muller-plots-raw/
    :param ax: matplotlib.Axes.axes: the axis to draw wedges on

    # content parameters
    :param family: family ID
    :param cum_pop_t: array of shape (time, families): cumulative populations of all families and all their children
    :param children_nlist: family tree as nested list of child family IDs, index=parent of the children
    :param timeline: array, timesteps of the simulation

    # plot customisation
    :param facecolour_map: callable that returns the facecolour of a wedge from the family ID
    :param edgecolour: edgecolour of wedges
    :param label_map: callable that returns the label for a wedge from the family ID
    :param lw: line width of the wedge edges

    # parameters for recursion
    :param bottom: lower boundary of wedge area to be filled
    :param level: level of family in the family tree
    :param rel_freq: array, relative cumulative population of the family and its children over time
    :param clipped_x: subset of the array indices where rel_freq > 0
    """

    # level is only provided in recursion, if not: this is the root family
    if level is None:
        level = 1

    # for root family: determine relative frequencies and where they are > 0
    if rel_freq is None or clipped_x is None:
        # relative frequencies
        abs_freq = cum_pop_t[:, family]  # population of family over time
        rel_freq = np.divide(abs_freq, cum_pop_t[:, 0])

        # define lower boundary of wedge to be filled
        if bottom is None:
            bottom = 0 - rel_freq / 2

        # skip drawing if population levels are never big enough
        # (make sure that clipping doesn't crash)
        if sum(rel_freq) <= 0.0:
            return bottom

        # find frequency indices to clip zeroes at the beginning and end
        clipped_x, x = clip_frequencies(rel_freq)
    # for branch/leaf families: skip drawing if population levels are never big enough
    else:
        if sum(rel_freq) <= 0.0:
            return bottom  # new bottom is old bottom, no drawing needs to be done

    # clip zeroes at the beginning and end
    clipped_timeline = timeline[clipped_x]  # x axis
    clipped_bottom = bottom[clipped_x]  # lower boundary of filled area
    clipped_values = clipped_bottom + rel_freq[clipped_x]  # upper boundary of filled area

    # plot customisation
    fc = facecolour_map(family)
    if edgecolour != 'align':
        ec = edgecolour
    else:
        ec = fc
    lab = label_map(family)

    # draw the wedge
    ax.fill_between(clipped_timeline, clipped_bottom, clipped_values, facecolor=fc,
                    edgecolor=ec, alpha=1.0, zorder=level, label=lab, lw=lw)  # plot frequency

    # recurse if family has children to draw smaller wedges over this wedge
    if children_nlist[family]:
        # filter children (relevant if they have children or an own population)
        children_ids = np.array([ch for ch in children_nlist[family] if (children_nlist[ch] != [] or np.any(cum_pop_t[:, ch] > 0))])
        # calculate relative frequencies of children
        child_rel_freqs = np.array([np.divide(cum_pop_t[:, ch], cum_pop_t[:, 0]) for ch in children_ids])

        # calculate padding for drawing children wedges
        #paddings = np.sum(child_rel_freqs > 0, axis=0) + 1  # number of paddings = number of children + 1
        # extend x reach of children to fade in and out
        #extend_left = np.where(N_paddings[:-1] < N_paddings[1:])
        #N_paddings[extend_left] += 1
        #extend_right = np.where(N_paddings[:-1] > N_paddings[1:])[0] + 1
        #N_paddings[extend_right] += 1
        children_sum = np.sum(child_rel_freqs, axis=0)  # sum of all children frequencies for next step
        available_space = rel_freq - children_sum  # family frequency - all children frequencies = total padding space
        individual_space = available_space / 2  # padding space between children and at top and bottom

        # start with parent's bottom, recursively add area and padding of each child
        # this is updated after drawing each child
        temp_bottom = bottom.copy()

        # draw children wedges from bottom to top and add calculated individual padding below each
        for c, child in enumerate(children_ids):
            # pad bottom with space available if this child is present
            padded_bottom = temp_bottom
            if c == 0:
                padded_bottom += individual_space

            # find frequency indices to clip zeroes at the beginning and end
            # skip drawing if population levels are never big enough
            # (make sure that clipping doesn't crash)
            if sum(child_rel_freqs[c]) == 0.0:
                # padding is needed for other children even if this one is not there
                temp_bottom = padded_bottom
                continue
            clipped_child_x, _ = clip_frequencies(child_rel_freqs[c])

            # draw frequency wedges for this child and its children, update bottom
            temp_bottom = draw_wedges(ax, child, cum_pop_t, children_nlist, timeline, bottom=padded_bottom,
                                      facecolour_map=facecolour_map, edgecolour=edgecolour, label_map=label_map, level=level + 1,
                                      rel_freq=child_rel_freqs[c], clipped_x=clipped_child_x)


    # if called in the recursion: update temp_bottom to old_bottom + this node's values
    return bottom + rel_freq


# can be called with **kwargs
def muller_plot(root_ID, cum_pop_t, children_nlist, parent_list, timeline, facecolour='identity',
                facecolour_map=None, cmap=None, norm=None, edgecolour=None,
                xlabel=r"Time $k$", ylabel="Relative frequency", title=None,
                label_map=None, legend_title=None, sort_labels=False, legend_on=False, **kwargs):
    """
    Draw a Muller plot from the given data.
    # content parameters
    :param root_ID: integer, ID of the root family in the family tree defined by children_nlist
    :param cum_pop_t: array of shape (time, families): cumulative populations of all families and all their children
    :param children_nlist: family tree as nested list of child family IDs, index=parent of the children
    :param parent_list: family tree as list of parent family IDs, index=child of this parent
    :param timeline: array, timesteps of the simulation

    # wedge colour customisation
    :param facecolour: ['identity', 'property', name of a matplotlib colour, None] how the wedges should be coloured.
                        identity: based on family identity
                        property: based on a property of the family
                        name of a matplotlib colour: all families equally in this colour
    :param facecolour_map: [callable, list, None] how to map from family ID to colour of the wedge based on
                           the face colouring strategy defined by 'facecolour'
    :param cmap: if facecolour == 'identity': [name of a matplotlib colourmap, list of colour names, ListedColormap, None] (defaults to 'tab20')
                 if facecolour == 'property': [name of a matplotlib colourmap, ListedColormap, None] (defaults to 'jet')
                 2 use cases: - colour map used for colouring in the wedges if facecolour_map is None
                              - colourmap to make a colourbar if facecolour == 'property' and facecolour_map is callable
    :param norm: matplotlib.colors.Normalize: used with cmap to make a colourbar if facecolour == 'property'
                                              and facecolour_map is callable
    :param edgecolour: edgecolour of wedges, if set to 'align' the edgecolour will be the same as the facecolour of each wedge

    # plot setup
    :param xlabel: label of the x axis
    :param ylabel: label of the y axis
    :param title: title of the Muller plot

    # labels and legend
    :param label_map: [callable, list, None] how to map from family ID to the label in the legend,
                                             only used if facecolour == 'identity' (defaults to family index)
    :param legend_title: title of the legend (if facecolour=='identity') or colourbar (if facecolour=='property')
    :param sort_labels: Boolean - if True, sort labels in the legend alphabetically (family tree = by level).
                        If False, labels will appear in the same order that the artists are drawn: following each
                        branch of the family tree from the root to the leaves, then the next branch

    :returns: (fig, ax, ret) fig = matplotlib figure handle, ax = Muller plot axis handle,
                             ret = handle of legend, handle of colourbar or None. The separate colourbar axis handle
                             can be retrieved as ret.ax
    """
    # set up drawing mode
    legend = False
    colourbar = False
    ret = None
    # colour wedges by family identity
    if facecolour == 'identity':
        legend = True
        colourbar = False
        if legend_title is None:
            legend_title = "Family"
        # custom function that determines the colour
        if callable(facecolour_map):
            fc_map = facecolour_map
        # predefined list with a colour for each family
        elif type(facecolour_map) == list:
            fc_map = lambda fam_ID: facecolour_map[fam_ID]
        # default: cycle through cmap, cmap default is 'tab 20'
        elif facecolour_map is None:
            fc_mapper = IdentityColourMapper(cmap, children_nlist, parent_list)
            fc_map = fc_mapper.get_colour
        else:
            raise TypeError("facecolour_map must be callable, list or None.")
    # colour wedges by a property of the family
    elif facecolour == 'property':
        legend = False
        colourbar = True
        # custom function that determines the colour from a property
        if callable(facecolour_map):
            fc_map = facecolour_map
            # colour scaling must be passed to interpret the values
            if cmap is None or norm is None:
                raise AttributeError("When facecolour mapping is set to a custom property mapping the colour scaling of "
                                     "the property values must be provided with keywords 'cmap' and 'norm'.")
            prop_cmap = cmap
            prop_norm = norm
        # list with property values for each family
        elif type(facecolour_map) == list:
            fc_mapper = PropertyColourMapper(facecolour_map, cmap, norm)
            fc_map = fc_mapper.get_colour
            prop_cmap, prop_norm = fc_mapper.get_colour_scale()
        else:
            raise TypeError("When facecolour mapping is set to property, the property value for each family must "
                                 "be provided as keyword 'facecolour_map' as a list or callable.")
        if legend_title is None:
            if callable(facecolour_map):
                legend_title = facecolour_map.__name__
            else:
                legend_title = "Property"
    # colour all wedges the same colour, user provided
    elif type(facecolour) == str:
        try:
            _ = mplcolors.to_rgba(facecolour)
            fc_map = lambda fam_ID: facecolour
            if edgecolour is None:
                edgecolour = 'k'
        except ValueError as e:
            raise ValueError("Colour not found! Facecolour string was interpreted as matplotlib colour name.") from e
    # colour all wedges the same colour, default colour
    else:
        fc_map = lambda fam_ID: 'tan'
        if edgecolour is None:
            edgecolour = 'k'

    # label wedges according to family identity
    if label_map is None:
        if facecolour == 'property':
            l_map = lambda fam_ID: None
            print("Labels overwritten by property colour coding!")
        else:
            l_map = lambda fam_ID: None if fam_ID == 0 else fam_ID
    # label wedges according to custom function
    elif callable(label_map):
        l_map = label_map
    # label wedges according to custom list
    elif type(label_map) == list:
        l_map = lambda fam_ID: label_map[fam_ID]
    else:
        raise TypeError("label_map must be callable, list or None.")
    legend = legend_on
    # set up figure and draw wedges=relative frequencies for each family
    fig = plt.figure()
    ax = plt.gca()
    draw_wedges(ax, root_ID, cum_pop_t, children_nlist, timeline, facecolour_map=fc_map, edgecolour=edgecolour, label_map=l_map)
    plt.title(title)

    # axes layout
    plt.xlim([timeline.min(), timeline.max()])
    plt.ylim([-0.5, 0.5])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # shift tick labels to start from 0 and stop at 1
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: x + 0.5))

    if legend:
        # set up legend, with sorted entries if desired
        # this is done for facecolour = 'identity'
        handles, labels = ax.get_legend_handles_labels()  # retrieve list of artists and label strings
        if sort_labels:
            sorted_handles = [han for _, han in sorted(zip(labels, handles))]
            sorted_labels = sorted(labels)
            handles = sorted_handles
            labels = sorted_labels
        leg = plt.legend(handles, labels, title=legend_title, loc='center left', bbox_to_anchor=[1.01, 0.5],
                         ncol=int(np.ceil(len(handles)/20)))
        ret = leg

    if colourbar:
        # set up a colour bar to interpret property values that colours reflect
        # this is done for facecolour = 'property'
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.12, pad=0.1)
        cbar = plt.colorbar(cm.ScalarMappable(norm=prop_norm, cmap=prop_cmap), cax=cax, spacing='proportional', label=legend_title)
        ret = cbar

    plt.tight_layout()
    return fig, ax, ret, fc_map
def colorbar_index(ncolors: int, cmap, use_gridspec: bool=False, cax=None):
    """
    Create a colorbar with `ncolors` colors.

    Builds a discrete colormap with `ncolors` colors from the near-continuous colormap `cmap`,
    adds it to the axis `cax` and draws tick labels in the center of each color. If
    ncolors is high, tick positions are determined automatically using
    :class:`~matplotlib.ticker.MaxNLocator` and labels are formatted with
    :class:`~matplotlib.ticker.FuncFormatter`.

    Parameters
    ----------
    ncolors : int
        Desired number of colors for the discretized colormap.
    cmap : str or :py:class:`matplotlib.colors.Colormap`
        Near-continuous colormap to create discrete colormap from, e.g. ``matplotlib.cm.jet`` or ``'jet'``.
    use_gridspec : bool, optional
        Passed on to :py:func:`matplotlib.pyplot.colorbar`.
    cax : :py:class:`matplotlib.axes.Axes` object, optional
        Axis into which the colorbar will be drawn.

    Returns
    -------
    colorbar : :py:class:`matplotlib.colorbar.Colorbar`
        Colorbar instance.

    """
    # discretize the colormap
    cmap = cmap_discretize(cmap, ncolors)

    # map colors to values
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors - 0.5)

    # create colorbar with discrete boundaries
    boundaries = np.arange(-0.5, ncolors, 1)
    colorbar = plt.colorbar(
        mappable, use_gridspec=use_gridspec, cax=cax, boundaries=boundaries
    )

    # configure ticks and labels using locators and formatters
    locator = MaxNLocator(nbins="auto", integer=True)
    formatter = FuncFormatter(lambda val, pos: int(val))
    colorbar.ax.yaxis.set_major_locator(locator)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.update_ticks()
    return colorbar


def cmap_discretize(cmap, N: int):
    """
    Downsample the near-continuous colormap `cmap` to the number of colors `N`.

    Parameters
    ----------
    cmap : str or :py:class:`matplotlib.colors.Colormap`
        Colormap to be discretized, e.g. ``matplotlib.cm.jet`` or ``'jet'``.
    N : int
        Number of colors of the new colormap.

    Returns
    -------
    :py:class:`matplotlib.colors.LinearSegmentedColormap`
        Discretized colormap with `N` colors.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> x = np.resize(np.arange(100), (5,20))
    >>> # discretize jet colormap
    >>> djet = cmap_discretize(cm.jet, 5)
    >>> # show color limits
    >>> plt.imshow(x, cmap=djet)

    The name of the colormap is updated to ``cmap.name + '_N'``:

    >>> djet.name
    'jet_5'

    """
    # see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#creating-linear-segmented-colormaps
    # for details
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    # create anchor points and fill them with colors from the colormap
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    # index rgba values according to discretization
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # create new linear segmented colormap
    return plt.matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def estimate_figsize(array, x: float=8., cbar: bool=False, dy: float=1.):
    """
    .. deprecated:: 1.0
        :py:func:`estimate_figsize` will be removed in biolgca 1.0, it is replaced
        by the default value for the figure size in :py:meth:`setup_figure` of the
        respective LGCA object.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        Array holding the data to be plotted.
    x : float, default=8.0
        Desired x dimension of the figure. Used to scale the y dimension.
    cbar : bool, optional
        If the figure will contain a colorbar.
    dy : float, default=1.0
        Scale of a unit in the y direction as compared to the x direction.

    Returns
    -------
    figsize : tuple(float, float)
        Optimal figure size.

    """
    lx, ly = array.shape
    if cbar:
        y = min([abs(x * ly /lx - 1), 10.])
    else:
        y = min([x * ly / lx, 10.])
    y *= dy
    figsize = (x, y)
    return figsize


def get_cmap(
    density,
    ax=None,
    vmax=None,
    cmap="viridis",
    cbar=True,
    cbarlabel="",
    colorbarwidth="5%",
    pad=0.1,
):
    if vmax is None:
        K = int(density.max())
    else:
        K = vmax

    cmap = copy(cm.get_cmap(cmap))  # do not modify a globally registered colormap in matplotlib > 3.3.2
    cmap.set_under(alpha=0.0)
    cmap_scaled = False

    if 1 < K <= cmap.N:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
    elif K > 1:
        cmap_scaled = True
        scaling_factor = K / cmap.N
        nbins = cmap.N
        density = density / scaling_factor
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(cmap.N + 1), cmap.N))
    else:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=1e-6, vmax=1))
    cmap.set_array(density)

    if not cbar:
        return cmap

    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbarwidth, pad=pad)

    if K <= 1:
        # requires extra treatment because there is only one colour
        cbar = plt.colorbar(
            cmap,
            cax=cax,
            extend="min",
            use_gridspec=True,
            boundaries=[0, 0.5, 1],
            values=[0, 1],
        )
    else:
        cbar = plt.colorbar(cmap, cax=cax, extend="min", use_gridspec=True)
    cbar.set_label(cbarlabel)
    if cmap_scaled:
        ncolors = nbins
    else:
        ncolors = max(1, K)
    # set a numbering interval for high densities (e.g. 5-10-15-20)
    if ncolors > 101:
        stride = 10
    elif ncolors > 51:
        stride = 5
    elif ncolors > 31:
        stride = 2
    else:
        stride = 1
    if K <= 1:
        # requires extra treatment because there is only one colour
        ticks = np.array([0.75])
    else:
        ticks = np.arange(1, ncolors + 1, 1) + 0.5
    if cmap_scaled:
        indices = np.arange(1, nbins + 2)
        low_label = np.ceil(indices * scaling_factor - 1e-6)
        low_label = np.roll(low_label, 1)
        low_label = np.delete(low_label, 0).astype(int)
        labels = list(low_label)
    else:
        labels = list(np.arange(1, ncolors + 1, 1, dtype=int))
    # if max label comes up automatically, leave it as it is
    if ticks[-1] == ticks[0::stride][-1]:
        cbar.set_ticks(ticks[0::stride])
        cbar.set_ticklabels(labels[0::stride])
    # if max label is too close to last strided label, leave the latter out
    elif stride > 1 and ticks[-1] != ticks[0::stride][-1] and ticks[-1] - ticks[0::stride][-1] < stride / 2:
        cbar.set_ticks(list(ticks[0::stride][:-1]) + [ticks[-1]])
        cbar.set_ticklabels(labels[0::stride][:-1] + [labels[-1]])
    # if there is enough space, just add the max label
    else:
        cbar.set_ticks(list(ticks[0::stride]) + [ticks[-1]])
        cbar.set_ticklabels(labels[0::stride] + [labels[-1]])
    plt.sca(ax)
    return cmap



