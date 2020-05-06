from .analysis import *

from matplotlib import colors

from .base import estimate_figsize
from .lgca_square import *
from .ib_interactions import driver_mut


def plot_families(nodes_t, lab_m, dims=50,
                  figsize=None, cmap='inferno', save=False, id=0):
    """
    Idee: für einen Zeitschritt (nodes_t entsprechend übergeben)
     für jeden Knoten die häufigste Familie plotten
    NACHTRÄGLICH -> daher cond für Gittereigenschaften
    :param nodes_t: EIN Eintrag
    :param lab_m: =lgca.props['lab_m]

    """
    cond = get_lgca(ib=True, geometry='hex', bc='reflecting', dims=dims,
                    interaction='mutations', effect=driver_mut)
    lx, ly, K = nodes_t.shape
    maxfam = max(lab_m)
    print('mf', maxfam)
    print('lx, ly, K', lx, ly, K)

    data = np.array([[-99] * lx] * ly)
    for dy in range(0, ly):
        for dx in range(0, lx):
            fams = []
            for lab in nodes_t[dy][dx]:
                if lab != 0:
                    fams.append(lab_m[lab])
            reg = {}
            for f in fams:
                reg[f] = fams.count(f)
            if len(reg) > 0:
                max_fam = max(reg, key=reg.get)
            else:
                max_fam = 0
            data[dy][dx] = max_fam
    print('data', data)
    if figsize is None:
        figsize = estimate_figsize(data, cbar=True, dy=cond.dy)
    maxi = data.max()
    print('maxi', maxi)
    fig, ax = cond.setup_figure(figsize=figsize, tight_layout=True)
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_under(alpha=0.0)
    if maxi > 1:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(maxi + 1), cmap.N))
    else:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(maxi+2), cmap.N))
    cmap.set_array(data)
    polygons = [RegularPolygon(xy=(x, y), numVertices=cond.velocitychannels, radius=cond.r_poly,
                               orientation=cond.orientation, facecolor=c, edgecolor=None)
                for x, y, c in zip(cond.xcoords.ravel(), cond.ycoords.ravel(), cmap.to_rgba(data.ravel()))]
    pc = PatchCollection(polygons, match_original=True)
    ax.add_collection(pc)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = fig.colorbar(cmap, extend='min', use_gridspec=True, cax=cax)
    # cbar.set_label('family')
    # if maxi > 10:
    #     cbar.set_ticks(np.linspace(1, maxi + 1, maxi + 1))
    #     cbar.set_ticklabels([1] + ['']*(maxi-1) + [maxi+1])
    # else:
    #     cbar.set_ticks(np.linspace(1, maxi + 1, maxi+1))
    #     cbar.set_ticklabels(np.arange(1, maxi+2, 1))

    # plt.sca(ax)

    if save:
        filename = str(id) + 'fams_hex' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)
    plt.show()
