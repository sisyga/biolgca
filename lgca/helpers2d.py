from .analysis import *
from matplotlib import colors

def plot_families(nodes):
    tend, lx, ly, K = nodes.shape
    print('tend, lx, ly, K', tend, lx, ly, K)
    mask = np.any(nodes, axis=-1)
    # meanprop = self.calc_prop_mean(propname=propname, props=props, nodes=nodes)
    fig, pc, cmap = self.plot_scalarfield(meanprop, mask=mask, **kwargs)
    return fig, pc, cmap