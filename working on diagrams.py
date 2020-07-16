from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(10, 101, 10)
# xlabs = [str(entry) + '%' for entry in x]
# print(xlabs)
# d = [18, 11, 4, 5, 2, 0, 0,	1, 0, 4]
# p = [0, 0, 0, 1, 1, 1, 4, 4, 11, 23]
# # node = [0, 0, 0, 5, 5, 3, 3, 8, 7, 14]
# # rc = [0, 0, 0, 1, 8, 5, 7, 7, 4, 13]
#
# width = 4
#
# fig, ax = plt.subplots(figsize=(12, 8))
# size_ticks = 20
# size_legend = 30
# rects1 = ax.bar(x - width/2, d, width, label='Driver', color=farben['driver'])
# rects2 = ax.bar(x + width/2, p, width, label='Passenger', color=farben['passenger'])
# # rects1 = ax.bar(x - width/2, node, width, label='I', color=farben['onenode'])
# # rects2 = ax.bar(x + width/2, rc, width, label='II', color=farben['onerc'])
#
# ax.set_ylabel('Anzahl Simulationen', fontsize=size_legend)
# ax.set_xlabel('prozentualer Anteil', fontsize=size_legend)
# ax.set_xticks(x)
# ax.set_xticklabels(xlabs)
# plt.xticks(fontsize=size_ticks)
# plt.yticks(fontsize=size_ticks)
#
# ax.legend(loc='upper center', fontsize=size_legend)
#
# # plt.savefig(pathlib.Path('pictures').resolve() / 'domfams.png')
# plt.savefig(pathlib.Path('pictures').resolve() / 'maxfamrelativ.png')
#
# plt.show()

cmaps = [('Miscellaneous', ['nipy_spectral', 'inferno', 'viridis'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    # fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()