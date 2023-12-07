import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from matplotlib import animation
from .lgca_hex import LGCA_Hex
import math
import matplotlib as mpl
# mpl.use('svg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, ListedColormap
import copy
import matplotlib.colors as mcolors


def boundary_coord(coord : list, lx, ly):
    new_coord = []
    for i in coord:
        if i[0] == 0 and i[1] != 0:
            new_coord.append((lx, i[1]))
        if i[0] != 0 and i[1] == 0:
            new_coord.append((i[0], ly))
        if i[0] == lx+1 and i[1] != ly+1:
            new_coord.append((0, i[1]))
        if i[0] != lx+1 and i[1] == ly+1:
            new_coord.append((i[0], 0))
        if i[0] != 0 and i[0] != lx+1 and i[1] != 0 and i[1] != ly+1:
            new_coord.append((i))
    return new_coord

def nb_ECM(values, coord, restchannels, K=6):
    nb = np.zeros(K+restchannels)
    nb[-1] = values[((coord[0] + 1, coord[1]))]
    nb[-4] = values[((coord[0] - 1, coord[1]))]
    nb[0:restchannels] = values[(coord)]

    if coord[1] % 2 == 0:
        nb[-3] = values[((coord[0] - 1, coord[1] + 1))]
        nb[-2] = values[((coord[0], coord[1] + 1))]
        nb[-6] = values[((coord[0], coord[1] - 1))]
        nb[-5] = values[((coord[0] - 1, coord[1] - 1))]

    if coord[1] % 2 != 0:
        nb[-3] = values[((coord[0], coord[1] + 1))]
        nb[-2] = values[((coord[0] + 1, coord[1] + 1))]
        nb[-6] = values[((coord[0] + 1, coord[1] - 1))]
        nb[-5] = values[((coord[0], coord[1] - 1))]
    return nb


def nb_coord (coord, lgca):
    nb = [None] * 7
    nb[3] = (coord[0] + 1, coord[1])
    nb[6] = (coord[0] - 1, coord[1])
    nb[0] = (coord[0], coord[1])

    if coord[1] % 2 == 0:
        nb[1] = (coord[0], coord[1] - 1)
        nb[2] = (coord[0] - 1, coord[1] - 1)
        nb[4] = (coord[0] - 1, coord[1] + 1)
        nb[5] = (coord[0], coord[1] + 1)

    if coord[1] % 2 != 0:
        nb[1] = (coord[0] + 1, coord[1] - 1)
        nb[2] = (coord[0], coord[1] - 1)
        nb[4] = (coord[0], coord[1] + 1)
        nb[5] = (coord[0] + 1, coord[1] + 1)

    # nb = [(np.mod(x, 20), np.mod(y, 20)) for (x,y) in nb]
    nb = [(1, y) if x==lgca.lx+1 else (x,y) for (x,y) in nb]
    nb = [(x, 1) if y==lgca.lx+1 else (x,y) for (x,y) in nb]
    nb = [(lgca.lx, y) if x==0 else (x,y) for (x,y) in nb]
    nb = [(x, lgca.lx) if y==0 else (x,y) for (x,y) in nb]
    return nb


def inertia_tensor(weights):
    tens = []
    x1 = [0.5, -0.5, -1, -0.5, 0.5, 1]
    y1 = [np.sin(np.pi / 3), np.sin(np.pi / 3), 0, -np.sin(np.pi / 3), -np.sin(np.pi / 3), 0]

    for i in np.arange(len(x1)):
        tens.append(weights[i] * np.array([[y1[i] ** 2, -x1[i] * y1[i]], [-x1[i] * y1[i], x1[i] ** 2]]))
    ew, ev = np.linalg.eig(sum(tens))
    if ew[1] >= ew[0]:
        ev = ev[::-1]
    if np.round(ew[1], 10) == np.round(ew[0], 10):
        ev[0] = [np.random.uniform(-1,1) for _ in ev[0]]
        ev[0] = ev[0]/np.linalg.norm(ev[0])
    guid_tens = np.outer(ev[0], ev[0]) - 0.5 * np.diag(np.ones(2))
    return guid_tens, ev, ew

class Ecm(LGCA_Hex):
    dy = np.sin(2 * np.pi / 6)
    r_int = 1
    def __init__(self, lx, ly, restchannels, d, d_neigh, timesteps, t, *args):
        if len(args) == 0:
            self.r_int = 1
            self.random_steps = 500
            self.timesteps = timesteps
            self.restchannels = restchannels
            self.lx = lx
            self.ly = ly
            self.d = d
            self.d_neigh = d_neigh
            self.init_coords()
            self.scalar_field = np.zeros((self.lx + 2, self.ly + 2))
            self.scalar_field_t = np.zeros((self.timesteps+1, self.lx + 2, self.ly + 2))
            self.nodes_t = np.zeros((self.timesteps+1, self.lx + 2, self.ly + 2))
            self.vector_field = np.zeros((self.lx + 2, self.ly + 2, 3))
            self.vector_field_t = np.zeros((self.timesteps+1, self.lx + 2, self.ly + 2, 3))
            self.tensor_field = np.zeros((self.lx + 2, self.ly + 2, 2, 2))
            self.init_scalar()
            self.update_dynamic_fields()
            self.periodic_rb()
            self.t = t
            self.list_paths_number = []
            self.list_paths_length = []
            self.paths = []
            self.p_inf = []
            self.perco_ratio = []

            new_rc_params = {'text.usetex': False,
                             "svg.fonttype": 'none'
                             }
            mpl.rcParams.update(new_rc_params)

    def init_scalar(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                self.scalar_field[(int(x), int(y))] = np.random.uniform(0, 1)
        self.scalar_field_t[0] = self.scalar_field

    def init_bigger_zebra(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                if y%4 == 0 or (y+1)%4 == 0:
                    self.scalar_field[(int(x), int(y))] = 0.3
                else :
                    self.scalar_field[(int(x), int(y))] = 0.7
        self.scalar_field_t[0] = self.scalar_field

    def init_biggest_zebra(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                if y%4 == 0 or (y+1)%4 or (y+2)%4  == 0:
                    self.scalar_field[(int(x), int(y))] = 0.3
                else :
                    self.scalar_field[(int(x), int(y))] = 0.7
        self.scalar_field_t[0] = self.scalar_field

    def init_zebra(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                if y%2 == 0:
                    self.scalar_field[(int(x), int(y))] = 0.7
                else :
                    self.scalar_field[(int(x), int(y))] = 0.0
        self.scalar_field_t[0] = self.scalar_field
    def init_isolated(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                # if y%2 == 0 and y%4 != 0 and (x+1)%2 == 0:
                #     self.scalar_field[(int(x), int(y))] = 0.7
                if y%2 == 0 and x%2 == 0 :
                    self.scalar_field[(int(x), int(y))] = 0.7
                else :
                    self.scalar_field[(int(x), int(y))] = 0.0
        self.scalar_field_t[0] = self.scalar_field
    def init_scalar_uniform(self, density_ecm =0.5):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                self.scalar_field[(int(x), int(y))] = density_ecm
        self.scalar_field_t[0] = self.scalar_field

    def init_scalar_discrete(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                self.scalar_field[(int(x), int(y))] = np.random.choice([0, 1], p=[0.5, 0.5])

    def init_scalar_split(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                if x >= (self.lx+1)/2:
                    self.scalar_field[(int(x), int(y))] = np.random.uniform(0.5, 1)
                if x < (self.lx+1)/2:
                    self.scalar_field[(int(x), int(y))] = np.random.uniform(0.0, 0.5)
    def init_obstacles(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                index = self.coord_pairs.index((x, y))
                (x1, y1) = self.coord_pairs_hex[index]
                self.scalar_field[(int(x), int(y))] = 0.8*np.abs(1* np.sin((4*np.pi*x1 / (1*self.lx))) * np.sin((4*np.pi*(y1+3) / (1*self.ly))))
    def init_sandwich(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                if y in [self.ly/2-2, self.ly/2-1]  :
                    self.scalar_field[(int(x), int(y))] = abs(np.random.uniform(0.1, 0.1))
                else:
                    self.scalar_field[(int(x), int(y))] = abs(np.random.uniform(0.9, 0.9))

    def init_cont(self):
        for x in np.arange(1, self.lx + 1):
            for y in np.arange(1, self.ly + 1):
                self.scalar_field[(int(x), int(y))] = x/(self.lx+1)


    def periodic_rb(self):
        self.scalar_field[0, :] = self.scalar_field[self.lx, :]
        self.scalar_field[self.lx+1, :] = self.scalar_field[1, :]
        self.scalar_field[:, 0] = self.scalar_field[:, self.ly]
        self.scalar_field[:, self.ly+1] = self.scalar_field[:, 1]

    def update_scalar_field(self, cell_densities):
        # degrade density (self.d) in each cell and deposit (self.d_neigh) in the surrounding cells depending on the cell_den
        assert cell_densities.shape == self.scalar_field.shape
        for coord in self.coord_pairs: 
            # make sure that the values are between 0 and 1
            self.scalar_field[coord] -= self.d * cell_densities[coord]
            self.scalar_field[coord] = np.clip(self.scalar_field[coord], 0, 1)
            for neigh in nb_coord(coord, self)[1:7]:
                self.scalar_field[neigh] += self.d_neigh * cell_densities[coord]  # Change here
                self.scalar_field[neigh] = np.clip(self.scalar_field[neigh], 0, 1)
    
    def update_dynamic_fields(self):
        for i in self.coord_pairs:
            weights = nb_ECM(self.scalar_field, i, self.restchannels)
            tensor, ev, ew = inertia_tensor(weights[self.restchannels:])
            weights = nb_ECM(self.scalar_field, i, self.restchannels)
            tensor, ev, ew = inertia_tensor(weights[self.restchannels:])
            self.vector_field[i] = np.array([ev[0][0], ev[0][1], max(ew)])
            self.tensor_field[i] = tensor
    
    def tensor_update(self, t):
        for i in self.coord_pairs:
            weights = nb_ECM(self.scalar_field, i, self.restchannels)
            tensor, ev, ew = inertia_tensor(weights[self.restchannels:])
            self.vector_field[i] = np.array([ev[0][0], ev[0][1], max(ew)])
            self.tensor_field[i] = tensor
            if t == 0:
                self.vector_field_t[0][i] = self.vector_field[i]
            else:
                self.vector_field_t[t][i] = np.array([ev[0][0], ev[0][1], max(ew)])

    def plot_ECM(self, edgecolor = None, show_tensor=False):
        r_poly = 0.5 / np.cos(np.pi / 6)
        dy = 0.5 / np.sin(np.pi / 6)

        fig = plt.figure()
        ax = plt.gca()
        cmap = plt.cm.get_cmap("Greys")
        my_cmap = cmap(np.arange(cmap.N))
        # Set alpha
        my_cmap[:, -1] = np.linspace(0, 0.5, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        cmap.set_under(alpha=0)
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm = norm)

        # cbar = fig.colorbar(cmap)
        # cbar.set_label(r'ECM density $\rho$')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        scalar_field_values = self.scalar_field[self.nonborder].ravel()
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=2.1)
        # cbar = fig.colorbar(cmap, extend='min', use_gridspec=True, cax=cax)
        # cbar.set_label(r'ECM density $\rho$')
        # plt.sca(ax)

        if show_tensor:
            self.tensor_vis(self.t)
        return fig, pc, cmap

    def plot_onlytensor(self, edgecolor = None):
        fig = plt.figure()
        ax = plt.gca()
        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + self.dy
        ymin = self.ycoords.min() - self.dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        #scalar_field_values = self.scalar_field[self.nonborder].ravel()
        scalar_field_values = np.zeros(np.shape(self.scalar_field[self.nonborder]))
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        # tensor_vis(lgca, lgca.t - 1)
        # ftimage = np.fft.fft2(lgca.scalar_field_t[lgca.t-1])
        # ftimage = np.fft.fftshift(ftimage)
        # plt.imshow(np.abs(ftimage))

        self.tensor_vis(self.t - 1)
        return fig, pc, cmap
    # def animate_random_percolation(self, lgca, interval=100 , save_anim = True):
    #     fig, pc, cmap = self.plot_ECM()
    #     title = plt.title('Time $k =$0')
    #     def animate(n):
    #         title.set_text('percentage of filled tiles = ${}$'.format(np.round(np.sum(self.random_perc_t[n])/900, 2)))
    #         scalar_field_values = self.random_perc_t[n]
    #         pc.set(facecolor=cmap.to_rgba(scalar_field_values.ravel()))
    #         return pc, title
    #
    #     anim = animation.FuncAnimation(fig, animate, interval=interval, frames=501)
    #     if save_anim == True:
    #         #
    #         # Writer = animation.writers['ffmpeg']
    #         # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    #         anim.save(f, fps=5, dpi=200)
    #     return anim

    def animate_ECM(self, lgca, interval=100, save_anim=False):
        fig, pc, cmap = self.plot_ECM()
        title = plt.title('Time $k =$0')

        def animate(n):
            title.set_text('Time $k =${}'.format(n))
            scalar_field_values = self.scalar_field_t[n][self.nonborder]
            pc.set(facecolor=cmap.to_rgba(scalar_field_values.ravel()))
            return pc, title

        anim = animation.FuncAnimation(fig, animate, interval=interval, frames=self.t+1)
        return anim

    def tensor_vis(self, t):
        param = []
        for i in np.arange(1, len(self.coord_pairs)):
            x, y = self.coord_pairs_hex[i][0], self.coord_pairs_hex[i][1]

            x1 = self.vector_field_t[t][self.coord_pairs[i]][0]
            y1 = self.vector_field_t[t][self.coord_pairs[i]][1]
            linewidth = np.exp(self.vector_field_t[t][self.coord_pairs[i]][2])*0.05

            param.append(self.vector_field_t[t][self.coord_pairs[i]][2])
            xy = (x - x1 / 2, y - y1 / 2)
            xy2 = (x + x1 / 2, y + y1 / 2)
            plt.annotate(text='', xy=xy, xytext=xy2, arrowprops=dict(arrowstyle='-', linewidth=1))
            plt.annotate(text=np.round(self.scalar_field_t[t][self.coord_pairs[i]],2) ,xy = (x,y) )

    def plot_tensor(self):
        r_poly = 0.5 / np.cos(np.pi / 6)
        dy = 0.5 / np.sin(np.pi / 6)
        zero_field = np.zeros(np.shape(self.scalar_field))
        fig = plt.figure()
        ax = plt.gca()
        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=r_poly,
                                   orientation=0, facecolor=c, edgecolor="black")
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(zero_field.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        self.tensor_vis(self.t - 1)
        # tensor_vis(lgca,  0)

    def animate_histo(self):

        bins = math.ceil((3.5) / 0.02)
        data = []
        max_values = []
        for i in np.arange(1, len(self.coord_pairs)):
            data.append(self.vector_field_t[0][self.coord_pairs[i]][2][0])


        fig, ax = plt.subplots()
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        def prepare_animation(bar_container):
            def animate(n):

                # simulate new data coming in
                data = []
                for i in np.arange(1, len(self.coord_pairs)):
                    data.append(self.vector_field_t[n][self.coord_pairs[i]][2][0])
                n, _ = np.histogram(data, bins)
                ax.set_ylim(0, max(n) +1)
                ax.set_yticks([i for i in range(0, max(n) + 1, int(max(n)/3))])


                max_values.append(max(n))
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)

                return bar_container.patches
            return animate

        _, _, bar_container = ax.hist(data, bins, lw=1,
                                      ec="yellow", fc="green", alpha=0.5)
        #ax.set_ylim(top=1000)

        ani = animation.FuncAnimation(fig, prepare_animation(bar_container), self.t,
                                      repeat=True, blit=True)
        return ani

    def plot_neigh_occupied(self, lgca, edgecolor = None):
        r_poly = 0.5 / np.cos(np.pi / 6)
        dy = 0.5 / np.sin(np.pi / 6)

        fig, ax = self.setup_figure(tight_layout=True)

        fig.suptitle('Celldensity in the neighbourhood', fontsize=16)

        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        # cbar = fig.colorbar(cmap)
        # cbar.set_label('ECM density $d$')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        scalar_field_values = self.scalar_field[self.nonborder].ravel()
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)

        occupied = np.where((lgca.cell_density >= 1), 1, lgca.cell_density)
        neigh = [x + 0.01 for x in lgca.nb_sum(lgca.cell_density)[lgca.nonborder]]
        occupied_nb = np.multiply(occupied[lgca.nonborder], neigh)

        eigv = np.transpose(self.vector_field[lgca.nonborder][:, :, 2])
        eigv = [x + 0.01 for x in eigv]
        occupied_nb_ecm = np.multiply(occupied[lgca.nonborder], eigv)


        scalar_field_values = occupied_nb

        cmap = plt.cm.get_cmap("viridis")
        cmap.set_under(alpha=0)

        cmap.set_bad('white')
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array(np.ma.masked_where(scalar_field_values == 0, scalar_field_values))
        cbar = fig.colorbar(cmap)
        cbar.set_label('Neighbour cell density $d$')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        # tensor_vis(lgca, lgca.t - 1)
        # ftimage = np.fft.fft2(lgca.scalar_field_t[lgca.t-1])
        # ftimage = np.fft.fftshift(ftimage)
        # plt.imshow(np.abs(ftimage))
        return fig, pc, cmap

    def plot_EV_occupied(self, lgca, edgecolor =None):

        fig, ax = self.setup_figure(tight_layout=True)

        occupied = np.where((lgca.cell_density >= 1), 1, lgca.cell_density)
        test = self.vector_field[lgca.nonborder]
        # print(np.shape(test))
        eigv = test[:, :, 2]
        eigv = np.reshape(eigv, (self.lx,self.ly))
        # print(eigv, np.shape(eigv))
        eigv = [(x + 0.01) for x in eigv]
        occupied_nb_ecm = np.multiply(occupied[lgca.nonborder], eigv)
        scalar_field_values = occupied_nb_ecm

        cmap = plt.cm.get_cmap("viridis")
        cmap.set_under(alpha=0)

        cmap.set_bad('white')
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array(np.ma.masked_where(scalar_field_values == 0, scalar_field_values))
        cbar = fig.colorbar(cmap)
        cbar.set_label('Eigenvalue of intertia tensor')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + self.dy
        ymin = self.ycoords.min() - self.dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)

        # tensor_vis(lgca, lgca.t - 1)
        # ftimage = np.fft.fft2(lgca.scalar_field_t[lgca.t-1])
        # ftimage = np.fft.fftshift(ftimage)
        # plt.imshow(np.abs(ftimage))

        return fig, pc, cmap

    def plot_response_to_radiation(self, lgca, edgecolor =None):

        fig, ax = self.setup_figure(tight_layout=True)
        fig.suptitle('Cell repair in response to radiational damage', fontsize=16)

        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        # cbar = fig.colorbar(cmap)
        # cbar.set_label('ECM density $d$')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        scalar_field_values = self.scalar_field[self.nonborder].ravel()
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        occupied = np.where((lgca.cell_density >= 1), 1, lgca.cell_density)
        eigv = self.vector_field[lgca.nonborder]
        # print(np.shape(test))
        eigv = eigv[:, :, 2]
        eigv = np.reshape(eigv, (self.lx,self.ly))
        # print(eigv, np.shape(eigv))
        eigv = [np.exp(1*x) for x in eigv]
        occupied_nb_ecm = np.multiply(occupied[lgca.nonborder], eigv)
        occupied_nb_ecm[-1,-1]= 1
        scalar_field_values = occupied_nb_ecm
        cmap = plt.cm.get_cmap("viridis")
        cmap.set_under(alpha=0)
        cmap.set_bad('white')
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array(np.ma.masked_where(scalar_field_values == 0, scalar_field_values))
        cbar = fig.colorbar(cmap)
        #cbar.ax.set_yticklabels(['', 'high'])
        cbar.remove()
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + self.dy
        ymin = self.ycoords.min() - self.dy
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        ax.set_title('t = 0')


        # cbar = fig.colorbar(cmap, ticks=[1.1, max(scalar_field_values.ravel())])
        # cbar.ax.set_yticklabels(['low', 'high'], fontsize=14)

        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        # cbar = fig.colorbar(cmap)
        # cbar.set_label('ECM density $d$')
        scalar_field_values = self.scalar_field_t[self.t-1][self.nonborder].ravel()
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)

        # ax2.add_collection(pc)
        # ax2.set_xlabel("$x(\epsilon)$")
        # occupied = np.where((self.nodes_t[self.t-1] >= 1), 1, self.nodes_t[self.t-1])
        # eigv = self.vector_field_t[self.t-1][lgca.nonborder]
        # # print(np.shape(test))
        # eigv = eigv[:, :, 2]
        # eigv = np.reshape(eigv, (self.lx, self.ly))
        # # print(eigv, np.shape(eigv))
        # eigv = [np.exp(1.5 * x + 0.01) for x in eigv]
        # occupied_nb_ecm = np.multiply(occupied[lgca.nonborder], eigv)
        # scalar_field_values = occupied_nb_ecm
        # cmap = plt.cm.get_cmap("viridis")
        # cmap.set_under(alpha=0)
        #
        # xmax = self.xcoords.max() + 0.5
        # xmin = self.xcoords.min() - 0.5
        # ymax = self.ycoords.max() + self.dy
        # ymin = self.ycoords.min() - self.dy
        # ax2.set_xlim(xmin, xmax)
        # ax2.set_ylim(ymin, ymax)
        # cmap.set_bad('white')
        # cmap = plt.cm.ScalarMappable(cmap=cmap)
        # cmap.set_array(np.ma.masked_where(scalar_field_values == 0, scalar_field_values))
        # cbar = fig.colorbar(cmap)
        # cbar.remove()
        # #cbar.set_label('Cell response to radiation')
        #
        #
        # polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=self.r_poly,
        #                            orientation=0, facecolor=c, edgecolor=edgecolor)
        #             for x, y, c in
        #             zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        # pc = PatchCollection(polygons, match_original=True)
        # ax2.add_collection(pc)
        # ax2.set_title('t = 100')
        # ax2.set_aspect(aspect='auto')
        #

        # # tensor_vis(lgca, lgca.t - 1)
        # # ftimage = np.fft.fft2(lgca.scalar_field_t[lgca.t-1])
        # # ftimage = np.fft.fftshift(ftimage)
        # # plt.imshow(np.abs(ftimage))

        return fig, pc, cmap

    def strands(self, dens = 0.9):

        all_paths = []
        set_coords = set(self.coord_pairs)

        def path_finder(path_list, neigh, set_coords):
            for i in neigh[1:]:
                if i in set_coords:
                    if self.scalar_field[i] >= dens:
                        path_list.append(i)
                        set_coords.remove(i)
                        neigh = boundary_coord(nb_coord(i), self.lx, self.ly)
                        path_finder(path_list, neigh, set_coords)



        for i in self.coord_pairs:
            path = []
            if i in set_coords and self.scalar_field[i] >= dens:
                path.append(i)
                neigh = boundary_coord(nb_coord(i), self.lx, self.ly)
                set_coords.remove(i)
                path_finder(path, neigh, set_coords)

            if path != []:
                all_paths.append(path)


        lengths = []
        for counter, i in enumerate(all_paths):
            lengths.append((counter, len(i)))
        # print(all_paths)
        x = [item[0] for item in lengths]
        y = [item[1] for item in lengths]
        # fig ,ax = plt.subplots()
        # ax.plot(x, y)
        #
        max_value = max(y)
        max_index = y.index(max_value)

        # plt.ylabel('Path length')
        # plt.xlabel('Path Number')
        count = sum([len(listElem) for listElem in all_paths])
        # print(f'nodes with {dens} density: {count} ')
        #
        print(f'längster Pfad: {max(y)} starting at {all_paths[max_index][0]}')
        print(f'Insgesamt {len(all_paths)} Pfade mit einer durchschnittlichen Länge von {np.mean(y)}')

        return len(all_paths), np.max(y), all_paths

    def strands_fixed_border(self, dens = 0.9):

        all_paths = []
        set_coords = set(self.coord_pairs)

        def path_finder(path_list, neigh, set_coords):
            for i in neigh[1:]:
                if i in set_coords:
                    if self.scalar_field[i] >= dens:
                        path_list.append(i)
                        set_coords.remove(i)
                        neigh = nb_coord(i)
                        path_finder(path_list, neigh, set_coords)



        for i in self.coord_pairs:
            path = []
            if i in set_coords and self.scalar_field[i] >= dens:
                path.append(i)
                neigh = nb_coord(i)
                set_coords.remove(i)
                path_finder(path, neigh, set_coords)

            if path != []:
                all_paths.append(path)


        lengths = []
        for counter, i in enumerate(all_paths):
            lengths.append((counter, len(i)))
        # print(all_paths)
        x = [item[0] for item in lengths]
        y = [item[1] for item in lengths]
        # fig ,ax = plt.subplots()
        # ax.plot(x, y)
        #
        if y is []:
            max_value = max(y)

            max_index = y.index(max_value)

            # plt.ylabel('Path length')
            # plt.xlabel('Path Number')
            count = sum([len(listElem) for listElem in all_paths])
            # print(f'nodes with {dens} density: {count} ')
            #
            # print(f'längster Pfad: {max(y)} starting at {all_paths[max_index][0]}')
            # print(f'Insgesamt {len(all_paths)} Pfade mit einer durchschnittlichen Länge von {np.mean(y)}')

            return len(all_paths), np.max(y), all_paths
        else:
            return len(all_paths), 0, all_paths

    def simple_percolation(self):
        x_arr = np.arange(1, self.lx + 1)
        connected = False
        index_connected_path = []

        for counter, paths in enumerate(self.paths):
            for x in x_arr:
                if (x, 1) in paths:
                    for x2 in x_arr:
                        if (x2, 30) in paths:
                            connected = True
                            index_connected_path.append(counter)

        return connected, index_connected_path

    def connectivity(self):
        x_arr = np.arange(1, self.lx+1)
        connected = False
        index_connected_path = []

        for counter, paths in enumerate(self.paths):
            for x in x_arr:
                if (x, 1) in paths:
                    neigh = boundary_coord(nb_coord((x,1)), self.lx, self.ly)
                    y_neigh = [item[1] for item in neigh[1:]]

                    index = (np.asarray(y_neigh) == self.lx).nonzero()

                    for i in index:
                        neigh_crit = neigh[1:][i[0]]
                        if neigh_crit in paths:
                            connected = True
                            index_connected_path.append(counter)


        return connected, index_connected_path

    def plot_percolationsites(self, edgecolor = None, cutoff = 0.9):
        r_poly = 0.5 / np.cos(np.pi / 6)
        dy = 0.5 / np.sin(np.pi / 6)

        fig = plt.figure()
        ax = plt.gca()
        cmap = plt.cm.get_cmap("Greys")
        cmap.set_under(alpha=0)
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cbar = fig.colorbar(cmap)
        cbar.set_label('ECM density $d$')
        ax.set_xlabel("$x(\epsilon)$")
        ax.set_ylabel("$y(\epsilon)$")
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        scalar_field_values = self.scalar_field[self.nonborder].ravel()
        perco = (scalar_field_values >= cutoff).astype(int)
        scalar_field_values = perco
        cmap.set_array(scalar_field_values)
        polygons = [RegularPolygon(xy=(x, y), numVertices=6, radius=r_poly,
                                   orientation=0, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(scalar_field_values.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)

        return fig, pc, cmap

    def animate_percolationsites(self, lgca, interval=100, save_anim=False, cutoff = 0.9):
        fig, pc, cmap = self.plot_ECM()
        title = plt.title('Time $k =$0')

        def animate(n):

            scalar_field_values = self.scalar_field_t[n][self.nonborder]
            percolation = (scalar_field_values.ravel() >= cutoff).astype(int)
            perco = (np.sum(percolation) / 900)
            perco = (np.sum(percolation) / 900)
            title.set_text('Percolation site ratio = ${}$'.format(np.round(perco,2)))
            pc.set(facecolor=cmap.to_rgba(percolation))

            return pc, title

        anim = animation.FuncAnimation(fig, animate, interval=interval, frames=self.t)
        return anim


    def random_sites_percolation(self):
        for i in np.arange(self.random_steps):
            self.scalar_field[self.nonborder] = self.random_perc_t[i]
            connection, path_index = self.simple_percolation()
            paths, length, all_paths = self.strands_fixed_border()
            scalar_field_values = self.scalar_field[self.nonborder].ravel()
            perco = (scalar_field_values >= 0.9).astype(int)
            self.perco_ratio.append(sum(perco) / 900)

            if connection:
                self.p_inf.append(len(self.paths[path_index[0]]))
                # ecm.perco_ratio.append((sum(perco)/900))
            else:
                self.p_inf.append(0)
            #
            # ecm.list_paths_number.append(paths)
            # ecm.list_paths_length.append(length)
            self.paths = all_paths
