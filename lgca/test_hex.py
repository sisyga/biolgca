import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
print(sys.path)

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import numpy as np
from Analysis import initial_values, random_points, grid_points, neigh, initial_values_hom, ind_coord
from mypackage.ECM import Ecm, nb_coord
from mypackage.__init__ import get_lgca
from data import DataClass
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import mplot3d
import csv



import itertools
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
# from Analysis import dens_r_plot


def data_plot(fc, lx, ly, restchannels, d, d_neigh, timesteps, t, nodes, beta, beta_agg, r_b):
    Abstand = []
    # betas = [0, 0.2, 0.4]
    repeats = 10
    data = DataClass(x=1, y=repeats, z=timesteps)
    print(data)
    data.y_counter = 0

    for j in np.arange(repeats):
        lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                            , beta=beta, beta_agg=beta_agg, r_b=r_b)

        ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)
        initial_values(lgca, ecm)
        lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
        data.y_counter += 1

    print(data.Abstand)
    print('axis 1 ', np.mean(data.Abstand, axis=1))
    means = np.mean(data.Abstand, axis=1)

    plt.figure()
    x = np.arange(timesteps)
    plt.plot(x, means[0], label=f'{d_neigh}')
    # plt.plot(x, means[1], label=f'{d_neigh[1]}')
    # plt.plot(x, means[2], label=f'{d_neigh[2]}')
    plt.legend()

def main():

       # ecm_loadin = np.loadtxt('ecm.txt')

       lx = 10
       ly = 10
       K = 6
       restchannels = 6
       timesteps = 1
       r_b = 0.2
       d = 0.1
       d_neigh = d/6
       beta_agg = 0.1

       beta = 10
       beta_rest =0
       fc =.2

       # r_b and beta_agg 0.2 for metasttic growth
       # r_b = 0.00000
       # d = 0.008
       # d_neigh = 0.002
       # beta_agg = 0.000
       # beta = 5
       nodes = np.zeros((lx, ly, K + restchannels))

       t = 0


       def mean_order_param(ecm, lgca):
           repeats = 1
           re = 0
           order_arr = np.zeros(repeats)
           dens_arr = np.zeros_like(order_arr)
           while re < repeats:
                   # ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)
                   nx = ecm.vector_field[:, :, 0]
                   ny = ecm.vector_field[:, :, 1]
                   ev = ecm.vector_field[:, :, 2]
                   scalar = ecm.scalar_field
                   eins = np.array([nx[0,1], ny[0,1]])
                   zwei = np.array([nx[0,2], ny[0,2]])
                   angle = np.dot(eins, zwei)
                   # print(angle)
                   all_list = []
                   angle_list  = []
                   for x in np.arange(2, lgca.lx):
                           for y in np.arange(2, lgca.lx):

                               nbs = nb_coord((x,y), lgca)
                               nbs_angle = []
                               angles = []
                               for i in nbs[1:]:

                                   angle = np.dot(np.array([nx[x,y], ny[x,y]]), np.array([nx[i], ny[i]]))
                                   q = angle**2
                                   # angle_grad = np.arccos(angle) / (2 * np.pi) * 360

                                   # print(angle)
                                   # print(x, y, i)
                               # print(angle)
                               #     print(x,y, i)
                               #     print((np.arccos(angle) / (2 * np.pi)) * 360)
                                   nbs_angle.append(q)
                                   angles.append(q)
                               # print(x,y , angles)
                               angle_list.append(np.mean(angles))
                               all_list.append(np.mean(nbs_angle))

                   q = 2*(np.mean(all_list) - 1/2)
                   nx_squared = np.sum((1-scalar)*nx**2)/lgca.lx**2
                   ny_squared = np.sum((1-scalar)*ny ** 2) / lgca.lx ** 2
                   nxy = np.sum(scalar*ny *nx) / lgca.lx ** 2
                   nyx = np.sum(scalar * nx*ny) / lgca.lx ** 2
                   order_arr[re] = q
                   dens_arr[re] = np.sum(ecm.scalar_field[ecm.nonborder])/lgca.lx**2
                   # return q,  np.sum(ecm.scalar_field[ecm.nonborder])/lgca.lx**2

                   re += 1
           return np.mean(order_arr), np.mean(dens_arr)

       def speedup(data):
           steps = 10
           contact_beta = np.linspace(1, 10, steps)
           fcs = 1/contact_beta
           print(contact_beta)
           print(1/fcs)
           speed_array = np.zeros((steps, steps))
           print(np.shape(speed_array))
           re = 0
           repeats = 10
           while re < repeats:
               print(re)
               for i in np.arange(steps):
                   print(i)
                   for j in np.arange(steps):
                       lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                       , beta=contact_beta[j], beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

                       ecm = Ecm(fcs[i], lx, ly, restchannels, d, d_neigh, timesteps, t)
                       initial_values_hom(lgca, ecm)
                       lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)

                       restc = lgca.nodes[lgca.nonborder][:, :, -lgca.restchannels:]
                       veloc = lgca.nodes[lgca.nonborder][:, :, :lgca.velocitychannels]
                       speed_array[i,j] += np.sum(veloc) / lgca.lx**2
               re += 1

           plt.imshow(speed_array/repeats, cmap='bwr', origin='lower', interpolation='bilinear')
           print(list(speed_array/repeats))
           plt.colorbar()

       def MSD(data, ecm, lgca):
           repeats = 1
           re= 0
           b = np.array([15.0, 12.990381056766578])
           MSD_time = np.zeros(timesteps)
           MSD_diff = np.zeros(timesteps)
           while re <= repeats:
               data.MSD  = []
               lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                               , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

               ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)

               initial_values(lgca, ecm)
               lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
               lgca.animate_density()
               time_average = []
               for counter,i in enumerate(data.MSD):
                   dist = np.linalg.norm(i - b)**2
                   if counter > 0 :
                       dist_1 = np.linalg.norm(data.MSD[counter-1] - i)
                   else:
                       dist_1 = 0
                   time_average.append(dist_1)
                   MSD_time[counter] += dist
                   MSD_diff[counter] += dist_1
               re += 1

           plt.figure()
           plt.plot(np.arange(len(MSD_time)), MSD_time/repeats, marker = "s")
           plt.ylim((0, 2.5))
           plt.legend()
           # x = np.arange(timesteps)
           # y = np.zeros_like(x)
           # for i in np.arange(len(y)):
           #     y[i] = np.sum((ecm.scalar_field_t[i][ecm.nonborder]))
           # plt.figure()
           # plt.plot(x[0:],y[0:])
           plt.figure()
           ecm.plot_ECM()
           anim = lgca.animate_density()
           plt.show()

       def patterns(repeats=1):
           re = 1
           d_neigh = np.linspace(0, d/2, 10, endpoint=True,)
           print(d_neigh)
           q_array = np.zeros_like(d_neigh)
           dens_array = np.zeros_like(d_neigh)
           while re <= repeats:
               print('repeats=', re)
               q_arr = np.zeros_like(d_neigh)
               dens_arr = np.zeros_like(d_neigh)
               for i in np.arange(len(d_neigh)):
                   nodes = np.zeros((lx, ly, K + restchannels))
                   t = 0
                   data = DataClass()
                   lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                   , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

                   ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh[i], timesteps, t)
                   # ecm.scalar_field = ecm_loadin
                   # ini_coords = initial_values(lgca, ecm)
                   initial_values_hom(lgca, ecm)
                   # grid_points(lgca, ecm)
                   lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
                   ecm.scalar_field = np.around(ecm.scalar_field, decimals=3)
                   # ecm.scalar_field = np.round(ecm.scalar_field, 5)
                   ecm.tensor_update(t)
                   q, dens = mean_order_param(ecm=ecm, lgca=lgca)
                   dens_arr[i] = dens
                   q_arr[i] = q

               q_array += q_arr
               dens_array += dens_arr


               # with open("orderparam_smallbeta.txt", "a") as f:
               #     f.write(str(q_arr) + '\n')
               # with open("density_smallbeta.txt", "a") as f:
               #     f.write(str(dens_arr) + "\n")

               re += 1
           q = q_array/repeats
           rho = dens_array/repeats
           host = host_subplot(111)
           ax2 = host.twinx()
           host.set_xlabel("synthesis rate")
           host.set_ylabel("Order Parameter q")
           host.set_ylim([-0.35, 0.05])
           ax2.set_ylabel("Mean density")
           ax2.set_ylim([-0.05, 1.05])


           p1, = host.plot(d_neigh, q, label='order parmeter', color='orange', marker = 's')
           p2, = ax2.plot(d_neigh, rho, label='mean density',  color='blue', marker = 's')


           leg = plt.legend()

           host.yaxis.get_label().set_color(p1.get_color())
           leg.texts[0].set_color(p1.get_color())

           ax2.yaxis.get_label().set_color(p2.get_color())
           leg.texts[1].set_color(p2.get_color())

       # plt.imshow(test_array, cmap='bwr', origin='lower', interpolation='bilinear')
       def adhesion_prolif(data, repeats=1):
           steps = 5
           d1 = np.linspace(0.0, 1, 10)
           # r1 = np.array([0.54444444, 0.62222222, 0.7])
           agg = np.linspace(0.0, 0.5, 10)

           neighbours = np.zeros((len(d1), len(agg)))
           cellnumbers = np.zeros_like(neighbours)
           for i in np.arange(len(d1)):
               for j in np.arange(len(agg)):
                    repeats_list = []
                    number_of_cells = []
                    while len(repeats_list) < repeats:
                            # print(len(repeats_list))
                            print(d1[i], agg[j])
                            nodes = np.zeros((lx, ly, K + restchannels))
                            lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                            , beta=beta, beta_agg=agg[j], beta_rest=beta_rest, r_b=r_b)

                            ecm = Ecm(fc, lx, ly, restchannels, d1[i], d1[i]/6, timesteps, t)
                            ini_coords = initial_values(lgca, ecm)

                            lgca.timeevo(timesteps=timesteps, record= True, ecm=ecm, data=data)

                            neigh_values = []
                            nb_Sum = lgca.nb_sum(lgca.cell_density)
                            for h in lgca.coord_pairs:
                                if lgca.cell_density[h] >= 1 and h not in ini_coords:
                                    neigh_values.append(nb_Sum[h]+lgca.cell_density[h])
                            print(sum(sum(lgca.cell_density[lgca.nonborder])))
                            number_of_cells.append(sum(sum(lgca.cell_density[lgca.nonborder])))
                            repeats_list.append(np.mean(neigh_values))
                            print(np.mean(neigh_values))

                    neighbours[i, j] = np.mean(repeats_list)
                    cellnumbers[i, j] = np.mean(number_of_cells)
                    #
                    # with open("neigh.txt", "a") as f:
                    #     f.write(str(repeats_list) + '\n')
                    # with open("cellnumbers.txt", "a") as f:
                    #     f.write(str(number_of_cells) + "\n")

           # np.savetxt("neigh_repeats.txt", neighbours)
           # np.savetxt("numbers_repeats.txt", cellnumbers)
           fig = plt.figure(figsize=(7, 7))
           ax = fig.add_subplot(projection='3d')
           ax.set_zlim3d(700, 1300)
           agg, d1 = np.meshgrid(agg, d1)
           color = neighbours
           norm = mcolors.Normalize(6, 15)
           ax.plot_surface(agg, d1, cellnumbers, facecolors=plt.cm.jet(norm(color)))
           m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
           m.set_array([])
           plt.colorbar(m)

       def mean_abstand_50(lgca, repeats = 50):
           d_array = np.linspace(0, 1, 10)
           re = 0
           abstand_array = np.zeros_like(d_array)
           while re < repeats:
               print(re)
               for i in np.arange(len(d_array)):
                   d_neigh = d_array[i]/30
                   nodes = np.zeros((lx, ly, K + restchannels))
                   t = 0
                   data = DataClass()
                   lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                   , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

                   ecm = Ecm(fc, lx, ly, restchannels, d_array[i], d_neigh, timesteps, t)
                   ini_coords = initial_values(lgca, ecm)
                   lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)

                   mittelpunkt= np.array([15, 12.90381056766578])
                   abstand = []
                   sum_part = 0
                   for h in lgca.coord_pairs:
                       if lgca.cell_density[h] >= 1:
                           sum_part += lgca.cell_density[h]
                           j = ind_coord(lgca.coord_pairs, lgca.coord_pairs_hex, [h])
                           abstand.append(lgca.cell_density[h]*np.linalg.norm(np.array([j[0], j[1]]) - mittelpunkt))
                   mean_distance = np.sum(abstand)/sum_part
                   abstand_array[i] += mean_distance
               re += 1
           abstaende = abstand_array/repeats
           fig, ax = plt.subplots()
           ax.plot(d_array, abstaende, marker='s', ls = "")

       def tumor_vol(repeats = 50):
           re = 0
           data = DataClass()
           data.tumor_vol(timesteps)
           while re < repeats:
               print('repeats=', re)


               nodes = np.zeros((lx, ly, K + restchannels))
               t = 0
               lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                   , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

               ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)

               ini_coords = initial_values(lgca, ecm)
               lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
               re += 1

           plt.plot(np.arange(len(data.tumor_vol)), data.tumor_vol/repeats, marker = 's', label="a")
           plt.ylim([1000, 7000])
           plt.legend()

       def outside(lgca, ecm, data, repeats=10):
           d = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
           # d  = [0.05, 0.2, 0.5]
           d_neigh_steps = 10
           outside_all = np.zeros((len(d), d_neigh_steps))

           for d_ind in np.arange(len(d)):
               d_neigh = np.linspace(0,d[d_ind],d_neigh_steps)
               for i in np.arange(len(d_neigh)):
                   re = 0
                   while re < repeats:
                   # print(len(repeats_list))
                       nodes = np.zeros((lx, ly, K + restchannels))
                       lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                                       , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

                       ecm = Ecm(fc, lx, ly, restchannels, d[d_ind], d_neigh[i], timesteps, t)
                       ini_coords = initial_values(lgca, ecm)

                       lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)

                       outside = 0
                       for h in lgca.coord_pairs:
                           if lgca.cell_density[h] >= 1 and h not in ini_coords:
                               outside += lgca.cell_density[h]
                       outside_all[d_ind][i] += outside/sum(sum(lgca.cell_density[lgca.nonborder]))
                       re += 1
           for counter, i in enumerate(outside_all):
               plt.plot(np.linspace(0, 1, d_neigh_steps), i, marker='s', label=f"{d[counter]}")
           plt.legend()
            # plt.plot(d_neigh, outside_arr/repeats, marker="s")

       data = DataClass()


       lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                               , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

       ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)
               # ecm.scalar_field = ecm_loadin
       ini_coords = initial_values(lgca, ecm)
       initial_values_hom(lgca, ecm)


       # time evolution
       lgca.timeevo(timesteps=0, record=True, ecm=ecm, data=data)
       cells_t0 = np.sum(lgca.cell_density[lgca.nonborder])
       # lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
       print(sum(sum(lgca.cell_density[lgca.nonborder])))
       speedup(data)
       # adhesion_prolif(data, repeats=20)
       # neigh_values = []
       # nb_Sum = lgca.nb_sum(lgca.cell_density)
       # for h in lgca.coord_pairs:
       #     if lgca.cell_density[h] >= 1 and h not in ini_coords:
       #         neigh_values.append(nb_Sum[h])
       # print(sum(sum(lgca.cell_density[lgca.nonborder])))
       # print(np.mean(neigh_values))
       # outside(lgca, ecm, data, repeats=10)
       # tumor_vol()
       # adhesion_prolif(data)
       # NICHE
       # ecm bigger zebra
       # lx = 50 and second midpoint
       # radius=80 p_apopt = 0.7
       # mean_abstand_50(lgca)
       # adhesion_prolif(data)
       # patterns()
       # with open('cellnumbers.txt', 'r') as f:
       #      data = f.read().rstrip()


       # print(np.mean(values))
       # if ecm.scalar_field_t[0] == ecm.scalar_field_t[-1]:
       #     print('mmgm')

       # data_plot(timesteps, lx, ly, nodes, beta, beta_agg, r_b, d, t)

       # plots
       # plt.figure()
       # ecm.plot_ECM()
       # ecm.plot_response_to_radiation(lgca)
       # ecm.plot_neigh_occupied(lgca)
       # ecm.plot_neigh_occupied(lgca)

       # plt.figure()
       # lgca.plot_density()
       # plt.figure()
       # lgca.plot_density()
       # ecm.plot_ECM(edgecolor='black')
       # ecm.plot_onlytensor()
       # print(ecm.scalar_field)
       # np.savetxt('ecm.txt', ecm.scalar_field)
       # print(f'shape = {np.shape(ecm.scalar_field)}')

       # ecm.strands()

       # plt.figure()
       # a = lgca.animate_density()
       # plt.figure()

       ######## Animations  ###########
       # _ = ecm.animate_ECM(lgca, interval=500)
       # ecm.plot_onlytensor()
       # plt.figure()
       ecm.plot_ECM(show_tensor=True, edgecolor='black')
       # plt.figure()
       # ecm.plot_ECM(show_tensor=True)
       # plt.figure()
       # anim2 = lgca.animate_density(interval= 500)
       print(lgca.cell_density)

       for h in lgca.coord_pairs:
           if lgca.cell_density[h] >= 1 and h in ini_coords:
               lgca.cell_density[h] = 0
       lgca.plot_density()
       a,b = mean_order_param(ecm, lgca)
       print(a,b)
       # f = fr"c://Users/tompa/Desktop/CELLS_{timesteps}ts_{lgca.lx}x{lgca.ly}_aggr={lgca.beta_agg}_guid={lgca.beta}_deg={ecm.d}_d_neig={ecm.d_neigh}.gif"
       # anim2.save(f, fps=5, dpi=80)
       # lgca.plot_config(grid=True)
       # lgca.plot_density(edgecolor='black')
       # plt.figure()
       # anim2 = ecm.animate_percolationsites(lgca)
       # plt.figure()
       # ecm.plot_percolationsites()
       # ecm.plot_ECM(show_tensor=True)
       # plt.figure()
       # lgca.plot_density()
       # plt.figure()
       # ecm.plot_ECM(show_tensor=True)
       # lgca.plot_density()

       ################################
       # # plt.savefig('c://Users/tompa/Desktop/animation.png')
       # plt.figure()

       # plt.figure()
       # ecm.plot_ECM()
       # lgca.plot_density()
       # ecm.tensor_vis(ecm.t-1)
       # anim_ecm = ecm.animate_ECM(lgca, save_anim= True)

       # f = fr"c://Users/tompa/Desktop/CELLS_{timesteps}ts_{lgca.lx}x{lgca.ly}_aggr={lgca.beta_agg}_guid={lgca.beta}_deg={ecm.d}_d_neig={ecm.d_neigh}.mp4"
       # anim2.save(f, fps=5, dpi=80)
       # f2 = fr"c://Users/tompa/Desktop/ECM_{timesteps}ts_{lgca.lx}x{lgca.ly}_aggr={lgca.beta_agg}_guid={lgca.beta}_deg={ecm.d}_d_neig={ecm.d_neigh}.mp4"
       # b.save(f2, fps=5, dpi=80)
       # print(data.MSD)


       # print(np.mean(time_average))
       # ecm_histo = ecm.animate_histo()
       # print("cell after/cells before ", sum(sum(lgca.cell_density[lgca.nonborder])), "/", cells_t0)
       # neigh_v = neigh(lgca)
       # print(neigh_v)
       #ecm.animate_ECM(lgca, save_anim=True)
       # dens_r_plot(timesteps=timesteps, lx=lx, ly=ly, beta=beta, nodes=nodes, beta_agg=beta_agg, r_b=r_b, t=0, d=d, d_neigh=d_neigh)


       #histogram1, histogram2 = hist(lgca)
       #
       # print(sum(ecm.scalar_field[ecm.nonborder].ravel()))

       # print(ecm.list_paths_length)

       # x = np.arange(len(ecm.list_paths_number))
       # x2 =  np.arange(len(ecm.list_paths_length))
       #

       # plt.figure()
       # plt.plot(x, ecm.list_paths_number)
       # plt.xlabel('timestep')
       # plt.ylabel('Number of paths')
       # plt.figure()
       # plt.xlabel('timestep')
       # plt.ylabel('longest Path length')
       # plt.plot(x2, ecm.list_paths_length)
       # plt.figure(x2, ecm.list_paths_length)
       # connection, path_index = ecm.connectivity()

       list_percolation = []
       list_sites = []
       list_percolation_when = []
       counter = 0
       # for i in np.arange(1):
       #
       #     data = DataClass()
       #     lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
       #                     , beta=beta, beta_agg=beta_agg, r_b=r_b)
       #
       #     ecm = Ecm(lx, ly, d, d_neigh, timesteps, t)
       #
       #     grid_points(lgca, ecm)
       #
       #     lgca.timeevo(timesteps=timesteps, record=True, ecm=ecm, data=data)
       #     connection, path_index = ecm.simple_percolation()
       #
       #     print(f'is connected = {connection}')
       #     # ecm.plot_percolationsites()
       #
       #     if connection:
       #          counter += 1
       #
       #     scalar_field_values = ecm.scalar_field[lgca.nonborder].ravel()
       #     perco = (scalar_field_values >= 0.9).astype(int)
       #     plt.figure()
       #     x = np.arange(len(ecm.p_inf))
       #     print(len(x))
       #     plt.plot(x, ecm.p_inf / (sum(perco)))
       #     plt.plot(x, ecm.perco_ratio)
       #     # plt.figure()
       #     ecm.animate_percolationsites(lgca)
       #     # plt.figure()
       #     # ani = ecm.animate_ECM(lgca)
       #     # anim2 = lgca.animate_density()
       #
       #     print(counter)

       # ecm.random_sites_percolation()
       # print(list_percolation_when)
       # an = ecm.animate_percolationsites(lgca, save_anim=True)
       # ecm.plot_percolationsites()
       #
       # scalar_field_values = ecm.scalar_field[lgca.nonborder].ravel()
       # perco = (scalar_field_values >= 0.9).astype(int)
       #
       # fig, ax1 = plt.subplots()
       # ax2 = ax1.twinx()
       #
       # x = np.arange(len(ecm.p_inf))
       # print(len(x))
       # ax1.plot(x, ecm.p_inf/(sum(perco)), color = 'b')
       # ax2.plot(x, ecm.perco_ratio, color = 'orange')
       # ax1.set_xlabel('time t')
       # ax1.set_ylabel('$P_{incluster}$', color ='blue')
       # ax2.set_ylabel('Percolation sites/total lattice sites $P_{perc}$', color = 'orange')
       #
       # dens_r_plot(timesteps, lx, ly, nodes, beta, beta_agg, r_b, t, d, d_neigh)
       # 0.45: 0.8362104039523395
       # 0.4: 0.6353557639271925
       # 0.35
       # 0.45233232346745
       # 0.3: 0.33678985107556536
       # 0.25: 0.24406224406224414
       # 0.2: 0.1849189189189189
       # 0.15: 0.14437837837837839
       # 0.1: 0.1054054054054054
       # 0.05: 0.08637837837837839
       #
       # the
       # most
       # common
       # form
       # of
       # celll
       # death
       # from radiation is mitotic
       # Death(Cells
       # die
       # attempting
       # to
       # divide
       # because
       # of
       # damaged
       # chromosomes)
       #




       plt.show()
if __name__ == '__main__':
    main()

