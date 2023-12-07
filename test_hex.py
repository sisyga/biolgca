import matplotlib.pyplot as plt
import numpy as np
from analysis import initial_values, initial_values_hom
from lgca.ecm import Ecm, nb_coord
from lgca import get_lgca
from data import DataClass
from mpl_toolkits.axes_grid1 import host_subplot


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
            order_arr[re] = q
            dens_arr[re] = np.sum(ecm.scalar_field[ecm.nonborder])/lgca.lx**2
            re += 1
    return np.mean(order_arr), np.mean(dens_arr)

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

def main():
    lx = 20
    ly = 20
    K = 6
    restchannels = 2
    timesteps = 1
    r_b = 0.0
    d = 2
    d_neigh = d/6
    beta_agg = 0.1

    beta = 10
    beta_rest = 0
    fc =.1

    nodes = np.zeros((lx, ly, K + restchannels))
    t = 0
    data = DataClass()
    lgca = get_lgca("hex", lx=lx, ly=ly, nodes=nodes, interaction='contact_guidance'
                            , beta=beta, beta_agg=beta_agg, beta_rest=beta_rest, r_b=r_b)

    ecm = Ecm(fc, lx, ly, restchannels, d, d_neigh, timesteps, t)
    # ini_coords = initial_values(lgca, ecm)
    initial_values_hom(lgca, ecm)
    # lgca.plot_density()
    # ecm.plot_ECM(show_tensor=True, edgecolor='black')
    ecm.plot_ECM(show_tensor=True, edgecolor='black')

    lgca.timeevo(timesteps=100, record=True, ecm=ecm, data=data)
    lgca.plot_density()
    ecm.plot_ECM(show_tensor=True, edgecolor='black')
    cells_t0 = np.sum(lgca.cell_density[lgca.nonborder])
 
    plt.show()
    # if input close all plots
  

if __name__ == '__main__':
    main()

