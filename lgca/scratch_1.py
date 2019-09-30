import pickle
import gc
import pickle

from lgca.lgca_1d import *
from lgca.lgca_square import *

gc.collect()


def run(dims=100, rest=[4], geo='1D', b=[0.2], bd=[0.05], runs=10, t_max=[5000], ini='full', mut='both',
        k_cen=0, k_ran=12, k_res=0.2, th_cen=0.5, th_ran=2, th_res=0.05, file_name=False, name_addition=''):
    # some parameters are lists, if they have multiple entries then all combinations will be simulated,
    # YOU SHOULD use one or two lists, with at most 3 items
    # DON't use when mut='kappa' or 'theta' (because such plots are not supported)

    # mut:   which parameter to mutate, the other will be fixed for the particular run, SEE ALSO k_cen - th_ran
    # dims:  size of the lattice in nodes
    # rest:  list, number of rest channels
    # geo:   which geometry is used: 1D or Square

    # b:     list, birth rate: chance of giving birth in a time step (only applies if in rest)
    # bd:    list, birth rate divided by death rate: relation of death to birth (although death may apply in any state)

    # runs:  how many times the simulation is repeated (with different random initialization if there are such)
    # t_max: list, how many time steps the simulations will last

    # ini:   cells to start with
    #       'full': all channels are filled
    #       integer value: one rest and one velocity channel of that many nodes in the center

    # k_cen - th_ran:
    #           initial distribution of kappa and theta !!! OR !!! range to vary the not mutable parameter
    # k_cen:     center of the initial distribution of kappa
    # k_ran:     range of kappa around the center
    # th_cen:    center of theta
    # th_ran:    range of theta
    # k_res:     IF kappa is not mutable, but has a range, this determines the size of the steps
    # th_res:    step size of fixed theta

    # 1 INITIATE

    par = {'dims': dims, 'rest': rest, 'b': b, 'bd': bd, 'runs': runs, 't_max': t_max, 'geo': geo, 'ini': ini,
           'mut': mut,
           'k_cen': k_cen, 'k_ran': k_ran, 'k_res': k_res, 'th_cen': th_cen, 'th_ran': th_ran, 'th_res': th_res}
    # putting all parameters in a single dictionary
    # this is used for coherence with older versions and to save in single dictionary ALL
    ALL = par  # dictionary to store all parameters and data

    # set parameters that are determined by the geometry
    if geo == '1D':
        vel = 2  # number of velocity channels
        par['num'] = dims  # number of nodes
    elif geo == 'Square':
        vel = 4
        par['num'] = dims ** 2
        print('number of nodes: ' + str(par['num']))
    ALL['vel'] = vel
    ALL['num'] = par['num']

    # initiate data variables
    if mut == 'both':  # if both parameters a mutable, the end result shows a heat map of kappa-theta combinations, therefore all values are stored
        ALL[
            'kappas'] = []  # lists of the kappas of all cells of a single run and parameter combination, 3-dimensional: [run][parameter combination][cell indices]
        ALL['thetas'] = []
    else:  # if only kappa or theta are mutable, the endresult shows the mean of the mutable parameter, therefore only mean and variance are stored
        ALL[
            'm_kappas'] = []  # lists of MEANS of the kappas of all cells of a single run and fixed parameter, 2-dimensional: [run][fixed parameter]
        ALL['m_thetas'] = []
        ALL['v_kappas'] = []  # lists variances ...
        ALL['v_thetas'] = []

    # initiate the fixed (not mutable) parameter
    # program will simulate each step (and repeat for each run)
    if par['mut'] == 'both':  # there are no fixed parameters of both are mutable
        par['fix'] = [0]
    elif par['mut'] == 'kappa':
        par['fix'] = np.arange(-th_ran + th_cen, th_ran + th_cen + 0.0001, th_res)  # fixed values for theta
    elif par['mut'] == 'theta':
        par['fix'] = np.arange(-k_ran + k_cen, k_ran + k_cen + 0.0001, k_res)

    # 2 RUN SIMULATION
    # for mytry in range(10):

    for attempt in range(10):

        # create name of the savefile
        if par['ini'] != 'full':
            s_ini = '_center' + str(par['ini'])
        else:
            s_ini = ''
        s_par = '_' + str(par['rest']) + str(par['b']) + str(par['bd'])
        s_sim = '_' + str(par['runs']) + str(par['t_max'])
        if len(par['fix']) > 1:
            s_fix = '_[' + str(par['fix'][0].__round__(3)) + ',' + str(
                par['fix'][-1].__round__(3)) + ']_step_' + str(
                (par['fix'][1] - par['fix'][0]).__round__(3))
        elif par['mut'] == 'both':
            s_fix = ''
        else:
            s_fix = '_fixed_at' + str(par['fix'][0].__round__(3))

        if file_name == False:
            full_name = 'S_' + geo + '_' + par[
                'mut'] + s_ini + s_par + s_sim + s_fix + '_' + name_addition + '.pkl'
        else:
            full_name = file_name
        print(full_name)

        try:
            # a = 1 / int(attempt)

            for i in range(par['runs']):  # repeat for each run
                # initiate new temporary data variables
                # they will be discarded after each run
                if mut == 'both':
                    kappas = []  # list of the kappas of a single run, 2-dim: [parameter combination][cell indices]
                    thetas = []
                else:
                    m_kappas = []  # list of MEANS of the kappas of a single run, 1-dim: [fixed parameter]
                    m_thetas = []
                    v_kappas = []  # list of the variance ...
                    v_thetas = []

                for r in par['rest']:  # loop through list of number of rest channels
                    n_channels = r + par['vel']  # total number of channels
                    if par['ini'] != 'full':  # if only a few central nodes are filled initially
                        if geo == '1D':
                            nodes = np.zeros((par['num'], n_channels),
                                             dtype=bool)  # create lattice, all channels false
                            left_node = int(len(nodes) / 2 - par['ini'] / 2)  # most left node that will be filled
                            nodes[left_node:left_node + par['ini'], [0, -1]] = 1
                            #   only par['ini'] many nodes in the center are 'filled'
                            #   'filling' meaning one rest and one velocity channel
                            #   therefore simple trick: the first and the last [0,-1]
                        else:
                            print('ERROR: initiation of filled center area only works for 1D')

                    for b in par['b']:  # loop through list of birth rates
                        for d in par['bd']:  # ... birth to death ratio
                            for f in par['fix']:  # ... fixed parameter (may be kappa or theta)
                                for t in par['t_max']:  # ... of number of time steps
                                    # print(i, r, b, d, f, t)  # to keep track of progress

                                    # each starting cell gets random or fixed value
                                    if par['mut'] == 'both':
                                        KAPPAS = npr.random(par[
                                                                'num'] * n_channels) * k_ran * 2 - k_ran + k_cen  # uniform distribution of kappas
                                        THETAS = npr.random(par['num'] * n_channels) * th_ran * 2 - th_ran + th_cen
                                    elif par['mut'] == 'kappa':
                                        KAPPAS = npr.random(par['num'] * n_channels) * k_ran * 2 - k_ran + k_cen
                                        THETAS = np.ones(
                                            par['num'] * n_channels) * f  # all thetas set to fixed value
                                    elif par['mut'] == 'theta':
                                        KAPPAS = np.ones(par['num'] * n_channels) * f
                                        THETAS = npr.random(par['num'] * n_channels) * th_ran * 2 - th_ran + th_cen

                                    if par['ini'] == 'full':
                                        if geo == '1D':
                                            lgca2 = IBLGCA_1D(density=1, bc='reflect', interaction='go_or_grow',
                                                              kappa=list(KAPPAS),
                                                              r_b=b, r_d=b * d, theta=list(THETAS), restchannels=r,
                                                              dims=par['dims'],
                                                              mut=par['mut'])  # iniate lgca object
                                        elif geo == 'Square':
                                            lgca2 = IBLGCA_Square(density=1, bc='reflect', interaction='go_or_grow',
                                                                  kappa=list(KAPPAS),
                                                                  r_b=b, r_d=b * d, theta=list(THETAS),
                                                                  restchannels=r, dims=par['dims'], mut=par['mut'])
                                    else:
                                        if geo == '1D':
                                            use_par = par[
                                                          'ini'] * 2  # indicates how many channels are currently occupied
                                            #   NOTE: if each 'occupied' node has one of each channels, then multiple by 2
                                            #   otherwise (all channels are filled), multiply by n_channels
                                            lgca2 = IBLGCA_1D(nodes=nodes, bc='reflect', interaction='go_or_grow',
                                                              kappa=list(KAPPAS[:use_par]), r_b=b, r_d=b * d,
                                                              theta=list(THETAS[:use_par]), restchannels=r,
                                                              dims=par['dims'])
                                        else:
                                            print('ERROR: initiation of filled center area only works for 1D')

                                    lgca2.timeevo(timesteps=t, recordLast=True,
                                                  showprogress=False)  # run simulation

                                    # update data variables for this parameter combination
                                    if mut == 'both':
                                        kappas.append(lgca2.props['kappa'])  # eigentlich nur letzte Hälfte ...
                                        thetas.append(lgca2.props['theta'])
                                    else:
                                        kappas = lgca2.props['kappa']
                                        thetas = lgca2.props['theta']
                                        m_kappas.append(np.mean(kappas[int(0.5 * len(
                                            kappas)):-1]))  # average the last half of all cells that ever existed
                                        v_kappas.append(np.var(kappas[int(0.5 * len(kappas)):-1]))
                                        m_thetas.append(np.mean(thetas[int(0.5 * len(thetas)):-1]))
                                        v_thetas.append(np.var(thetas[int(0.5 * len(thetas)):-1]))
                                    gc.collect()  # clear space in the memory
                # update data variables for this run
                if mut == 'both':
                    ALL['kappas'].append(kappas)
                    ALL['thetas'].append(thetas)
                else:
                    ALL['m_kappas'].append(m_kappas)
                    ALL['m_thetas'].append(m_thetas)
                    ALL['v_kappas'].append(v_kappas)
                    ALL['v_thetas'].append(v_thetas)
        except:
            if par['runs'] > 5:
                par['runs'] = par['runs']
            par['t_max'] = list(np.round(np.array(par['t_max']) * 0.9))
            print('failed')
            continue
        else:
            ALL['runs'] = par['runs']
            ALL['t_max'] = par['t_max']

            with open(full_name, 'wb') as handle:
                pickle.dump(ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(full_name + '_successfully saved!')

            with open('S_CURRENT.pkl', 'wb') as handle:
                pickle.dump(ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return ALL
            break

    # from P01 import plot01
    # plot01(ALL=ALL)

# run( runs=2, t_max=[50], name_addition='test', geo='Square', dims=25)
