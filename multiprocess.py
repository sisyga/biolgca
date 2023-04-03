from pathos.pools import ParallelPool as Pool
from itertools import product
import numpy as np
from lgca import get_lgca
from tqdm import tqdm

PATH = '.\\data\\gog\\nonlocaldensity_5more_reps\\'

def iteration(args):
    from pickle import dump
    index, kwargs = args
    lgca = get_lgca(**kwargs)
    lgca.timeevo(kwargs['tmax'], record=False, showprogress=False, recorddens=False, recordN=False)
    data = {'nodes_t': lgca.nodes[lgca.nonborder], 'kappa': lgca.props['kappa'], 'lgca_params': kwargs}
    # save data using pickle
    with open(kwargs['PATH'] + 'data{}.pkl'.format(index), 'wb') as f:
        dump(data, f)
    return


def preprocess(variablearray, reps, **constparams):
    params = {**constparams}
    paramstobeiterated = [(i + (j,), dict(params, **p)) for (i, p), j in product(np.ndenumerate(variablearray),
                                                                                 range(reps))]
    return paramstobeiterated


def multiprocess(function, iterator, **poolkwargs):
    with Pool(**poolkwargs) as pool:
        results = list(tqdm(pool.imap(function, iterator), total=len(iterator)))
        pool.close()
        pool.join()

    return results


def postprocess(result, arr):
    for index, n_t in result:
        arr[index] = n_t

    return arr


if __name__ == '__main__':
    restchannels = 1
    l = 1001
    dims = l,
    capacity = 100
    tmax = 1000
    kappa_max = 4
    kappa_std = 0.05 * kappa_max
    # interaction parameters
    r_b = 1.  # initial birth rate
    # r_d = 0.5 * r_b / 2  # initial death rate

    nodes = np.zeros((l,) + (2 + restchannels,), dtype=int)
    nodes[l // 2, -1] = capacity

    # rhoeq = 1 - r_d / r_b
    reps = 5  # number of repetitions of for each parameter
    r_ds = np.linspace(0, .25, 6)
    thetas = np.linspace(0, 1., 11)
    kappa = np.random.random(r_ds.shape + thetas.shape + (reps, capacity)) * kappa_max * 2 - kappa_max
    # lp = len(ps)
    # lk = len(ks)

    constparams = {'l': l, 'ib': True, 'bc': 'reflect', 'interaction': 'go_or_grow_kappa', 've': False, 'geometry': 'lin',
                  'capacity': capacity, 'r_b': r_b, 'tmax': tmax, 'restchannels': restchannels, 'nodes': nodes,
                   'kappa_std': kappa_std, 'PATH': PATH}
    params = np.empty((len(r_ds), len(thetas), reps), dtype=object)
    for i, r_d in enumerate(r_ds):
        for j, t in enumerate(thetas):
            for k in range(reps):
                params[i, j, k] = {'kappa': kappa[i, j, k], 'r_d': r_d, 'theta': t}
                params[i, j, k] = {'constparams': {**constparams}}
                params[i, j, k]['r_d'] = r_d
                params[i, j, k]['theta'] = t
                params[i, j, k]['kappa'] = kappa[i, j, k]

    np.savez(PATH+'params.npz', constparams=constparams, r_ds=r_ds, thetas=thetas)
    paramstobeiterated = preprocess(params, 1, **constparams)  # as each repetition needs a different kappa, we need to pretend it is a different parameter, and only have one repetition
    result = multiprocess(iteration, paramstobeiterated, nodes=7)
    # n_pr = np.empty(params.shape + (reps,), dtype=object)
    # n_pr = postprocess(result, n_pr)
    # np.save(PATH+'n_pr.npy', n_pr)
