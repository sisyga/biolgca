from pathos.pools import ParallelPool as Pool
from itertools import product
import numpy as np
from lgca import get_lgca
from tqdm import tqdm

PATH = '.\\data\\gog\\'

def iteration(args):
    index, kwargs = args
    lgca = get_lgca(**kwargs)
    lgca.timeevo(kwargs['tmax'], record=True, showprogress=False, recorddens=False, recordN=False)
    return index, lgca


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
    l = 1000
    dims = l, l
    capacity = 100
    tmax = 100
    kappa_max = 4
    # interaction parameters
    r_b = 1.  # initial birth rate
    # r_d = 0.5 * r_b / 2  # initial death rate

    # nodes = np.zeros((l,) + (2 + restchannels,), dtype=int)
    # nodes[l // 2, -1] = capacity
    # kappa = np.random.random(capacity) * kappa_max * 2 - kappa_max

    # rhoeq = 1 - r_d / r_b
    r_ds = np.linspace(0., .5, 3)
    thetas = np.linspace(0, 1, 3)
    # lp = len(ps)
    # lk = len(ks)
    reps = 1  # number of repetitions of for each parameter
    constparams = {'l': l, 'ib': True, 'bc': 'reflect', 'interaction': 'go_or_grow', 've': False, 'geometry': 'lin',
                  'capacity':capacity, 'r_b': r_b, 'tmax': tmax}
    params = np.empty((len(r_ds), len(thetas)), dtype=object)
    for i, r_d in enumerate(r_ds):
        for j, t in enumerate(thetas):
            params[i, j] = {'constparams': {**constparams}}
            params[i, j]['r_d'] = r_d
            params[i, j]['theta'] = t

    np.savez(PATH+'params.npz', constparams=constparams, r_ds=r_ds, thetas=thetas)
    paramstobeiterated = preprocess(params, reps, **constparams)
    result = multiprocess(iteration, paramstobeiterated, processes=7)
    n_pr = np.empty(params.shape + (reps,), dtype=object)
    n_pr = postprocess(result, n_pr)
    np.save(PATH+'n_pr.npy', n_pr)
