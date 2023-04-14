from pathos.pools import ParallelPool as Pool
from itertools import product
import numpy as np
from lgca import get_lgca
from tqdm import tqdm

INPUT = '.\\data\\gog\\nonlocaldensity_10reps\\'
OUTPUT = '.\\data\\gog\\nonlocaldensity_recurrence\\'

def iteration(args):
    from pickle import dump, load
    index, kwargs = args
    with open(kwargs['INPUT'] + 'data{}.pkl'.format(index), 'rb') as f:
        d = load(f)

    theta = kwargs['theta']
    r_d = kwargs['r_d']
    lgca = get_lgca(**constparams, theta=theta, nodes=d['nodes_t'][1:-1], kappa=0., r_d=0.999)
    lgca.props['kappa'] = d['kappa']
    N = lgca.cell_density[lgca.nonborder].sum()
    lgca.timestep()
    lgca.interaction_params['r_d'] = r_d
    tmax = 1
    n_it = lgca.cell_density[lgca.nonborder].sum()
    while n_it < N:
        tmax += 1
        lgca.timestep()
        n_it = lgca.cell_density[lgca.nonborder].sum()
        if n_it == 0:
            break

    data = {'nodes_t': lgca.nodes[lgca.nonborder], 'kappa': lgca.props['kappa'], 'lgca_params': kwargs, 'tmax': tmax,
            'N': N, 'n_it': n_it}
    # save data using pickle
    with open(kwargs['OUTPUT'] + 'data{}.pkl'.format(index), 'wb') as f:
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
    parameters = np.load(INPUT + 'params.npz', allow_pickle=True)
    constparams = parameters['constparams'].item()
    r_ds = parameters['r_ds']
    thetas = parameters['thetas']
    del constparams['nodes']
    reps = 10
    constparams['INPUT'] = INPUT
    constparams['OUTPUT'] = OUTPUT

    params = np.empty((len(r_ds), len(thetas), reps), dtype=object)
    for i, r_d in enumerate(r_ds):
        for j, t in enumerate(thetas):
            for k in range(reps):
                params[i, j, k] = {'constparams': {**constparams}}
                params[i, j, k]['r_d'] = r_d
                params[i, j, k]['theta'] = t
                # params[i, j, k]['kappa'] = kappa[i, j, k]

    # np.savez(PATH+'params.npz', constparams=constparams, r_ds=r_ds, thetas=thetas)
    paramstobeiterated = preprocess(params, 1, **constparams)  # as each repetition needs a different kappa, we need to pretend it is a different parameter, and only have one repetition
    result = multiprocess(iteration, paramstobeiterated, nodes=7)
    # n_pr = np.empty(params.shape + (reps,), dtype=object)
    # n_pr = postprocess(result, n_pr)
    # np.save(PATH+'n_pr.npy', n_pr)
