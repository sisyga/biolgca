import argparse
import logging
import multiprocessing
import sys
import time
import lgca
from lgca.helpers2d import *
from lgca.ib_interactions import driver_mut, passenger_mut
import numpy as np
import numpy.random
import random
import uuid


class Simulation(object):
    def __init__(self, dim, rc, steps, driver, trange, save_dir=None):
        self.dim = dim
        self.rc = rc
        self.steps = steps
        self.save_dir = save_dir
        self.driver = driver
        self.trange = trange

    def __call__(self, id_and_seed):
        uu, seed = id_and_seed
        numpy.random.seed(seed)

        logging.info("starting simulation %s with seed %d", str(uu), seed)

        nodes = np.zeros((self.dim, self.dim, 6 + self.rc))
        for i in range(0, 6 + self.rc):
            nodes[self.dim // 2, self.dim // 2, i] = i + 1

        effect = driver_mut if self.driver else passenger_mut
        lgca_hex = lgca.get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                                 r_b=0.09, r_d=0.08, r_m=0.001, effect=effect)

        lgca_hex.timeevo(timesteps=self.steps, recordoffs=True, callback=self.prep_during_timeevo(uu))

        logging.info('completed simulation %s', str(uu))
        if self.save_dir != None:
            self.save_data(lgca_hex, uu)

    def prep_during_timeevo(self, uu):
        def during_timeevo(lgca, step):
            if step % self.trange == 0:
                print(step, uu)
                if sum(lgca.offsprings[-1][1:]) > 0:
                    lgca.plot_families(save=True, id=str(uu) + '_step=' + str(step) + '_famplot')
                    lgca.plot_density(cbar=False, save=True, id=str(uu) + '_step=' + str(step) + '_densplot')
                else:
                    print('\n--------- ausgestorben -----------\n')
        return during_timeevo


    def save_data(self, l, uu):
        eff = '_driver' if self.driver else '_passenger'
        prefix = self.save_dir + str(uu) + '_' + str(self.dim) + 'x' + str(self.dim)\
                 + 'rc=' + str(self.rc) + eff
        logging.info("saving data with prefix '%s'", prefix)
        numpy.save(prefix + '_tree', l.tree_manager.tree)
        numpy.save(prefix + '_families', l.props['lab_m'])
        numpy.save(prefix + '_offsprings', l.offsprings)


def rand_int():
    return random.randint(0, 4294967295)


def get_arg_parser():
    parser = argparse.ArgumentParser(description='LGCA Mutation',
                                     formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument('-d', '--dimensions', dest='d', type=int, required=True,
                        help="dimensions of the lgca used in simulations")
    parser.add_argument('-m', '--mutationeffect', dest='m', action='store_true',
                        help="True -> driver mut, False -> passenger")
    parser.add_argument('-r', '--rest-channels', dest='r', type=int, required=True,
                        help="number of rest channels (per node) in the lgca used in simulations")
    parser.add_argument('-t', '--timesteps', dest='t', type=int, required=True, help="number of timesteps to simulate")
    parser.add_argument('-R', '--timerange', dest='tr', type=int, required=True, help="timesteps between plots")
    parser.add_argument('-n', '--simulations', dest='n', type=int, default=1, help="number of concurrent simulations")
    parser.add_argument('-o', '--output-dir', dest='o', nargs='?',
                        help="simulation data will be written to this directory (if set)")
    parser.add_argument('-v', '--verbose', dest='v', action='store_true', help="enable debug logging")
    return parser


def main(args):
    args = get_arg_parser().parse_args(args)

    # configure logging
    logging.basicConfig(format='%(asctime)s [PID%(process)d] %(name)s - %(levelname)s: %(message)s',
                        level=logging.DEBUG if args.v else logging.INFO)

    logging.info("simulation parameters: dim=%d, rc=%d, steps=%d, driver=%r, trange=%d",
                 args.d, args.r, args.t, args.m, args.tr)
    logging.info("starting %d concurrent simulations", args.n)
    start = time.time()

    if args.n < 2:
        Simulation(args.d, args.r, args.t, args.m, args.tr, save_dir=args.o).__call__((str(uuid.uuid4())[:7], rand_int()))
    else:
        with multiprocessing.Pool() as pool:
            pool.map(Simulation(args.d, args.r, args.t, args.m, args.tr, save_dir=args.o),
                     [(str(uuid.uuid4())[:7], rand_int()) for _ in range(args.n)])

    logging.info('all threads completed in {:5.3f} minutes'.format((time.time() - start) / 60))


main(sys.argv[1:])
