import argparse
import logging
import multiprocessing
import sys
import time

from lgca import get_lgca
from numpy import save, savez
from uuid import uuid4 as uuid

class Simulation(object):
  def __init__(self, dim, rc, steps, save_dir=None):
    self.dim = dim
    self.rc = rc
    self.steps = steps
    self.save_dir = save_dir

  def __call__(self, id):
    logging.info("starting simulation %s", str(id))
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
                    variation=False, density=1, dims=self.dim, restchannels=self.rc, pop={1: 1})
    lgca.timeevo(timesteps=self.steps, recordMut=True)
    logging.info('completed simulation %s', str(id))

    if self.save_dir != None:
      prefix = self.save_dir + str(2 * self.dim + self.dim * self.rc) + str(self.dim) + "_mut_" + str(id)
      logging.info("saving data with prefix '%s'", prefix)
      save(prefix  + '_tree',       lgca.tree_manager.tree)
      save(prefix  + '_families',   lgca.props['lab_m'])
      save(prefix  + '_offsprings', lgca.offsprings)
      savez(prefix + '_Parameter',  density=lgca.density, restchannels=lgca.restchannels, dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)

def get_arg_parser():
  parser = argparse.ArgumentParser(description='LGCA Mutation', formatter_class=argparse.RawDescriptionHelpFormatter,)
  parser.add_argument('-d', '--dimensions',    dest='d', type=int, required=True, help="dimensions of the lgca used in simulations")
  parser.add_argument('-r', '--rest-channels', dest='r', type=int, required=True, help="number of rest channels (per node) in the lgca used in simulations")
  parser.add_argument('-t', '--timesteps',     dest='t', type=int, required=True, help="number of timesteps to simulate")
  parser.add_argument('-n', '--simulations',   dest='n', type=int, default=1, help="number of concurrent simulations")
  parser.add_argument('-o', '--output-dir',    dest='o', nargs='?', help="simulation data will be written to this directory (if set)")
  parser.add_argument('-v', '--verbose',       dest='v', action='store_true', help="enable debug logging")
  return parser

def main(args):
  args = get_arg_parser().parse_args(args)

  # configure logging
  logging.basicConfig(format='%(asctime)s [PID%(process)d] %(name)s - %(levelname)s: %(message)s',
                      level=logging.DEBUG if args.v else logging.INFO)

  logging.info("simulation parameters: dim=%d, rc=%d, steps=%d", args.d, args.r, args.t)
  logging.info("starting %d concurrent simulations", args.n)
  start = time.time()
  with multiprocessing.Pool() as pool:
    pool.map(Simulation(args.d, args.r, args.t, args.o), [uuid() for _ in range(args.n)])

  logging.info('all threads completed in {:5.3f} minutes'.format((time.time() - start)/60))

main(sys.argv[1:])