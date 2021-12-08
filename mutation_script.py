import argparse
import logging
import multiprocessing
import sys
import time
import lgca
import numpy
import numpy.random
import os.path as path
import random
import uuid

class Simulation(object):
	def __init__(self, dim, rc, steps, save_dir=None):
		self.dim = dim
		self.rc = rc
		self.steps = steps
		self.save_dir = save_dir

	def __call__(self, id_and_seed):
		uu, seed = id_and_seed
		numpy.random.seed(seed)
		
		logging.info("starting simulation %s with seed %d", str(uu), seed)
		start = time.time()
		l = lgca.get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
										variation=False, density=1, dims=self.dim, restchannels=self.rc, pop={1: 1})
		l.timeevo(timesteps=self.steps, recordMut=True)
		logging.info('completed simulation %s in %s', str(uu), '{:5.3f} minutes'.format((time.time() - start)/60))
		
		if self.save_dir != None:
			self.save_data(l, uu)

	def save_data(self, l, uu):
		prefix = path.join(self.save_dir, str(2 * self.dim + self.dim * self.rc) + str(self.dim) + "_mut_" + str(uu))
		logging.info("saving data from simulation %s", str(uu))
		logging.debug("writing '%s_tree.npy'", prefix)
		numpy.save(prefix  + '_tree',       l.tree_manager.tree)
		logging.debug("writing '%s_families.npy'", prefix)
		numpy.save(prefix  + '_families',   l.props['lab_m'])
		logging.debug("writing '%s_offsprings.npy'", prefix)
		numpy.save(prefix  + '_offsprings', l.offsprings)
		logging.debug("writing '%s_Parameter.npz'", prefix)
		numpy.savez(prefix + '_Parameter',  density=l.density, restchannels=l.restchannels, dimension=l.l, kappa=l.K, rb=l.r_b, rd=l.r_d, rm=l.r_m, m=l.r_int)

def rand_int():
	return random.randint(0, 4294967295)

def get_arg_parser():
	parser = argparse.ArgumentParser(description='LGCA Mutation', formatter_class=argparse.RawDescriptionHelpFormatter,)
	parser.add_argument('-d', '--dimensions',    dest='d', type=int, required=True, help="dimensions of the lgca used in simulations")
	parser.add_argument('-r', '--rest-channels', dest='r', type=int, required=True, help="number of rest channels (per node) in the lgca used in simulations")
	parser.add_argument('-t', '--timesteps',     dest='t', type=int, required=True, help="number of timesteps to simulate")
	parser.add_argument('-n', '--simulations',   dest='n', type=int, default=1, help="number of concurrent simulations")
	parser.add_argument('-o', '--output-dir',    dest='o', nargs='?', help="simulation data will be written to this directory (if set)")
	parser.add_argument('-v', '--verbose',    	 dest='v', action='store_true', help="enable debug logging")
	return parser

def main(args):
	args = get_arg_parser().parse_args(args)

	# configure logging
	logging.basicConfig(format='%(asctime)s [PID%(process)d] %(name)s - %(levelname)s: %(message)s',
											level=logging.DEBUG if args.v else logging.INFO)

	# check output dir
	out_dir = path.abspath(args.o) if not args.o is None else None
	if out_dir is None:
		logging.warn("DATA WON'T BE SAVED! (set option -o/--output-dir to do so)")
	elif not(path.exists(out_dir) and path.isdir(out_dir)):
		logging.error("Output directory doesn't exist: %s", out_dir)
		sys.exit(1)

	# work
	logging.info("simulation parameters: dim=%d, rc=%d, steps=%d", args.d, args.r, args.t)
	logging.info("starting %d concurrent simulations", args.n)
	start = time.time()
	with multiprocessing.Pool() as pool:
		pool.map(Simulation(args.d, args.r, args.t, save_dir=out_dir),
						 [(uuid.uuid4(), rand_int()) for _ in range(args.n)])

	# wrap up
	logging.info('all threads completed in {:5.3f} minutes'.format((time.time() - start)/60))

main(sys.argv[1:])