import argparse
import logging
import multiprocessing
import sys
import time
import lgca
import numpy
import numpy.random
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
		l = lgca.get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
										variation=False, density=1, dims=self.dim, restchannels=self.rc, pop={1: 1})
		l.timeevo(timesteps=self.steps, recordMut=True)
		
		logging.info('completed simulation %s', str(uu))
		if self.save_dir != None:
			self.save_data(l, uu)

	def save_data(self, l, uu):
		prefix = self.save_dir + str(2 * self.dim + self.dim * self.rc) + str(self.dim) + "_mut_" + str(uu)
		logging.info("saving data with prefix '%s'", prefix)
		numpy.save(prefix  + '_tree',       l.tree_manager.tree)
		numpy.save(prefix  + '_families',   l.props['lab_m'])
		numpy.save(prefix  + '_offsprings', l.offsprings)
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

	logging.info("simulation parameters: dim=%d, rc=%d, steps=%d", args.d, args.r, args.t)
	logging.info("starting %d concurrent simulations", args.n)
	start = time.time()
	with multiprocessing.Pool() as pool:

		pool.map(Simulation(args.d, args.r, args.t, save_dir=args.o),
						 [(uuid.uuid4(), rand_int()) for _ in range(args.n)])

	logging.info('all threads completed in {:5.3f} minutes'.format((time.time() - start)/60))

main(sys.argv[1:])