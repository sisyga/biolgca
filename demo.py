from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

"""GETTING STARTED"""
#envoke package and obtain appropriate class instance
# lattice dimensions: dims=(x,y) or dims=x
# interaction rule: interaction= ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'random_walk',
#   'excitable_medium', 'nematic', 'persistant_motion', 'chemotaxis', 'contact_guidance'], default: random walk
# boundary conditions: bc= ['refl' |
# geometry: geometry= ['hex' | 'lin']

#lgca = get_lgca(geometry='hex')

"""INTERACTIONS"""
# print all available interactions
#lgca.get_interactions()

# change interaction rule
#lgca.set_interaction(interaction='alignment', beta=3.0)
#alignment interaction needs beta parameter; default: 2

#nodes = np.array([[0,0],[2,0],[0,0],[4,0],[1,0],[0,0],[0,0],[0,0],[0,0],[30,0],[14,0],[0,0]])
nodes = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[1,0],[3,0],[0,0]])

"""INITIAL STATE"""
#lgca2 = get_lgca(density=0.1, dims=10, geometry='lin', ve=True, bc='refl', interaction='go_and_grow')
lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', nodes=nodes, beta=2.0)
#default: homogeneous with constant mean density: density=
#TODO: set random seed to reproduce results?
#restchannels=
#dims=5 gives 7 cells? -> interaction range IR is given by user, minimum 1 -> IR cells at boundaries to impose
#   boundary conditions
#nodes=np.ndarray initial configuration manually

#lgca2.print_nodes()
#1D: rest channels in the middle of the print?

"""TIMESTEPPING"""
#lgca2.timestep()
#lgca2.print_nodes()
#simulate with step recording: class method timeevo
# timesteps=
# record=True: record all configurations
# recorddens=True: record the density profile; default
# recordN=True: record total number of cells
lgca2.print_nodes()
lgca2.timeevo(timesteps=4, record=True)
#print("Print nodes")
lgca2.print_nodes()
#print("plot density")
ani = lgca2.plot_density()
ani2 = lgca2.plot_flux()
plt.show()

#done: how to print 1D time evolution as 2D simple plot: LGCA_1D.plot_density

# numbers_list = [2, 5, 62, 5, 42, 52, 48, 5]
# print(numbers_array[2:5]) # 3rd to 5th
# print(numbers_array[:-5]) # beginning to 4th
# print(numbers_array[5:])  # 6th to end
# print(numbers_array[:])   # beginning to end
# 1:5:2 from 1 to 5 in steps of 2 7 stride 2
# ::3 first and then every 3rd object