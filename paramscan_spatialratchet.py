import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from tqdm.auto import tqdm

# cancer params

r_b = 0.5
r_d = r_b / 4 * 3
a_max = 1
# p_mut = 1
p_l = 1e-8
T_d = 700
# T_d = 1400
T_p = 5e6
# T_p = 1e7
p_d = T_d * p_l
p_p = T_p * p_l
p_mut = 1 - (1 - p_d) * (1 - p_p)
p_pos = T_d / (T_d + T_p)
s_p = 0.001 * r_b
s_d = 0.2 * r_b
# tmax = 100000
# Da = var / 2
dens0 = 1 - r_d / r_b
# mean_mut_effect = p_pos * s_d - (1 - p_pos) * s_p
# var = p_pos * s_d**2 + (1 - p_pos) * s_p**2
# secondmommu = var + mean_mut_effect**2
Ncrit = T_p * (s_p / r_b) / T_d / (s_d/r_b)**2
mucrit = s_d / T_p / r_b

# L0 = 1
# K = round(np.ceil(Ncrit / 2 / L0))


# t_half = calc_t_half(N0*L0, Ncrit, K*L0, r_d, p_p, s_p/r_b)
# t_half = calc_t_half_naive(r_b, r_d, p_p, s_p)
# t_half_naive = 1 / r_d**(N0/2)
# vcrit = L0 / t_half * 2
# vtarget = vcrit * 4
# vtarget = 1e-1
# gamma = -np.log(vtarget**2 / (4 * (r_b - r_d)))
# gamma = 13
# gamma = np.log(t_half * N0)  # approximate gamma value
# D = 1. / (2 + np.exp(gamma))
# Tfill = l / (2 * np.sqrt(D * (r_b - r_d)))
nK = 5
ngamma = 5
Ks = np.logspace(-1, .5, nK) * Ncrit  # adjust later, very broad range now
Ks = Ks.astype(int)
gammas = np.linspace(0, 18, ngamma)
reps = 10
p_cancer = np.zeros((len(gammas), len(Ks), reps))
print('Ks: ', Ks, 'gammas: ', gammas, 'reps: ', reps, 'Ncrit: ', Ncrit)
# use tqdm to get a progress bar for the three loops

with tqdm(total=len(gammas) * len(Ks) * reps) as pbar:
    for i, gamma in enumerate(gammas):
        for j, K in enumerate(Ks):
            for rep in range(reps):
                l = round(np.ceil(8 * Ncrit / K))
                dims = l,
                N0 = dens0 * K
                nodes = np.zeros((l, 3), dtype=int)
                nodes[0, 0] = round(dens0 * K)
                lgca = get_lgca(ib=True, nodes=nodes, bc='refl', interaction='birthdeath_cancerdfe', ve=False, capacity=K,
                                r_d=r_d, r_b=r_b, a_max=a_max, geometry='lin', dims=dims, restchannels=1, s_d=s_d, p_d=p_d,
                                p_p=p_p, s_p=s_p, gamma=gamma)
                n = nodes.sum()
                while 0 < n < round(3 * N0 / 4):
                    lgca.timestep()
                    n = lgca.cell_density[lgca.nonborder].sum()

                if n > 0:
                    p_cancer[i, j, rep] = 1

                pbar.update(1)


np.save('p_cancer_trial1.npy', p_cancer)
np.savez(Ks=Ks, gammas=gammas, reps=reps, Ncrit=Ncrit, r_b=r_b, r_d=r_d, p_l=p_l, s_d=s_d, p_d=p_d, p_p=p_p, s_p=s_p,
         T_d=T_d, T_p=T_p, p_mut=p_mut, dens0=dens0, )

