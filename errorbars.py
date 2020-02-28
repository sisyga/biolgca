# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

thom01 = np.load('saved_data/thoms/' + 'rc=178_thom1500.npy')
plot_lognorm_distribution(thom01, int_length=1000)
exit(123456)

mini = np.array([1,3,2,3,2,3,4])

thombsp = np.array([1,3,5,7,\
        11,11,13,15,16,12,18,17,\
        22,21,25,27,21,22,\
        35,33])

    # 0 = mini  1 = bsp  2 = thom01
which = 2

if which == 0:
    thom = mini
    int_length = 1
elif which == 1:
    thom = thombsp
    int_length = 10
elif which == 2:
    thom = thom01
    int_length = 1000

max = thom.max().astype(int)
print('max', max)
print('mean', thom.mean())
print('std', thom.std())
fig, ax = plt.subplots()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('thom', fontsize=15)
plt.ylabel('absolute frequency', fontsize=15)
fitted_data, maxy, y = calc_lognormaldistri(thom=thom, int_length=int_length)

maxfit = fitted_data.max()
x = np.arange(0, max, int_length) + int_length/2

#     sqd = 0
#     for i in range(0, len(x+int_length/2)):
#         sqd += (y[i] - pdf_fitted[i]*maxy/maxfit)**2
#     sqd = math.sqrt(sqd/len(x+int_length/2))
#     error = np.array([sqd]*len(x+int_length/2))
plt.xlim(0, x.max() + int_length/2)

# plt.hist(thom, bins=np.arange(0, thom.max() + 2*int_length, int_length))
plt.bar(x, y, width=int_length, color='grey', alpha=0.5)

xe = []
ye = []
for index, entry in enumerate(y):
    if entry != 0:
        xe.append(x[index])
        ye.append(y[index])
print(len(x), len(xe), len(ye))
plt.errorbar(xe, ye, yerr=0.75, linestyle='')
# plt.plot(x + int_length / 2, fitted_data * maxy / maxfit, color=c, label=id)
plt.legend()
#     error = [1] * len(x+int_length/2)
#     plt.errorbar(x+int_length/2, pdf_fitted*maxy/maxfit, yerr=error)

plt.show()
