# from kernel import *
from surface import Surface
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse

from json import load, dump
with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

grid_size = const["surface"]["gridSize"][0] 
c = const["constants"]["lightSpeed"][0]
z0 = const["antenna.z"][0]





surface = Surface(const)
host_constants = surface.export()
print(host_constants[-1].shape)
print(host_constants[-1])
# stream = cuda.stream()
# k, phi, A, F, psi = (cuda.to_device(host_constants[i], stream = stream) for i in range(len(host_constants)))


# Hs = 4 * np.sqrt(surface.sigma_sqr)
# xmax = 4 * np.sqrt(Hs * z0)
# print(xmax)

# x0 = np.linspace(-xmax, xmax, grid_size)
# y0 = np.linspace(-xmax, xmax, grid_size)
# x0, y0 = np.meshgrid(x0,y0)
# x0 = x0.flatten()
# y0 = y0.flatten()


# # threadsperblock = TPB 
# blockspergrid = math.ceil(x0.size / threadsperblock)
# # kernels = [kernel_default, kernel_cwm]
# labels = ["default", "cwm"] 
# fig, ax = plt.subplots()


# data_p = pd.Series(dtype='object')
# data_s = {}
# data_m = pd.Series(dtype='object')

# mn = []
# disp = []

# T0 = (z0 - 5)/c

# for j, kernel in enumerate(kernels):


#     surf = np.zeros((2, 3, x0.size))
#     stream = cuda.stream()
#     kernel[blockspergrid, threadsperblock, stream](surf, x0, y0, k, phi, A, F, psi)


#     T = np.linspace(T0-Hs*1.2/c, np.sqrt(z0**2+xmax**2)/c, 104)
#     P = np.zeros(T.size)

#     pulse = Pulse(surf, x0, y0, const)

#     index = pulse.mirror_sort()
#     for i in range(T.size):
#         P[i] = pulse.power1(T[i])


#     dT = pd.Series(T - T0, name = 't_%s' % (labels[j]))
#     dP = pd.Series(P/P.max(), name = 'P_%s' % (labels[j]))
#     data_p = pd.concat([data_p, dT, dP], axis=1)

#     ax.plot(T-T0, P/P.max(), label = labels[j])





# now = datetime.datetime.now().strftime("%m%d_%H%M")
# os.mkdir(str(now))
# data_p = pd.DataFrame(data_p)




# ax.set_xlabel('t')
# ax.set_ylabel('P')
# plt.legend()
# plt.savefig('%s/impulse.png' % (now))

# data_p.to_csv('%s/impulse%d_%s.csv' % (now, const["wind.speed"][0], now), index = False, sep=';')



# with open('%s/rc.json' % (now), 'w', encoding="utf-8") as f:
#     dump(const, f, indent=4)


# plt.show()
