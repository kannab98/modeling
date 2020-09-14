from kernel import *
from surface import Surface
import surface as srf
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse
import numpy.random as rnd


from json import load, dump
with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

grid_size = const["surface"]["gridSize"][0] 
c = const["constants"]["lightSpeed"][0]
z0 = const["antenna"]["z"][0]

surface = Surface(const)
host_constants = surface.export()

xmax = 5000
# xmax = const["surface"]["x"][0]
ymax = 5000
x0 = np.linspace(0, xmax, grid_size)
y0 = np.linspace(0, ymax, grid_size)
x0, y0 = np.meshgrid(x0,y0)
x0 = x0.flatten()
y0 = y0.flatten()



xi = np.deg2rad(np.linspace(-17, 17))


F = 0.5

kernels = [srf.kernel_default, srf.kernel_cwm]
sigma = np.zeros((len(kernels),xi.size))


fig, ax = plt.subplots()


for i in range(xi.size):

    Surf, X, Y = srf.run_kernels(kernels, x0, y0, host_constants)
    for j in range(len(kernels)):
        surf = Surf[j]
        x0 = X[j]
        y0 = Y[j]


        const["antenna"]["deviation"][0] = xi[i]



        (  
            const["surface"]["mean"][0], 
            const["surface"]["sigmaxx"][0], 
            const["surface"]["sigmayy"][0]
        ) \
        = surface.get_moments(x0,y0,surf) 

        sigmaxx = const["surface"]["sigmaxx"][0]
        sigmayy = const["surface"]["sigmayy"][0]

        pulse = Pulse(surf, x0, y0, const)
        sigma[j][i] = pulse.cross_section(xi[i])

        y0 += 5000

    print(xi[i])

ax.plot( np.rad2deg(xi), sigma[0] )
ax.plot( np.rad2deg(xi), sigma[1] )
plt.savefig("kek")
    # index = pulse.mirror_sort()
    # for i in range(T.size):
        # P[i] = pulse.power1(T[i])


    # dT = pd.Series(T - T0, name = 't_%s' % (labels[j]))
    # dP = pd.Series(P/P.max(), name = 'P_%s' % (labels[j]))
    # data_p = pd.concat([data_p, dT, dP], axis=1)

    # ax.plot(T-T0, P/P.max(), label = labels[j])





# now = datetime.datetime.now().strftime("%m%d_%H%M")
# os.mkdir(str(now))
# data_p = pd.DataFrame(data_p)




# ax.set_xlabel('t')
# ax.set_ylabel('P')
# plt.legend()
# plt.savefig('%s/impulse.png' % (now))

# data_p.to_csv('%s/impulse.csv' % (now), index = False, sep=';')


# with open('%s/rc.json' % (now), 'w', encoding="utf-8") as f:
    # dump(const, f, indent=4)


# plt.show()
