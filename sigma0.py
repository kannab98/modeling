from surface import Surface
import surface as srf
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse
import numpy.random as rnd


from tools.prepare import *

ymax = 100000
xmax = 100000
x0 = np.linspace(-xmax, xmax, grid_size)
y0 = np.linspace(-ymax, ymax, grid_size)
x0, y0 = np.meshgrid(x0,y0)
x0 = x0.flatten()
y0 = y0.flatten()



xi = np.deg2rad(np.arange(-10, 10, 2))
# xi = np.deg2rad([2])

fig, ax = plt.subplots()

G = np.zeros((grid_size, grid_size))
for i in range(xi.size):
        print(xi[i])


        # const["antenna"]["deviation"][0] = xi[i]

        (  
            const["surface"]["mean"][0], 
            const["surface"]["sigmaxx"][0], 
            const["surface"]["sigmayy"][0]
        ) \
        = (0,0,0)
        # = surface.get_moments(x0,y0,surf) 



        surf = np.zeros((3, x0.size))
        pulse = Pulse(surf, x0, y0, const)

        print(pulse.z0)
        # sigma[j][i] = pulse.cross_section(xi[i])
        G0 = pulse.G0(xi=xi[i], phi=90)
        G += G0.reshape((grid_size, grid_size))
        x = pulse.R[0, :].reshape((grid_size, grid_size))
        y = pulse.R[1, :].reshape((grid_size, grid_size))
        ax.contourf(x,y,  G)

        # y0 += 5000
        # x0 += 5000


    # print(xi[i])

# ax.plot( np.rad2deg(xi), sigma[0] )
# ax.plot( np.rad2deg(xi), sigma[1] )
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