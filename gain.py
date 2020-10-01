from surface import Surface
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse


from tools.prepare import *

ymax = 100000
xmax = 100000
x0 = np.linspace(-xmax, xmax, grid_size)
y0 = np.linspace(-ymax, ymax, grid_size)
x0, y0 = np.meshgrid(x0,y0)
x0 = x0.flatten()
y0 = y0.flatten()



xi = np.deg2rad(np.arange(-10, 10, 2))
x = np.arange(0, xmax, 15000)




def Gain(Xi, X, const):
    Xi = np.array(Xi)
    X = np.array(X)

    fig, ax = plt.subplots()
    G = np.zeros((grid_size, grid_size))
    for j in range(X.size):
        const["antenna"]["x"][0] = X[j]
        for i in range(Xi.size):
            const["antenna"]["polarAngle"][0] = 90
            const["antenna"]["deviation"][0] = Xi[i]
            surf = np.zeros((3, x0.size))
            pulse = Pulse(surf, x0, y0, const)
            G0 = pulse.G()
            G += G0.reshape((grid_size, grid_size))
            x = pulse.R[0, :].reshape((grid_size, grid_size))
            y = pulse.R[1, :].reshape((grid_size, grid_size))
            ax.contourf(x, y,  G)

    fig.savefig("kek")

Gain(xi, x, const)
# ax.plot( np.rad2deg(xi), sigma[0] )
# ax.plot( np.rad2deg(xi), sigma[1] )
