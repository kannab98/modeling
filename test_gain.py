from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modeling import Experiment


rc.surface.x = [-100000, 100000]
rc.surface.y = [-100000, 100000]
rc.surface.gridSize = [512, 512]

X, Y = surface.meshgrid
x = X.flatten()
y = Y.flatten()

surface.coordinates = [x , y, rc.antenna.z*np.ones_like(x) ]
surface.normal = [ np.zeros_like(x), np.zeros_like(x), np.ones_like(x)]

sat = Experiment.experiment()

xi = np.deg2rad(np.linspace(-17,17, 1))
N = np.zeros(xi.size)

fig, ax = plt.subplots()
# local = sat.localIncidence.reshape(X.shape)

for i in range(N.size):
    N[i]= sat.crossSection(xi[i])



ax.plot(np.rad2deg(xi),N/N.max())
fig.savefig("kek")