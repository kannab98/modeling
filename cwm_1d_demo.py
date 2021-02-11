from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rc.surface.x = [-2500, 2500]
rc.surface.y = [-2500, 2500]
rc.surface.gridSize = [256, 256]
rc.surface.phiSize = 128
rc.surface.kSize = 1024

srf0 = kernel.simple_launch(cuda.default)
# srf0 = kernel.convert_to_band(srf, 'Ka')

srf1 = kernel.simple_launch(cuda.cwm)
# srf1 = kernel.convert_to_band(srf, 'Ka')

X, Y = surface.meshgrid
x = X.flatten()
y = Y.flatten()

print(x[x.size//2])

df = pd.DataFrame({"x": x, "y": y, 
                    "elevation": srf0[0].flatten(),
                    "slopes x":  srf0[1].flatten(),
                    "slopes y":  srf0[2].flatten(),
                    "velocity z": srf0[3].flatten(),
                    "velocity x": srf0[4].flatten(),
                    "velocity y": srf0[5].flatten(), })

df1 = pd.DataFrame({"x": x, "y": y, 
                    "elevation": srf1[0].flatten(),
                    "slopes x":  srf1[1].flatten(),
                    "slopes y":  srf1[2].flatten(),
                    "velocity z": srf1[3].flatten(),
                    "velocity x": srf1[4].flatten(),
                    "velocity y": srf1[5].flatten(), })

surface.coordinates = [x, y, df['elevation'] ]
surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
cov = np.cov(df['slopes x'], df['slopes y'])


from modeling import Experiment
sat = Experiment.experiment()




z = df['elevation'].values.reshape(rc.surface.gridSize)
fig, ax = plt.subplots(ncols=2)
ax[1].plot(X[0,:], z[0,:])
Xi = np.linspace(-12, 12, 50)
N = np.zeros_like(Xi, dtype=float)
for i, xi in enumerate(Xi):
    theta0 = sat.localIncidence(xi=xi)
    ind = sat.sort(theta0, xi=xi)
    theta1 = theta0.reshape(rc.surface.gridSize)

    # ax[0].plot(X[0,:], np.rad2deg(theta1[0,:]))
    # ax[1].plot(X.flatten()[ind], z.flatten()[ind], '.')
    N[i] = ind[0].size

ax[0].plot(Xi, N/N.max())
sigma = surface.cross_section(np.deg2rad(Xi), cov)
ax[0].plot(Xi, sigma/sigma.max())




surface.coordinates = [x, y, df1['elevation'] ]
surface.normal = [df1['slopes x'], df1['slopes y'], np.ones(x.size) ]
cov = np.cov(df1['slopes x'], df1['slopes y'])


sat = Experiment.experiment()


z = df1['elevation'].values.reshape(rc.surface.gridSize)
ax[1].plot(X[0,:], z[0,:])
Xi = np.linspace(-12, 12, 50)
N = np.zeros_like(Xi, dtype=float)
for i, xi in enumerate(Xi):
    theta0 = sat.localIncidence(xi=xi)
    ind = sat.sort(theta0, xi=xi)
    theta1 = theta0.reshape(rc.surface.gridSize)

    N[i] = ind[0].size

ax[0].plot(Xi, N/N.max())

fig.savefig("kek")





