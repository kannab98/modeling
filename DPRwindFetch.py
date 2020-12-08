import numpy as np
from modeling import rc
from modeling.surface import Surface
import modeling.surface as srf
from modeling.surface import kernel_default 
from modeling.experiment import Experiment
import matplotlib.pyplot as plt


kernels = [kernel_default]
labels = ["default", "cwm"] 

surface = Surface()

ex = Experiment(surface)
z = ex.sattelite_coordinates[-1]

U = surface.windSpeed
g = rc.constants.gravityAcceleration



surface.spectrum.nonDimWindFetch = 20170
surface.nonDimWindFetch = 20170
xi = np.arctan(5000/z)
Xi = np.deg2rad(np.linspace(-17, 17, 49))

# rc = surface._rc
# z = rc["antenna"]["z"]
# R = rc["constants"]["earthRadius"]



# plt.plot(np.rad2deg(Xi), np.rad2deg(f(Xi)))
# plt.show()
# sigma0 = np.zeros(Xi.size)

# fig, ax = plt.subplots()
# X = U**2 * 20170 / g 

# for i, xi in enumerate(Xi):
#     arr, X0, Y0 = srf.run_kernels(kernels, surface)
#     sigma0[i] = surface.crossSection(xi)


# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0/sigma0.max())

Xsize = 10
wind = np.linspace(3,15, Xsize)
# Xsize = direction.size
# sigma0 = np.zeros((Xi.size, Xsize))
# fetch = np.linspace(5000, 20170, Xsize)
fetch = [10170]
direction = np.linspace(-np.pi/2, np.pi/2, 180)


# Xb, Yb, Zb = ex.surface_coordinates
# for i, xi in enumerate(Xi):
# X = Xb
# Y = Yb + z*np.tan(xi)
sigmaxx = np.zeros(len(fetch))
sigmayy = np.zeros(len(fetch))
for j in range(len(fetch)):
    # X, Y, Z = ex.surface_coordinates
    surface.nonDimWindFetch= fetch[j]
    # surface.spectrum.peakUpdate(x=fetch[j])
    print("перед входом:", surface.spectrum.limit_k)
    # print(surface.nonDimWindFetch, surface.direction)
    # surface.windSpeed = wind[j]
    # surface.direction[0] = direction[j]
    # ex.surface_coordinates = (X,Y,Z)
    # arr, X0, Y0 = srf.run_kernels(kernel_default, surface)
    # moments = surface._staticMoments(X0,Y0, arr)
    sigmaxx[j], sigmayy[j] = surface._theoryStaticMoments(rc.surface.band)
    # sigma0[i][j] = surface.crossSection(xi, moments)
    # X += 5000




# y = np.array([z*np.tan(xi) for xi in Xi])
# x = np.array([5000*i for i in range(Xsize)])
# x, y = np.meshgrid(x,y)

import pandas as pd
df = pd.DataFrame({'fetch':fetch, 'sigmaxx': sigmaxx, 'sigmayy': sigmayy})
# df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'sigma': sigma0.flatten()})

df.to_csv('fetch.tsv' , sep='\t', float_format='%.6f')

# plt.contourf(sigma0, levels=100)
# # plt.savefig("track_fetch")

# plt.savefig("direction2" )

# X1, Y1 = np.meshgrid(X1, Y1)
# plt.contourf(X1, Y1, sigma0.T, levels=100)

# plt.imshow(sigma0)

# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0.max())

# plt.savefig("sigma0")
    # for i in range(len(kernels)):
    #     surface.plot(X0[i], Y0[i], arr[i][0], label="default%s" % (U))


# plt.figure()
# for i in range(t.size):
    # p[i] = ex.power(t=t[i])

# plt.plot(t,p)
# plt.show()

# rc.pickle()
# pulse = Pulse(rc)

# rc = pulse.rc
# rc.surface.gridSize = 252
# print(rc.surface.gridSize) 


# import matplotlib.pyplot as plt


# G = np.zeros(pulse.gain.shape)
# pulse.polarAngle = 90
# x = pulse.x
# y = pulse.y



# for i in range(8):
#     for xi in np.arange(-10,10,3):
#         r0 = pulse.sattelite_position
#         pulse.sattelite_position = np.array([r0[0]+5000,r0[1], r0[2]])
#         pulse.deviation = xi 
#         G  += pulse.gain

# plt.contourf(x,y,G)
# plt.savefig("kek")
    

    

    
