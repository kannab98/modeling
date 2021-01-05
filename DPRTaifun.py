import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




rc.wind.speed = 10
U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z

R = 245e3/4 # Радиус тайфуна
Rx = 0 # Координаты тайфуна
Ry = 0

direction = np.arange(0, np.pi,np.deg2rad(1))
xmax = (U**2 * rc.surface.nonDimWindFetch) / g 

xim = np.arctan(245e3/z/2)
step = np.arctan(5e3/z)
xi = np.arange(0, xim+step, step)
xi = np.unique(np.array([*xi*(-1), *xi]))

y = z * np.tan(xi)
x = np.arange(-R, R, 5e3)
# y = np.array([Rx, Rx+5e3])
# x = np.array([Ry, Ry+5e3])


x, y = np.meshgrid(x, y)
z = np.zeros((x.shape))
z[np.where( np.sqrt((x-Rx)**2 + (y-Ry)**2) <=R )] = 1
plt.contourf(x,y,z)
plt.savefig("taifun")
# print(y.size,x.size)



# cov = np.zeros((x.size, y.size, 2, 2))
# dU = 7

# rc.wind.direction = 0
# rc.wind.speed =  10
# cov0 = spec.cov()
# for i in range(x.size):
#     for j in range(y.size):
#         print(i+j)
#         r = np.sqrt((x[i]-Rx)**2 + (y[j]-Ry)**2)
#         if  r <= R:
#             rc.wind.direction = np.rad2deg(np.arctan(y[j]/x[i]))
#             rc.wind.speed =  r/R * dU + 3
#             print(rc.wind.direction, rc.wind.speed)
#             cov[i][j] = spec.cov()
#         else:
#             cov[i][j] = cov0


# import pandas as pd
# x, y  = np.meshgrid(x,y)
# df = pd.DataFrame({'x':x.flatten(), 'y': y.flatten(),  'sigmaxx':cov[:,:,0,0].flatten(), 'sigmayy': cov[:,:,1,1].flatten(), 'sigmaxy': cov[:,:,1,0].flatten()})
# df.to_csv('taifun.tsv' , sep='\t', float_format='%.6f')

# # with open("direction.tsv", "r") as f:
#     # df = pd.read_csv(f, sep="\t", header=0)

