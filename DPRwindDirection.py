import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z

direction = np.arange(0, 180, 5)
# xmin = (U**2 * 730) / g 
xmax = (U**2 * rc.surface.nonDimWindFetch) / g 

x = np.arange(xmax, xmax + 180*5e3, 5e3)
# fetch = x*g/U**2
# fetch[np.where(fetch> 20170)] = 20170

cov = np.zeros((direction.size, 2, 2))
for j in range(direction.size):
    rc.wind.direction = direction[j]
    cov[j] = spec.cov()


import pandas as pd
df = pd.DataFrame({'direction':direction, 'sigmaxx': cov[:,0,0], 'sigmayy': cov[:,1,1], 'sigmaxy': cov[:,1,0]})
df.to_csv('direction.tsv' , sep='\t', float_format='%.6f')

# with open("direction.tsv", "r") as f:
#     df = pd.read_csv(f, sep="\t", header=0)

# plt.plot(U**2 * df["fetch"]/g, df["sigmaxx"])
# plt.plot(U**2 * df["fetch"]/g, df["sigmayy"])
# plt.plot(U**2 * df["fetch"]/g, df["sigmayy"] + df["sigmaxx"])
# plt.show()

