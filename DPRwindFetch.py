import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z

# xmin = (U**2 * 730) / g 
# xmax = (U**2 * 20170) / g 

# x = np.arange(xmin, xmax+1e4, 5e3)

# fetch = x*g/U**2
# fetch[np.where(fetch> 20170)] = 20170

# cov = np.zeros((fetch.size, 2, 2))
# for j in range(fetch.size):
#     rc.surface.nonDimWindFetch = fetch[j]
#     cov[j] = spec.cov()


import pandas as pd
# df = pd.DataFrame({'fetch':fetch, 'sigmaxx': cov[:,0,0], 'sigmayy': cov[:,1,1], 'sigmaxy': cov[:,1,0]})
# df.to_csv('fetch.tsv' , sep='\t', float_format='%.6f')

with open("fetch.tsv", "r") as f:
    df = pd.read_csv(f, sep="\t", header=0)

plt.plot(U**2 * df["fetch"]/g, df["sigmaxx"])
plt.plot(U**2 * df["fetch"]/g, df["sigmayy"])
plt.plot(U**2 * df["fetch"]/g, df["sigmayy"] + df["sigmaxx"])
plt.show()

