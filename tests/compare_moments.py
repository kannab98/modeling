import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from modeling.surface import Surface
from modeling.surface import kernel_default as kernel
from modeling.surface import run_kernels
# from modeling.experiment import Experiment

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

kernels = kernel


surface = Surface()
# U = np.linspace(3, 20, 18)
U = np.array([15,20])

band = ["C", "X", "Ku", "Ka"]
# band = ["C"]
label = ["var", "xx", "yy", "xx+yy",  "xy"]
tp = ["theory", "model"]

sigma1 = np.zeros((U.size, len(band), len(label)))
sigma2 = np.zeros((U.size, len(band), len(label)))

columns = pd.MultiIndex.from_arrays([["U"],[""],[""]])
df = pd.DataFrame(U, columns=columns )

iterables = [tp, band, label]
columns = pd.MultiIndex.from_product(iterables) 
df0 = pd.DataFrame(columns=columns, index=df.index)

df = pd.concat([df, df0], axis=1)
for j in range(U.size):
    print(j)
    surface.spectrum.windSpeed =  U[j]
    surface.windSpeed =  U[j]

    arr, X0, Y0 = run_kernels(kernels, surface)
    arr = np.array([arr[0],
                    arr[0]+arr[1],
                    arr[0]+arr[1]+arr[2],
                    arr[0]+arr[1]+arr[2]+arr[3]])

    for i in range(len(band)):
        moments = surface._theoryStaticMoments(band=band[i])
        sigma1[j,i] = moments
        moments = surface._staticMoments(X0[0], Y0[0], arr[i], )
        sigma2[j,i] = moments


for i, b in enumerate(band):
    for j,m in enumerate(label):
        df.loc[:,("theory",b,m) ] = sigma1[:,i,j]
        df.loc[:, ("model",b,m)] = sigma2[:,i,j]

print(df)

plt.contourf(arr[0][0].reshape((128,128)))
plt.colorbar()
plt.savefig("kek.png")
# print(df*0.0081/2)
# print(df["theory"]["Ka"]["xx"])
df.to_excel("check.xlsx")

    
    # for i in range(4):




