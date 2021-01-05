import numpy as np
import pandas as pd

from modeling import rc 
from modeling import kernel

rc.surface.band = "Ka"
rc.surface.y = 2500
rc.surface.x = 2500



U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z

direction = np.arange(0, 180, 5)
xmax = (U**2 * rc.surface.nonDimWindFetch) / g 
x = np.arange(xmax, xmax + 180*5e3, 5e3)



label = ["sigmaxx", "sigmayy", "sigmaxy", "sigmayx"]
t = ["model"]
band = ["C", "X", "Ku", "Ka"]

columns = pd.MultiIndex.from_arrays([["Wind Direction"],[""],[""]])
df = pd.DataFrame(direction, columns=columns )
iterables = [t, band, label]
columns = pd.MultiIndex.from_product(iterables) 
df0 = pd.DataFrame(columns=columns, index=df.index)
df = pd.concat([df, df0], axis=1)



cov = np.zeros((direction.size, len(band), 2, 2))
for i in range(direction.size):
    rc.wind.direction = direction[i]
    arr, X0, Y0 = kernel.launch(kernel.cwm)
    arr = np.array([arr[0],
                    arr[0] + arr[1],
                    arr[0] + arr[1] + arr[2],
                    arr[0] + arr[1] + arr[2] + arr[3]])

    for j in range(len(band)):
        print(i,j)
        cov[i][j] = np.cov(arr[j][1], arr[j][2])


for i, b in enumerate(band):
    df.loc[:,("model", b, 'sigmaxx') ] = cov[:, i, 0, 0]
    df.loc[:,("model", b, 'sigmayy') ] = cov[:, i, 1, 1]
    df.loc[:,("model", b, 'sigmaxy') ] = cov[:, i, 0, 1]
    df.loc[:,("model", b, 'sigmayx') ] = cov[:, i, 1, 0]

with open("ModelMoments_WIND.xlsx", "wb") as f:
    df = df.to_excel(f)

