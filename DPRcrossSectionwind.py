import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




rc.wind.speed  = 12
U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z


xim = np.arctan(245e3/z/2)
print(np.rad2deg(xim))
step = np.arctan(5e3/z)
xi = np.arange(0, xim+step, step)
xi = np.unique(np.array([*xi*(-1), *xi]))
xi = srf.angle_correction(xi)

index = np.where(np.abs(np.rad2deg(xi)) <= 12)
xi  = xi[index]


import pandas as pd
with open("direction.tsv", "r") as f:
    df = pd.read_csv(f, sep="\t", header=0)

xmax = (U**2 * rc.surface.nonDimWindFetch) / g 

x = df["direction"]

cov = np.zeros((x.size, 2, 2))
for i in range(df["direction"].size):
    cov[i] = [ [df["sigmaxx"][i], df["sigmaxy"][i]],
               [df["sigmaxy"][i], df["sigmayy"][i]] ]


sigma0 = srf.cross_section(xi, cov)
print(sigma0.shape, xi.shape)
sigma = sigma0 + np.random.normal(0, 0.3, sigma0.shape)
sigma = 10*np.log10(np.abs(sigma))
sigma0 = 10*np.log10(srf.cross_section(xi, cov))


X = df["direction"].values
y = np.linspace(-125, 125, xi.size)
x, y = np.meshgrid(X, xi)

plt.figure(figsize=(10, 5))
# plt.contourf(x, np.rad2deg(y), (sigma), levels=100)
plt.pcolor(x, np.rad2deg(y), (sigma))
plt.xlabel("Азимутальный угол, град")
plt.ylabel("Theta, град")
plt.colorbar()
plt.savefig("slices/direction.png")



# U = np.ones(x.shape) 
# V = np.ones(x.shape) 





xim = np.arctan(245e3/z/2)
step = np.arctan(5e3/z)
xi = np.arange(0, xim+step, step)
xi = np.rad2deg(np.unique(np.array([*xi*(-1), *xi])))
# xi = np.abs(np.round(xi, 1))
xi = np.linspace(-17, 17, 33)
# print(xi.shape, sigma.shape)



import pandas as pd


print(x.shape, sigma0.shape)
for i in range(0, xi.size//2+1, 1):
    plt.figure(figsize=(3, 3))

    if xi[i] != 0:
        plt.plot(x[i, :], sigma[-i-1, :],'.-',  label="$\sigma-$")

    plt.plot(x[i, :], sigma[i, :], '.-', label="$\sigma+$")
    plt.plot(x[i, :], sigma0[i, :],  label="$\sigma_т$")
#     plt.close()

    plt.title("$\\theta = %.2f$" % np.abs(xi[i]))
    plt.xlabel("Направление волнения, град")
    plt.ylabel("$\\sigma_0$, dB")
    plt.legend()
    plt.savefig('slices/wind_slice_%sdeg.png' % xi[i], bbox_inches="tight", transparent=False)
    df = pd.DataFrame({"direction": X, "sigma0+": sigma[i,:], "sigma0-": sigma[-i-1,:], "sigma0_theory": sigma0[i, :]})
    df.to_csv('slices/wind_slice_%sdeg.csv' % xi[i])







