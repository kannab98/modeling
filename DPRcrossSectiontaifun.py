import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z





import pandas as pd
with open("taifun.tsv", "r") as f:
    df = pd.read_csv(f, sep="\t", header=0)

x = df["x"].values
y = df["y"].values

xi = np.arctan(y/z)

sigmaxx = df["sigmaxx"].values
sigmayy = df["sigmayy"].values
sigmaxy = df["sigmaxy"].values
cov = np.zeros((x.size, 2, 2))
for i in range(x.size):
    cov[i] = [ [sigmaxx[i], sigmaxy[i]],
                [sigmaxy[i], sigmayy[i]] ]


theta = xi
# Коэффициент Френеля
F = 0.8

if len(cov.shape) <= 2:
    cov = np.array([cov])

K = np.zeros(cov.shape[0])
for i in range(K.size):
    K[i] = np.linalg.det(cov[i])

sigma =  F**2/( 2*np.cos(theta)**4 * np.sqrt(K) )
sigma *= np.exp( - np.tan(theta)**2 * cov[:, 1, 1]/(2*K))
sigma += np.random.normal(0, .5, sigma.shape)
sigma = np.abs(sigma)

# # plt.plot(xi, sigma[:, 2])

# # plt.show()
print(x.size, y.size)
x = x.reshape((49,25))
y = y.reshape((49,25))
sigma = sigma.reshape((49,25))
sigmaxx = cov[:,1,1].reshape((49,25))
plt.contourf(x, y, sigmaxx)
# plt.xlabel("X, км")
# plt.ylabel("Theta, градусы")

bar = plt.colorbar()
bar.set_label("sigma0, dB")
plt.savefig("taifun")

# # plt.plot(U**2 * df["fetch"]/g, df["sigmaxx"])
# # plt.plot(U**2 * df["fetch"]/g, df["sigmayy"])
# # plt.plot(U**2 * df["fetch"]/g, df["sigmayy"] + df["sigmaxx"])
# # plt.show()

