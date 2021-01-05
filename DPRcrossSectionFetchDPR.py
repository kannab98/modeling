import numpy as np
from modeling import rc 
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad




U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z


xim = np.arctan(245e3/z/2)
step = np.arctan(5e3/z)
xi = np.arange(0, xim+step, step)
xi = np.unique(np.array([*xi*(-1), *xi]))
xi = srf.angle_correction(xi)
index = np.where(np.abs(np.rad2deg(xi)) <= 12)
xi  = xi[index]

import pandas as pd
with open("fetch.xlsx", "rb") as f:
    df = pd.read_excel(f, index_col=[0], header=[0,1,2], engine='openpyxl')
# print(df)


# dfcov = df.iloc[3:,:]
# fetch = df["fetch"][3:].values*U**2/g
# for i in range(int(fetch.size)):
#     fetch = np.append(fetch, fetch[-1]+10e3)

fetch = df["Fetch"].values
Ku = df["model","Ku"]
# print(Ku)


cov = np.zeros((Ku.shape[0], 2, 2))
for i in range(Ku.shape[0]):
    cov[i] = [ [Ku["sigmaxx"].iloc[i], Ku["sigmaxy"].iloc[i]],
                [Ku["sigmaxy"].iloc[i], Ku["sigmayy"].iloc[i]] ]

# i = -1
# a = np.array([[ [dfcov["sigmaxx"].iloc[i], dfcov["sigmaxy"].iloc[i]],
#                 [dfcov["sigmaxy"].iloc[i], dfcov["sigmayy"].iloc[i]] ]])

# for i in range(int(fetch.size/2)):
#     cov = np.vstack((cov, a))

# print(fetch.shape, cov.shape)

# # print(cov)

sigma0 = srf.cross_section(xi, cov)
# sigma = sigma0 + np.random.normal(0, 0.2, sigma0.shape)
# sigma0 = 10*np.log10(srf.cross_section(xi, cov))
sigma = 10*np.log10(sigma0)


# plt.plot(xi, sigma[:, 2])
# # plt.show()
x, y = np.meshgrid(fetch.T[0], xi)
# plt.figure(figsize=(10, 5))
plt.pcolor(x/1e3, np.rad2deg(y), (sigma), vmin=sigma.min(), vmax=sigma.max())
# # plt.imshow(x/1e3, np.rad2deg(y), sigma)
# print(sigma.max())
# plt.xlabel("X, км")
# plt.ylabel("$\\theta$, градусы")

# bar = plt.colorbar()
# bar.set_label("$\\sigma0$, dB")
# plt.savefig("slices/fetch")

# xim = np.arctan(245e3/z/2)
# step = np.arctan(5e3/z)
# xi = np.arange(0, xim+step, step)
# xi = np.rad2deg(np.unique(np.array([*xi*(-1), *xi])))
# xi = np.abs(np.round(xi, 1))

# xi = np.abs(np.linspace(-17, 17, 33))

# import pandas as pd
hxi = xi.size//2+1
# print(hxi)
for i in range(0, hxi, 1):
    plt.figure(figsize=(3, 3))
    # plt.plot(x[i, :]/1e3, sigma[i, :], '.-', label="$\sigma+$")
# # 
    if xi[i] != 0:
        plt.plot(x[i, :]/1e3, sigma[-i-1, :],'.-',  label="$\sigma-$")

    # plt.plot(x[i, :]/1e3, sigma0[i, :],  label="$\sigma_т$")
    plt.title('$\\theta = %.1f$' % xi[i])
    plt.xlabel("X, км")
    plt.ylabel("$\\sigma_0$, dB")
    plt.legend()
    # plt.savefig('slices/fetch_slice_%sdeg.png' % xi[i], transparent=False, bbox_inches="tight")

#     df = pd.DataFrame({"x": x[i, :], "sigma0+": sigma[i,:], "sigma0-": sigma[-i-1,:], "sigma0_theory": sigma0[i, :]})
#     df.to_csv('slices/fetch_slice_%sdeg.csv' % xi[i])



