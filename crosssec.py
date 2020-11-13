from modeling.surface import Surface
from modeling.surface import kernel_default as kernel
from modeling.surface import run_kernels

from modeling.spectrum import Spectrum
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import matplotlib

surface = Surface()
with open("data/Moments.xlsx", "rb") as f:
    df = pd.read_excel(f, header=[0,1,2], index_col=[0])
    U = df["U"].values
    df = df["ryabkova"]

with open("data/ModelMomentsCWM.xlsx", "rb") as f:
    df1 = pd.read_excel(f, header=[0,1,2], index_col=[0])
    df1 = df1["model"]

with open("data/ModelMoments.xlsx", "rb") as f:
    df = pd.read_excel(f, header=[0,1,2], index_col=[0])
    df = df["model"]



moments = np.zeros(4)
moments[1:3] = df["Ku"].iloc[5,1:3].values

theta = np.deg2rad(np.linspace(-17, 17, 49))
theta = surface.angleCorrection(theta)

cs = surface.crossSection(theta, moments)
plt.plot(theta, 10*np.log10(cs))

moments = np.zeros(4)
moments[1:3] = df1["Ku"].iloc[5,1:3].values
cs = surface.crossSection(theta, moments)
plt.plot(theta, 10*np.log10(cs))

# plt.plot(U, df["Ku"].iloc[:,0] - df1["Ku"].iloc[:,0])
# plt.show()

# F = lambda phi, k: surface.Phi(k, phi)
# a1 = surface.spectrum.partdblquad(F, 0, 2)
# a2 = surface.spectrum.partdblquad(F, 2, 0)
# a3 = surface.spectrum.partdblquad(F, 1, 1)

# sigma = a3**2 - a2*a1
# var0 = surface.spectrum.quad(2)
# var  = var0 - 2*sigma


# print(var0, var)