
from modeling.surface import Surface
from modeling.spectrum import Spectrum
from modeling.retracking import Brown
from modeling import rc

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

brown = Brown()
surf = Surface()
    

df = pd.read_excel("data/ModelMomentsCWM.xlsx", header=[0,1,2], index_col=[0])
df0 = pd.read_excel("data/check.xlsx", header=[0,1,2], index_col=[0])

u = df["U"]
print(u)
df = df["model"]["Ku"]
df0 = df0["ryabkova"]["Ku"]


theta = np.deg2rad(np.linspace(-17, 17, 100))


csu = np.zeros(u.size)
csu0 = np.zeros(u.size)
for i in range(u.size):
    moments = np.zeros(4)
    moments[:-1] = df.iloc[i][:-1]
    csu[i] = 10*np.log10(surf.crossSection(0, moments))
    moments[:-1] = df0.iloc[i][:-1]
    csu0[i] = 10*np.log10(surf.crossSection(0, moments))

dfcs = pd.DataFrame({"U": u.values.flatten(), "default": csu0.flatten(), "cwm": csu.flatten()})
dfcs.to_csv("crosssec_wind.tsv", sep="\t", index=False)


moments = np.zeros(4)
moments[:-1] = df.iloc[7][:-1]
cs = 10*np.log10(surf.crossSection(theta, moments))
moments[:-1] = df0.iloc[7][:-1]
cs0 = 10*np.log10(surf.crossSection(theta, moments))
dfcs = pd.DataFrame({"theta": np.rad2deg(theta), "default": cs0, "cwm": cs})
dfcs.to_csv("crosssec10.tsv", sep="\t", index=False)

t = brown.t()
P0 = brown.pulse(t)
P0 *=  cs0.max()/P0.max()
P1 = brown.pulse(t, cwm=True)
P1 *=  cs.max()/P1.max()

plt.plot(t, P0)
plt.plot(t, P1)
d = pd.DataFrame({"t": t, "linear": P0, "cwm": P1})
d.to_csv("impulse_cwm10.tsv", sep="\t", index=False)



moments[:-1] = df.iloc[0][:-1]
cs = 10*np.log10(surf.crossSection(theta, moments))
moments[:-1] = df0.iloc[0][:-1]
cs0 = 10*np.log10(surf.crossSection(theta, moments))

dfcs = pd.DataFrame({"theta": np.rad2deg(theta), "default": cs0, "cwm": cs})
dfcs.to_csv("crosssec3.tsv", sep="\t", index=False)

t = brown.t()
P0 = brown.pulse(t)
P0 *=  cs0.max()/P0.max()
P1 = brown.pulse(t, cwm=True)
P1 *=  cs.max()/P1.max()

plt.plot(t, P0)
plt.plot(t, P1)
d = pd.DataFrame({"t": t, "linear": P0, "cwm": P1})
d.to_csv("impulse_cwm3.tsv", sep="\t", index=False)



