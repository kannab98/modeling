import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from modeling import rc
from modeling.retracking import Retracking

rc.constants.lightSpeed = 1500
rc.antenna.impulseDuration = 40e-6




retracking = Retracking()
with open("impulses/01022020_12_1.txt") as f:
    df = pd.read_csv(f, sep="\s+", comment="#")


t = df.iloc[:,0].values
P = df.iloc[:,1].values
popt, ice = retracking.pulse(t, P)
print(popt)


fig, ax = plt.subplots()
pulse = ice(t, *popt)
ax.plot(t, P)
ax.plot(t, pulse)
ax.set_xlabel('$t$, с')
ax.set_ylabel('$P$ ')
ax.text(0.05,0.95, '\n'.join((
    '$H_s = %.2f$ м' % (retracking.swh(popt[3])  ),
    '$c\\Delta t  = %.2f$ м' % (retracking.height(popt[2])),
    )),
    verticalalignment='top',transform=ax.transAxes,)

fig.savefig("lel")
