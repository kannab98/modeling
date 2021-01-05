import numpy as np
from modeling import rc 
import pandas as pd
import re
import os
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad





U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z


regex = re.compile(r'ku_line.*.dat')
i = 0
xi = np.linspace(-17,17,)
for root, dirs, files in os.walk('./'):
  for file in files:
    if regex.match(file):
        with open(file, "r") as f:
            df = pd.read_csv(f, sep="\s+", header=None)
            cov = np.zeros((2, 2))
            cov[0,0] = 0.0217
            cov[1,1] = 0.0325

            xi_real = df[0].values[np.where(np.abs(df[0]) < 12)]
            sigma_real = df[1].values[np.where(np.abs(df[0]) < 12)]

            func = lambda xi, x, y: 10*np.log10(srf.cross_section(np.deg2rad(xi), np.array([ [x,0], [0,y] ])).T[0])
            from scipy.optimize import curve_fit
            popt = curve_fit(func, 
                                xdata=xi_real,
                                ydata=sigma_real,
                                p0=[1, 1],
                                bounds = [0, 1]
                            )[0]
            A = 0.95
            # A = 1
            popt[0] *= A
            popt[1] *= 1/A

            fig, ax = plt.subplots()
            ax.text(0.5, 0.35, '$\\sigma^2_{xx} = %.3f$ \n $\\sigma^2_{yy} = %.3f$' % tuple(popt),
                    horizontalalignment = 'center',    #  горизонтальное выравнивание
                    fontsize = 12)
            ax.plot(xi, func(xi, *popt), label="теория")
            ax.plot(df[0], df[1], label="данные DPR")
            ax.set_xlabel("$\\theta$, градусы")
            ax.set_ylabel("$\\sigma_0$, dB")

            plt.legend()

            plt.savefig("sigma0_real/sigma0_real%d" % i)
            i+=1