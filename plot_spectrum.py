import numpy as np
from modeling import rc 
import pandas as pd
import re
import os
from modeling import surface as srf
from modeling import spectrum as spec

import matplotlib.pyplot as plt
from scipy.integrate import quad


plt.figure(figsize=(4,4))
rc.surface.band = "Ku"
rc.wind.speed = 3
spec.plot()
rc.wind.speed = 5
spec.plot()
rc.wind.speed = 10
spec.plot()
rc.wind.speed = 15
spec.plot()
rc.wind.speed = 20
spec.plot()
plt.ylabel('$S(\\kappa), м^3 \\cdot рад^{-1}$')
plt.xlabel('$\\kappa, рад \\cdot м^{-1}$')
plt.legend(("$U_{10} = 3$", '$U_{10} = 5$', '$U_{10} = 10$', '$U_{10} = 15$', '$U_{10} = 20$'))
plt.savefig("Ku")


plt.figure(figsize=(4,4))
rc.surface.band = "Ka"
rc.wind.speed = 3
spec.plot()
rc.wind.speed = 5
spec.plot()
rc.wind.speed = 10
spec.plot()
rc.wind.speed = 15
spec.plot()
rc.wind.speed = 20
spec.plot()
plt.ylabel('$S(\\kappa), м^3 \\cdot рад^{-1}$')
plt.xlabel('$\\kappa, рад \\cdot м^{-1}$')
plt.legend(("$U_{10} = 3$", '$U_{10} = 5$', '$U_{10} = 10$', '$U_{10} = 15$', '$U_{10} = 20$'))
plt.savefig("Ka")