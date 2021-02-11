
from modeling.tools import correlate
from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft
import scipy.fft as fft
from scipy import signal

xmax = 100
K1 = spectrum.fftcorrelate(xmax)
Kcwm = spectrum.fftcorrelate(xmax, 1, 0, 3/2)
x1 = spectrum.fftfreq(xmax)

fig, ax= plt.subplots()

ax.plot(x1, K1)

ax.plot(x1, np.real(K1-20*Kcwm))

# ax.plot(x1, Kcwm)
plt.savefig("kek1")