from modeling.tools import correlate
from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft
from scipy import signal

# rc.surface.x = [0, *np.pi/spectrum.KT[0]]
factor1 = 8
factor2 = 1

rc.surface.x = [0, factor1 * np.pi/spectrum.KT[0]]
rc.surface.y = [0, 100]

step = np.pi/spectrum.KT[-1]/factor2
count = int(np.ceil(np.max(rc.surface.x)/step))

rc.surface.gridSize = [count, 1]
rc.surface.phiSize = 1
rc.surface.kSize = 1024

srf = kernel.simple_launch(cuda.default)

ind = int(np.ceil(srf[0].size/2))
K = np.zeros(ind)

for i in range(0, ind):
    Z = fft(srf[0][i:i+ind].flatten())
    sigma = spectrum.quad(0)
    K += np.real(ifft(np.abs(Z)**2))/Z.size
    # S += fft(K)


K *= 1/ind

x = surface.gridx.flatten()
# if K.size % 2 == 0:
    # S = S[:ind] + np.flip(S[ind:])
    # K = K[:ind]
    # k = np.linspace(spectrum.KT[0], spectrum.KT[-1], ind)
    # x = x[:ind]
# else:
    # K = K[:ind-1]
    # S = S[:ind-1] + np.flip(S[ind:])
    # k = np.linspace(spectrum.KT[0], spectrum.KT[-1], ind-1)
    # x = x[:ind-1]
# print(k[0], x[0])
# x0 = np.linspace(x.min(), x.max(), 400) 
# K0 = spectrum.correlate(x0)
# K1 = spectrum.fftcorrelate()
# x1 = np.linspace(x.min(), x.max(), K1.size) 
# print(K1.size)
# print(K1.max())

plt.figure()
plt.plot(K)
# plt.plot(x0, K0)
plt.savefig("kek1")
# plt.figure()

# plt.loglog(k, np.abs(S))
# plt.savefig("kek2")


# X = np.real(rfft(srf[0])).flatten()
# x = irfft(X)
# K = irfft(np.conj(X)*X)

# plt.figure()
# plt.plot(srf[0])
# plt.plot(x)

# plt.figure()
# plt.plot(K)