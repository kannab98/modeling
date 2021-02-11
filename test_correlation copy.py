from modeling.tools import correlate
from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft
import scipy.fft as fft
from scipy import signal

xmax = 100
rho = np.linspace(0, 500, 400)
K1 = spectrum.fftcorrelate(xmax, 0, 0)
x1 = spectrum.fftfreq(xmax)

plt.figure()
# plt.plot(K0)
F = lambda k, phi: spectrum.azimuthal_distribution(k, phi)
S = lambda k: spectrum.ryabkova(k)

def Spectrum2D(k, phi):
    return S(k)[np.newaxis].T * (F(k,phi))/k/np.cos(phi)*np.sin(phi)


step = spectrum.fftstep(xmax)

k = np.arange(-spectrum.KT[-1], spectrum.KT[-1], step)
phi = np.linspace(-np.pi, np.pi, k.size)
kx = k*np.cos(phi)
ky = k*np.sin(phi)
# ky = np.arange(-spectrum.KT[-1], spectrum.KT[-1], step)

# K = fft.ifft2(S2d(k,phi))*(2*np.pi)*spectrum.KT[-1]
S = Spectrum2D(k, phi)
K = fft.ifft2(S)

K = fft.fftshift(K)*kx.max()*ky.max()



# K, Phi = np.meshgrid(k, phi)

# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
fig, ax = plt.subplots()
# # ax.set_rscale("log")
# # mappable = ax.contourf(phi, np.log10(k), np.log10(K), levels=100)
# mappable = ax.contourf(phi, np.log10(k), K.T, levels=100)
x = spectrum.fftfreq(xmax)
# # X, Y = np.meshgrid(x,x)
K = np.abs(K)/np.max(np.abs(K))
# print(K)
mappable = ax.contourf(x[:,0],x[:,0],K)
fig.colorbar(mappable=mappable)
# # print(k)

# fig, ax = plt.subplots()
# # # ax.loglog(k, spectrum.ryabkova(k))

# # # ax.plot(k, S[:, 0])
# # # for i in range(phi.size):
# # #     ax.loglog(k, S[:, i])

# # # ax.plot(x, K[:, 0])

# ind = [K.shape[0]//2]
# for i in ind:
#     K0 = K[:, i]
#     K0 = K0/K0.max()
#     K0 = np.abs(K0)
#     ax.plot(x, K0)

# K1 = K1/K1.max()
# K1 = np.abs(K1)
# ax.plot(x1, K1)

plt.savefig("kek1")