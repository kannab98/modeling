from modeling.tools import correlate
from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal

rc.surface.x = [0, 1000]
rc.surface.y = [0, 100]
rc.surface.gridSize = [1024*1024, 1]
rc.surface.phiSize = 128
rc.surface.kSize = 2048



srf = kernel.simple_launch(cuda.default)

X, Y = surface.meshgrid
x = X.flatten()

z = srf[0].flatten()

index = np.where(x <= 10*spectrum.lambda_m)[0][-1]
x0 = x[:index]
z0 = z[:index]

K = np.zeros(z.size - index)

for i in range(z.size - index):
    K[i] = (np.trapz(z0*z[i:index+i], x0) )/(x0[-1] - x0[0])

# # F = scipy.fft.fft(K)
plt.plot(x[:-index], K)

plt.figure()
F = scipy.fft.rfft(K)
w = scipy.fft.rfftfreq(z.size-index, (x[1] - x[0])/2/np.pi)


F = scipy.fft.fftshift(F)
w = scipy.fft.fftshift(w)
S = spectrum.get_spectrum()
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.loglog(w, np.abs(F))
k0 = np.logspace( np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), 10**3)
ax.loglog(k0, S(k0)*k0**0)
# # print(spectrum.lambda_m)
# # var = np.trapz(z*z, X.flatten())/np.max(X)
# # print(var)

# # K = signal.convolve(z, z, mode='full')
# # print(K.max()/var)



# # var = 
# # K = signal.convolve(srf[0].flatten(), srf[0].flatten())/X.max()
# # print(var)
# # print(K.max())
# # print(K.max()/var)


def correlate(x, y, p=0):
    K = signal.convolve(y, y)/4/(np.pi)**2

    Y = scipy.fft.ifft(K)
    X = scipy.fft.fftfreq(K.size, (x[1] - x[0])/2/np.pi) 


    X = scipy.fft.fftshift(X)
    print(X)
    Y = scipy.fft.fftshift(Y)
    Y = np.abs(Y)



    fig, ax = plt.subplots(nrows=1, ncols=2)

    rho = np.arange(-y.size + 1, y.size) * (x[1] - x[0])
    ax[0].plot(rho, K)

    S = spectrum.get_spectrum()
    ax[1].loglog(X, Y)
    k0 = np.logspace( np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), 10**3)
    ax[1].plot(k0, S(k0)*k0**p)


# plt.show()


# correlate(X.flatten(), srf[0].flatten())
plt.savefig("kek")
# # correlate(X.flatten(), (srf[1]+srf[2]).flatten(), p=2)