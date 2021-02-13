from modeling import rc
from modeling.spectrum import spectrum
from modeling.Spectrum import spectrum2d, azimuthal_distribution
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp




k = np.logspace( np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), 10**3)
phi = np.linspace(-np.pi, np.pi, 256)
spectrum1d = spectrum.ryabkova
azdist = azimuthal_distribution

S = spectrum1d(k)*azdist(k, phi, spectrum.k_m).T
S = S.T
# print(S.shape)
fig, ax = plt.subplots(ncols=1, figsize=(6, 6))

# ax.set_rlim(0)

# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# ax.set_rscale('log')
print(rc.wind.direction)
from matplotlib.colors import LogNorm
ax.set_yscale("log")
img = plt.pcolormesh(np.rad2deg(phi), k, S, norm=LogNorm(), shading='auto')
cbar = plt.colorbar(img)
# cbar.ax.minorticks_off()
# plt.pcolormesh(S)
plt.savefig("kek")
# fig, ax = plt.subplots(ncols=1, figsize=(6, 6))

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.plot(k, S[:,0])



# print(S2d.x)
