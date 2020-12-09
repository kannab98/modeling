from modeling import rc, spectrum
from modeling.surface import Surface
import matplotlib.pyplot as plt

surface = Surface()

plt.figure()
# spectrum.plot()
fetch = 3000
rc.surface.nonDimWindFetch = fetch
plt.polar(surface.phi, surface.Phi(spectrum.peak, surface.phi).T)
plt.savefig("kek")


# print("Truly: ", spectrum.k_m, spectrum.limit_k, rc.surface.nonDimWindFetch) 
sigma, sigmaxx, sigmayy = surface._theoryStaticMoments(rc.surface.band)
print(sigma, sigmaxx, sigmayy, sigmaxx+sigmayy)
# print(spectrum.k_m, spectrum.limit_k, rc.surface.nonDimWindFetch)