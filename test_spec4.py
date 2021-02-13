from modeling import rc
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
from modeling import spectrum
import math



# class spectrum:
#     def __call__(self, k):
#         print("return kek")

#     def __init__(self):
#         self._k = 0
    
# print(spectrum.__sizeof__())
# print(spectrum(10))


# print(y)

S = spectrum.JONSWAP(spectrum.k)

# S = spectrum.JONSWAP(spectrum.k)

plt.loglog(spectrum.k, S)

# limit = np.array([spectrum.KT[0], *spectrum.limit_k, spectrum.KT[-1], np.inf])
# S = np.zeros(spectrum.k.size)
# S0 = spectrum.piecewise_spectrum(0, spectrum.k, where= (limit[0] < spectrum.k) & ( spectrum.k <= limit[1]))
# spectrum.piecewise_spectrum(0, spectrum.k, where= (limit[0] < spectrum.k) & ( spectrum.k <= limit[1]))
# plt.loglog(spectrum.k, S0)
# # S1 = spectrum.piecewise_spectrum(1, spectrum.k, where= (limit[1] < spectrum.k) & ( spectrum.k <= limit[2]))
# # plt.loglog(spectrum.k, S1)

# # S2 = spectrum.piecewise_spectrum(2, spectrum.k, where= (limit[2] < spectrum.k) & ( spectrum.k <= limit[3]))
# # plt.loglog(spectrum.k, S2)

# k = spectrum.k[np.where(spectrum.k <= limit[1])]
# F = spectrum(k)
# plt.loglog(k, F)

k = spectrum.k
# S = np.zeros_like(k, dtype=float)
# spectrum.piecewise_spectrum(0,)
# S = np.frompyfunc(spectrum.JONSWAP_np, 1, 1)
S = spectrum(spectrum.k)
plt.loglog(k, S)



plt.savefig("kek")
plt.show()

