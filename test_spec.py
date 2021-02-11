
# from modeling.tools import correlate
# from modeling import rc, kernel, cuda, surface, spectrum
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft
# import scipy.fft as fft
# from scipy import signal



# z = np.linspace(-5, 5, 500)
# # fig, ax = plt.subplots(ncols=2)
# # ax[0].plot(z, spectrum.pdf_heights(z, 'default'))
# # ax[0].plot(z, spectrum.pdf_heights(z, 'cwm'))

# # z = np.linspace(-0.3, 0.3, 500)
# # ax[1].plot(z, spectrum.pdf_slopes(z, 'default'))
# # ax[1].plot(z, spectrum.pdf_slopes(z, 'cwm'))
# # sigma = 0.03 

# # f = 1/np.sqrt(2*np.pi*sigma) * np.exp(-z**2/jj)
# # ax[1].plot(z, spectrum.pdf_slopes(z, 'cwm'))




# xmax = 1000 
# step = spectrum.fftstep(xmax)
# k = np.arange(-spectrum.KT[-1], spectrum.KT[-1], step)

# S = spectrum.spectrum_cwm(xmax)
# fig, ax = plt.subplots()
# ax.loglog(k, np.abs(S)) 

# k = np.linspace(spectrum.KT[0], spectrum.KT[-1], 10**5)
# ax.loglog(k, spectrum.ryabkova(k)) 
# # plt.show()


from modeling import rc
from modeling.spectrum import spectrum
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp


# def kedges(band):


#     rc.surface.band = "Ku"

#     spec = self.get_spectrum()


#     k_edge = 2000








#     print(limits[-1])



#         return answ
# print(spectrum.kedges('Ku'))


bands = ['Ka', 'Ku', 'X', 'C']
kedges = [spectrum.kEdges(spectrum.k_m, band)[-1] for band in bands]
wls = [0.0075, 0.016, 0.025, 0.0375]
wls = [0.0075*2**i for i in range(4)]


def curv_criteria():
    # Сейчас попробуем посчитать граничное волновое число фактически из экспериментальных данных
    # Из работы Панфиловой известно, что полная дисперсия наклонов в Ku-диапазоне задается формулой

    S0 = spectrum.get_spectrum()
    # Дисперсия наклонов из статьи
    var = lambda U10: 0.0101 + 0.0034*U10

    # Необходимо найти верхний предел интегрирования, чтобы выполнялось равенство
    # интеграла по спектру и экспериментальной формулы для дисперсии
    S = lambda k: S0(k) * np.power(k, 2)
    # Интеграл по спектру наклонов
    integral = lambda k_bound: sp.integrate.quad(S, spectrum.KT[0], k_bound)[0]

    Func = lambda k_bound: integral(k_bound) - var(rc.wind.speed)
    # Поиск граничного числа Ku диапазона
    # (Ищу ноль функции \integral S(k) k^2 dk = var(U10) )
    opt = sp.optimize.root_scalar(Func, bracket=[spectrum.KT[0], spectrum.KT[-1]]).root

    # Спектр кривизн
    S = lambda k: S0(k) * np.power(k, 4)
    curv = lambda k_bound: sp.integrate.quad(S, S0.x[0], k_bound, )[0]

    # Значение кривизны для Ku диапазона
    curv0 = curv(opt)

    KuWaveLength = 0.022

    # Критерий выбора волнового числа
    eps = np.power(KuWaveLength/(2*np.pi) * np.sqrt(curv0), 1/3)

    return eps

U10 = np.linspace(8, 25, 10)
km = np.zeros_like(U10)
Eps = np.zeros_like(U10)
for i in range(U10.size):
    rc.wind.speed = U10[i]
    spectrum.peakUpdate()
    km[i] = spectrum.k_m
    Eps[i] = curv_criteria()



func = lambda k_m: 2.6376 * k_m**2 - 0.9241*k_m + 0.3437
plt.plot(U10, func(km), label="masha")
# plt.plot(U10, km)
plt.plot(U10, Eps, label='kirill')
# plt.plot(U10, (spectrum.Omega(20170)/U10)**2 * 9.81)
# plt.plot(U10, km)

plt.legend()
plt.savefig("kek")



# def findK(l):


#     func = lambda k_m: 2.6376 * k_m**2 - 0.9241*k_m + 0.3437
#     print(eps, func(spectrum.k_m))

    # Func = lambda k_bound: np.power( l/(2*np.pi) * np.sqrt(curv(k_bound)), 1/3 ) - eps

    # # Func = lambda k_edge: curv(k_edge) - func(spectrum.k_m)


    # root = sp.optimize.root_scalar(Func, bracket=[spectrum.KT[0], spectrum.KT[-1]]).root
    # print(root)
    # return root

# rc.wind.speed = 10
# spectrum.peakUpdate()
# wls = [0.022]
# plt.plot(k,x)
# wls = np.array(wls)
# wls[0] *= 1/2
# wls[1] *= 2
# wls[2] *= 1/2
# wls[3] *= 1
# L = np.linspace(0.0079, 0.0375, 10)
# K = [findK(l) for l in np.array(L)2]
# K0 = [findK(l) for l in np.array(wls)]

# print(wls)
# plt.semilogy(wls, kedges, 'r.')
# plt.semilogy(L, K)
# plt.semilogy(wls, K0, 'bx')
# plt.savefig("kek")
# bounds = sp.optimize.Bounds(lb=S0.x[10], ub=S0.x[-1])

# answ = sp.optimize.minimize(Func, x0=S0.x[1], bounds=bounds)