


from modeling import rc
from modeling.spectrum import spectrum
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp


rc.wind.speed = 10
S0 = spectrum.get_spectrum()

def curv_criteria(band='Ku'):
    # Сейчас попробуем посчитать граничное волновое число фактически из экспериментальных данных
    # Из работы Панфиловой известно, что полная дисперсия наклонов в Ku-диапазоне задается формулой

    # S0 = spectrum.get_spectrum()
    # Дисперсия наклонов из статьи
    if band == "Ku":
        # var = lambda U10: 0.0101 + 0.0022*np.sqrt(U10)
        var = lambda U10: 0.0101 + 0.0022*U10
        radarWaveLength = 0.022

    elif band == "Ka":
        var = lambda U10: 0.0101 + 0.0034*U10
        radarWaveLength = 0.008

    # Необходимо найти верхний предел интегрирования, чтобы выполнялось равенство
    # интеграла по спектру и экспериментальной формулы для дисперсии
    S = lambda k: S0(k) * np.power(k, 2)
    # Интеграл по спектру наклонов
    integral = lambda k_bound: sp.integrate.quad(S, spectrum.KT[0], k_bound)[0]

    Func = lambda k_bound: integral(k_bound) - var(rc.wind.speed)
    # Поиск граничного числа 
    # (Ищу ноль функции \integral S(k) k^2 dk = var(U10) )
    opt = sp.optimize.root_scalar(Func, bracket=[spectrum.KT[0], spectrum.KT[-1]]).root

    # Значение кривизны 
    curv0 = curvature(opt)

    # Критерий выбора волнового числа
    # eps = np.power(KuWaveLength/(2*np.pi) * np.sqrt(curv0), 1/3)
    eps = np.power(radarWaveLength/(2*np.pi) * np.sqrt(curv0), 1/3)

    return eps

def find(l, band='Ku'):

    func = lambda k_m: 2.6376 * k_m**2 - 0.9241*k_m + 0.3437
    eps = curv_criteria(band)
    Func = lambda k_bound: np.power( l/(2*np.pi) * np.sqrt(curvature(k_bound)), 1/3 ) - eps
    root = sp.optimize.root_scalar(Func, bracket=[spectrum.KT[0], spectrum.KT[-1]]).root
    # print(root)
    return root

def curvature(k_bound):
    # Спектр кривизн
    S = lambda k: S0(k) * np.power(k, 4)
    return sp.integrate.quad(S, S0.x[0], k_bound)[0]

print(curv_criteria('Ku'), curv_criteria('Ka'))



# band = 'C'

if band == "Ku":
    var = lambda U10: 0.0101 + 0.0022*U10
    radarWaveLength = 0.022
    root = find(radarWaveLength, 'Ku')

elif band == "Ka":
    var = lambda U10: 0.0101 + 0.0034*U10
    radarWaveLength = 0.008
    root = find(radarWaveLength, 'Ku')

elif band == 'C':
    var = lambda U10: 0.0101 + 0.0034*U10
    radarWaveLength = 0.05
    root = find(radarWaveLength, 'Ku')



# print(root)
# slopes = lambda k: k**2*S0(k)
# print(sp.integrate.quad(slopes, spectrum.KT[0], root)[0], var(rc.wind.speed))


