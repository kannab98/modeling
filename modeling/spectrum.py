import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import interpolate, integrate, optimize
from json import load,  dump

from . import rc




"""
Спектр ветрового волнения и зыби. Используется при построении морской поверхности. 
"""

def Omega(x):
    if x >= 20170:
        return 0.835

    omega_tilde = (0.61826357843576103
                    + 3.52883010586243843e-06*x
                    - 0.00197508032233982112*np.sqrt(x)
                    + 62.5540113059129759/np.sqrt(x)
                    - 290.214120684236224/x
                    )
    return omega_tilde

g = rc.constants.gravityAcceleration

class Spectrum():
    def __init__(self):
        self.peakUpdate()

    @staticmethod
    def kEdges(k_m, band):

        """
        Границы различных электромагнитных диапазонов согласно спецификации IEEE
        
        Band        Freq, GHz            WaveLength, cm
        Ka          26-40                0.75 - 1.13 
        Ku          12-18                1.6  - 2.5 
        X           8-12                 2.5  - 3.75
        C           4-8                  3.75 - 7.5

        """
        bands = {"C":1, "X":2, "Ku":3, "Ka":4}

        bands_edges = [

            lambda k_m: k_m/4,

            lambda k_m: (
                2.74 - 2.26*k_m + 15.498*np.sqrt(k_m) + 1.7/np.sqrt(k_m) -
                0.00099*np.log(k_m)/k_m**2
            ),


            lambda k_m: (
                25.82 + 25.43*k_m - 16.43*k_m*np.log(k_m) + 1.983/np.sqrt(k_m)
                + 0.0996/k_m**1.5
            ),


            lambda k_m: (
                + 68.126886 + 72.806451 * k_m  
                + 12.93215 * np.power(k_m, 2) * np.log(k_m) 
                - 0.39611989*np.log(k_m)/k_m 
                - 0.42195393/k_m
            ),

            lambda k_m: (
                #   24833.0 * np.power(k_m, 2) - 2624.9*k_m + 570.9
                2000
            )

        ]


        edges = np.array([ bands_edges[i](k_m) for i in range(bands[band]+1)])
        return edges

    def peakUpdate(self, x=rc.surface.nonDimWindFetch, U=rc.wind.speed):
        self.U10 = U
        # коэффициент gamma (см. спектр JONSWAP)
        self._gamma = self.Gamma(x)
        # коэффициент alpha (см. спектр JONSWAP)
        self._alpha = self.Alpha(x)

        # координата пика спектра по волновому числу
        self.k_m = (Omega(x) / U)**2  * g 
        self.peak = self.k_m
        # координата пика спектра по частоте
        self.omega_m = Omega(x) * g / U
        # длина доминантной волны
        self.lambda_m = 2 * np.pi / self.k_m

        # массив с границами моделируемого спектра.
        self.KT = self.kEdges(self.k_m, rc.surface.band)

        # k0 -- густая сетка, нужна для интегрирования и интерполирования
        self.k0 = np.logspace(
            np.log10(self.KT[0]), np.log10(self.KT[-1]), 10**5)

        limit = np.zeros(5)
        limit[0] = 1.2 * self.omega_m
        limit[1] = ( 0.8*np.log(U) + 1 ) * self.omega_m
        limit[2] = 20.0
        limit[3] = 81.0
        limit[4] = 500.0


        __limit_k = np.array([self.find_decision(limit[i]) for i in range(limit.size)])
        self.limit_k = __limit_k[np.where(__limit_k <= self.KT.max())]
        del __limit_k, limit


    def plot(self, stype="ryabkova"):
        S = self.get_spectrum(stype)
        k = self.k0
        plt.loglog(k, S(k))

        # plt.savefig("spectrum")

    @property
    def nonDimWindFetch(self):
        return rc.surface.nonDimWindFetch

    @nonDimWindFetch.setter
    def nonDimWindFetch(self, x):
        rc.surface.nonDimWindFetch = x
        self.peakUpdate(x=x)

    @property
    def windSpeed(self):
        return self.U10

    @windSpeed.setter
    def windSpeed(self, U10):
        self.U10 = U10
        rc.wind.speed = U10
        self.peakUpdate(U=U10)

    def get_spectrum(self, stype="ryabkova"):
        # self.peakUpdate()
        # интерполируем смоделированный спектр
        if stype == "ryabkova":
            spectrum = self.interpolate(self.ryabkova)
        elif stype == 'slick':
            f = lambda k: self.ryabkova(k)*self.with_slick(k)
            spectrum = self.interpolate(f)
            # spectrum = self.full_spectrum
            # spec = self.interpolate(self.full_spectrum)
            # spectrum = lambda k: spec(k)*0.0081/2
            # self.sigma_sqr = np.trapz(spectrum(self.k0), self.k0)
            # print("SWH=%.2f" % (4*np.sqrt(self.sigma_sqr)))
            # print("plot wind surface")

        # if self.swell.enable:
        #     spectrum = self.interpolate(self.swell_spectrum)
        #     self.sigma_sqr = np.trapz(spectrum(self.k0), self.k0)
            # print("plot swell surface")

        return spectrum

    def find_decision(self, omega):

        """
        Поиск корней полинома третьей степени. 
        Возвращает сумму двух комплексно сопряженных корней

        """
        p = [74e-6, 0, g, omega**2]
        k = np.roots(p)
        return 2*np.real(k[0])

    def det(self, k):
        #        Функция возвращает Якобиан при переходе от частоты к
        #    волновым числам по полному дисперсионному уравнению
        det = (g + 3*k**2*0.074/1000) / \
            (2*np.sqrt(g*k+k**3*0.074/1000))
        return det

    def k_max(self, omega_max):
        # k_max -- координата пика спектра
        k_max = omega_max**2/g
        return k_max

    def omega_k(self, k):
        #    Пересчет волнового числа в частоту по полному дисперсионному
        # уравнению
        omega_k = (g*k+0.074*k**3/1000)**(1/2)
        return omega_k


    def JONSWAP(self, k):
        self.peakUpdate()
        if not isinstance(k, np.ndarray):
            k = np.array([k])

        sigma = 0.09 * np.ones(k.size, dtype=np.float64)
        sigma[ np.where(k < self.k_m) ] = 0.07
        Sw = ( self._alpha/2 *
            k**(-3)* np.exp( -1.25 * (self.k_m/k)**2 ) *
                np.power(self._gamma,
                    np.exp(- (np.sqrt(k/self.k_m)-1)**2 / (2*sigma**2))
                )
        )


        # Sw =  k**(-3)* np.exp( -1.25 * (self.k_m/k)**2 ) 
        return  Sw 

    # Безразмерный коэффициент Gamma
    def Gamma(self, x):
        if x >= 20170:
            return 1.0
        gamma = (
            +5.253660929
            + 0.000107622*x
            - 0.03778776*np.sqrt(x)
            - 162.9834653/np.sqrt(x)
            + 253251.456472*x**(-3/2)
        )
        return gamma

    # Безразмерный коэффициент Alpha
    def Alpha(self, x):
        if x >= 20170:
            return 0.0081
        else:
            alpha = np.array([], dtype='float64')
            alpha = [(
                +0.0311937
                - 0.00232774 * np.log(x)
                - 8367.8678786/x**2
                + 4.5114599e617*np.exp(-x)
            )]
        return alpha[0]

    # Вычисление безразмерной частоты Omega по безразмерному разгону x


    def spectrum0(self, n, k):
        power = [   
                    4, 
                    5, 
                    7.647*np.power(rc.wind.speed, -0.237), 
                    0.0007*np.power(rc.wind.speed, 2) - 0.0348*rc.wind.speed + 3.2714,
                    5
                ]

        if n == 0:
            return self.JONSWAP(k)

        else:
            omega0 = self.omega_k(self.limit_k[n-1])
            beta0 = self.spectrum0(n-1, self.limit_k[n-1]) * \
                omega0**power[n-1]/self.det(self.limit_k[n-1])
            omega0 = self.omega_k(k)
            return beta0/omega0**power[n-1]*self.det(k)
    
    def quad(self, p, stype="ryabkova"):

        func = lambda k, i: self.spectrum0(i,k) * k**p

        a = self.KT[0]
        b = self.KT[-1]

        var = integrate.quad(func, a, self.limit_k[0], args=(0))[0]

        n = self.limit_k.size

        for i in range(1, n):
            var += integrate.quad(func, self.limit_k[i-1], self.limit_k[i], args=(i))[0]

        var += integrate.quad(func, self.limit_k[-1], b, args=(n))[0]

        return var

    def quad_check(self, func):

        k = self.k0
        var = np.zeros_like(k)
        for i in range(k.size):
            print(i)
            var[i] = integrate.quad(func, -np.pi, np.pi, args=(k[i]))[0]
        return var
    def dblquad(self, func, stype="ryabkova"):

        print("начальные:", self.limit_k)
        S = lambda k: self.spectrum0(0,k) * k**2

        var = integrate.dblquad(
                                           lambda phi, k: func(phi, k)*S(k),  
                                           a=self.KT[0], b=self.limit_k[0],
                                           gfun=lambda phi: -np.pi, 
                                           hfun=lambda phi:  np.pi)[0]
        print("##############################")
        print(self.KT[0], self.limit_k[0], var)

        for i in range(1, self.limit_k.size):
            S = lambda k: self.spectrum0(i,k) * k**2
            var += integrate.dblquad(
                                            lambda phi, k: func(phi, k)*S(k),  
                                            a=self.limit_k[i-1], b=self.limit_k[i],
                                            gfun=lambda phi: -np.pi, 
                                            hfun=lambda phi:  np.pi)[0]
            print(self.limit_k[i-1], self.limit_k[i], var)



        S = lambda k: self.spectrum0(self.limit_k.size, k) * k**2
        var += integrate.dblquad(
                                            lambda phi, k: func(phi, k)*S(k),  
                                            a=self.limit_k[-1], b=self.KT[-1],
                                            gfun=lambda phi: -np.pi, 
                                            hfun=lambda phi:  np.pi)[0]
        print(self.limit_k[-1], self.KT[-1], var)
        print("##############################")

        return var

    def partdblquad(self, func, a, b, c):

        S = lambda k: self.spectrum0(0, k)
        Phi = lambda phi, k: func(phi, k) * np.cos(phi)**a * np.sin(phi)**b

        var = integrate.dblquad(
                                           lambda phi, k: Phi(phi, k) * S(k) * k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b,
                                           a=self.KT[0], b=self.limit_k[0],
                                           gfun=lambda phi: -np.pi, 
                                           hfun=lambda phi:  np.pi)[0]

        for i in range(1, self.limit_k.size):
            S = lambda k: self.spectrum0(i,k)
            var += integrate.dblquad(
                                            lambda phi, k: Phi(phi, k) * S(k) * k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b,
                                            a=self.limit_k[i-1], b=self.limit_k[i],
                                            gfun=lambda phi: -np.pi, 
                                            hfun=lambda phi:  np.pi)[0]


        S = lambda k: self.spectrum0(self.limit_k.size, k)
        var += integrate.dblquad(
                                            lambda phi, k: Phi(phi, k) * S(k) * k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b,
                                            a=self.limit_k[-1], b=self.KT[-1],
                                            gfun=lambda phi: -np.pi, 
                                            hfun=lambda phi:  np.pi)[0]

        return var



        # var = integrate.quad(func, -np.pi, np.pi, args=(self.KT[0]))[0] * self.quad(2, stype)
        # return var


    def ryabkova(self, k):

        self.peakUpdate()
        if not isinstance(k, np.ndarray):
            k = np.array([k])
        
        ind = np.zeros((self.limit_k.size + 1), dtype=int)
        n = self.limit_k.size

        ryabkova = np.zeros(k.size, dtype=np.float64)
        for j in range(n):
            tmp =  np.where(k <= self.limit_k[j])[0]
            if tmp.size == 0:
                break
            ind[j+1] = np.max(tmp)
            ryabkova[ ind[j] : ind[j+1] ] =  self.spectrum0(j, k[ind[j]:ind[j+1]])

        ryabkova[ind[-1]:] = self.spectrum0(n, k[ind[-1]:])

        return ryabkova
    
    @staticmethod
    def __wind__(x, z):
        f = lambda x, z: x/0.4 * np.log(z/(0.684/x + 428e-7* x**2 - 443e-4))
        return f(x,z)



    def  find_friction(self):

        # Finds zeroes, X - friction velocity, Z - height, Karman's constant = 0.4
        z = 10
        f = lambda x: x/0.4 * np.log(z/(0.684/x + 428e-7* x**2 - 443e-4))
        root = optimize.ridder( f, 1e-8, 80)
        return root

        
    def kontrast(self, k, beta0, sigma_w, sigma_m, e):

        g = 981
        R = 1
        nu = 0.01

        def gamma(k, e, omega):

            Rw = R * np.power(omega, 2)

            x1 = 2*nu*k**2/omega 
            x2 =  e * np.power(k, 3) * np.sqrt(x1) / Rw
            x3 = 1 * x2 * e * np.power(k, 3) / ( 2 * x1 * Rw )
            x4 = 2 * x2
            x5 = 2 * x2 * e * np.power(k, 3) / ( 1 * np.sqrt(x1) * Rw )

            return 2*nu*k**2*(x1 - x2 + x3)/(x1 - x4+ x5)
        
        omega_w = np.sqrt(g*k + sigma_w/R * np.power(k, 3))
        omega_m = np.sqrt(g*k + sigma_m/R * np.power(k, 3))



        G0 = gamma(k, 0, omega_w)
        G1 = gamma(k, e, omega_m)


        uftr = self.find_friction()
        beta = beta0*np.power(k*uftr, 2)/omega_w

        # kontrast = 0.1 * np.ones(k.size)
        # slick_kontrast = 0

        # flag = True
        # for i in range(k.size):
        #     if (beta[i] > 2*G0[i]) & (beta[i] > 2*G1[i]):
        #         kontrast[i] = (beta[i] - 2*G1[i])/(beta[i] - 2*G0[i])

        #         print(kontrast[i])
        #     elif (beta[i] < 2*G0[i]) & (beta[i] < 2*G1[i]):
        #         kontrast[i] = (2*G0[i] - beta[i]) / (2*G1[i] - beta[i])
        #         print(kontrast[i])
        #     else:
        #         flag = False
            
        #     if flag:
        #         slick_kontrast = kontrast[i]
        #     else:
        #         kontrast[i] *= slick_kontrast
        #         print(kontrast[i])



        # ind = np.where( (beta > 2*G0) & (beta > 2*G1) )[0]
        # if len(ind) != 0:
        #     kontrast[ind] = \
        #         (beta - 2*G1[ind])/(beta - 2*G0[ind])

        # ind = np.where( (beta < 2*G0) & (beta < 2*G1) )[0]
        # print(ind)
        # if len(ind) != 0:
        #     kontrast[ind] = \
        #         (beta - 2*G0[ind])/(beta - 2*G1[ind])


        # return kontrast
        return np.abs( (beta - np.minimum(G0, G1) ) / (beta - np.maximum(G0, G1)) )
        
    
    def with_slick(self, k):
        beta0 = rc.slick.beta0
        sigma_w = rc.slick.sigmaWater
        sigma_m = rc.slick.sigmaOil
        sigma = rc.slick.sigma

        return self.kontrast(k/100, beta0, sigma_w, sigma_m, sigma)


    def full_spectrum0(self, k):


        for j in range(self.limit_k.size):
            if k <= self.limit_k[j]:
               return self.spectrum0(j, k)

        return self.spectrum0(5, k)



    def interpolate(self, spectrum):
        spectrum = interpolate.interp1d(self.k0, spectrum(self.k0))
        return spectrum

    def swell_spectrum(self, k):

        omega_m = self.Omega(20170) * g/self.U10
        W = np.power(omega_m/self.omega_k(k), 5)

        sigma_sqr = 0.0081 * g**2 * np.exp(-0.05) / (6 * omega_m**4)
        spectrum = 6 * sigma_sqr * W / \
            self.omega_k(k) * np.exp(-1.2 * W) * self.det(k)
        return spectrum