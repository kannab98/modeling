import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate

from .rc import Surface, Constants, Swell, Wind


"""
Спектр ветрового волнения и зыби. Используется при построении морской поверхности. 
"""


class Spectrum(Surface, Constants, Swell, Wind):
    def __init__(self):
        self._surface = Surface()
        self._constants = Constants()
        self.swell = Swell()
        self.windWaves = Wind()

        x = self._surface.nonDimWindFetch
        self._nonDimWindWetch = x
        self.band = self._surface.band
        self.g = self._constants.gravityAcceleration

        self.windSpeed = self.windWaves.speed
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

    def peakUpdate(self):
        x = self._surface.nonDimWindFetch
        U = self.windSpeed
        # коэффициент gamma (см. спектр JONSWAP)
        self._gamma = self.Gamma(x)
        # коэффициент alpha (см. спектр JONSWAP)
        self._alpha = self.Alpha(x)
        # координата пика спектра по частоте
        self.omega_m = self.Omega(x) * self.g/U
        # координата пика спектра по волновому числу
        self.k_m = self.omega_m**2/self.g
        # print("begin", self.k_m)

        # print(self.band)
        # длина доминантной волны
        self.lambda_m = 2 * np.pi / self.k_m
        # массив с границами моделируемого спектра.
        self.KT = self.kEdges(self.k_m, self.band)
        # k0 -- густая сетка, нужна для интегрирования и интерполирования

        self.k0 = np.logspace(
            np.log10(self.KT[0]), np.log10(self.KT[-1]), 10**5)

        limit = np.zeros(5)
        limit[0] = 1.2 * self.omega_m
        # limit[1] = (
        #     + 0.371347584096022408
        #     + 0.290241610467870486 * U
        #     + 0.290178032985796564 / U) * self.omega_m
        limit[1] = ( 0.8*np.log(U) + 1 ) * self.omega_m
        limit[2] = 20.0
        limit[3] = 81.0
        limit[4] = 500.0


        self.limit_k = np.array([self.find_decision(limit[i]) for i in range(limit.size)])
        self.limit_k = self.limit_k[np.where(self.limit_k <= self.KT.max())]


    def plot(self):
        fig, ax = plt.subplots(ncols=2)
        for x in [4000,6000, 11000,16000, 20000]:
            self.nonDimWindFetch = x
            S = self.get_spectrum()
            k = self.k0
            ax[0].loglog(k, S(k))

        for U in [5,10,15]:
            self.windSpeed = U
            S = self.get_spectrum()
            k = self.k0
            ax[1].loglog(k, S(k))

        plt.savefig("spectrum")

    @property
    def nonDimWindFetch(self):
        return self._surface.nonDimWindFetch

    @nonDimWindFetch.setter
    def nonDimWindFetch(self, x):
        self._surface.nonDimWindFetch = x
        self.peakUpdate()

    @property
    def windSpeed(self):
        return self.U10

    @windSpeed.setter
    def windSpeed(self, U10):
        self.U10 = U10
        self.peakUpdate()

    def get_spectrum(self):
        # self.peakUpdate()
        # интерполируем смоделированный спектр
        if self.windWaves.enable:
            spectrum = self.interpolate(self.full_spectrum)
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
        P = self.g * 1000.0/0.074
        Q = -1000.0*omega**2/0.074
        x1 = -Q/2.0 + np.sqrt((Q/2)**2 + (P/3)**3)
        x2 = -Q/2.0 - np.sqrt((Q/2)**2 + (P/3)**3)
        k = x1**(1/3)-(-x2)**(1/3)
        return k

    def det(self, k):
        #        Функция возвращает Якобиан при переходе от частоты к
        #    волновым числам по полному дисперсионному уравнению
        det = (self.g + 3*k**2*0.074/1000) / \
            (2*np.sqrt(self.g*k+k**3*0.074/1000))
        return det

    def k_max(self, omega_max):
        # k_max -- координата пика спектра
        k_max = omega_max**2/self.g
        return k_max

    def omega_k(self, k):
        #    Пересчет волнового числа в частоту по полному дисперсионному
        # уравнению
        omega_k = (self.g*k+0.074*k**3/1000)**(1/2)
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


        # print("end", self.k_m)
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
    def Omega(self, x):
        if x >= 20170:
            return 0.835
        omega_tilde = (0.61826357843576103
                       + 3.52883010586243843e-06*x
                       - 0.00197508032233982112*np.sqrt(x)
                       + 62.5540113059129759/np.sqrt(x)
                       - 290.214120684236224/x
                       )
        return omega_tilde

    def spectrum0(self, n, k):
        power = [   
                    4, 
                    5, 
                    7.647*np.power(self.windSpeed, -0.237), 
                    0.0007*np.power(self.windSpeed, 2) - 0.0348*self.windSpeed + 3.2714,
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
    
    def quad(self, p):

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
    def dblquad(self, func):


        # var = integrate.dblquad(
        #                                    lambda phi, k: func(phi, k)*,  
        #                                    a=self.limit_k[1], b=self.limit_k[2],
        #                                    gfun=lambda phi: -np.pi, 
        #                                    hfun=lambda phi:  np.pi)[0]

        # S = lambda k: self.spectrum0(0,k) * k**2

        # var = integrate.dblquad(
        #                                    lambda phi, k: func(phi, k)*S(k),  
        #                                    a=self.KT[0], b=self.limit_k[0],
        #                                    gfun=lambda phi: -np.pi, 
        #                                    hfun=lambda phi:  np.pi)[0]
        # for i in range(1, self.limit_k.size):
        #     S = lambda k: self.spectrum0(i,k) * k**2
        #     var += integrate.dblquad(
        #                                     lambda phi, k: func(phi, k)*S(k),  
        #                                     a=self.limit_k[i-1], b=self.limit_k[i],
        #                                     gfun=lambda phi: -np.pi, 
        #                                     hfun=lambda phi:  np.pi)[0]

        # return var

        # var = integrate.dblquad(
        #                                    func,
        #                                    a=-np.pi, b=np.pi,
        #                                    gfun=lambda k: self.limit_k[1], 
        #                                    hfun=lambda k: self.limit_k[3]
        # )[0]


        var = integrate.quad(func, -np.pi, np.pi, args=(self.KT[0]))[0] * self.quad(2)
        return var


    def full_spectrum(self, k):

        self.peakUpdate()
        if not isinstance(k, np.ndarray):
            k = np.array([k])
        
        ind = np.zeros((self.limit_k.size + 1), dtype=int)
        n = self.limit_k.size

        full_spectrum = np.zeros(k.size, dtype=np.float64)
        for j in range(n):
            tmp =  np.where(k <= self.limit_k[j])[0]
            if tmp.size == 0:
                break
            ind[j+1] = np.max(tmp)
            full_spectrum[ ind[j] : ind[j+1] ] =  self.spectrum0(j, k[ind[j]:ind[j+1]])

        full_spectrum[ind[-1]:] = self.spectrum0(n, k[ind[-1]:])
        return full_spectrum

    def full_spectrum0(self, k):


        for j in range(self.limit_k.size):
            if k <= self.limit_k[j]:
               return self.spectrum0(j, k)

        return self.spectrum0(5, k)



    def interpolate(self, spectrum):
        spectrum = interpolate.interp1d(self.k0, spectrum(self.k0))
        return spectrum

    def swell_spectrum(self, k):

        omega_m = self.Omega(20170) * self.g/self.U10
        W = np.power(omega_m/self.omega_k(k), 5)

        sigma_sqr = 0.0081 * self.g**2 * np.exp(-0.05) / (6 * omega_m**4)
        spectrum = 6 * sigma_sqr * W / \
            self.omega_k(k) * np.exp(-1.2 * W) * self.det(k)
        return spectrum