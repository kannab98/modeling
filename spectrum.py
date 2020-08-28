import numpy as np
from scipy import interpolate, integrate

class Spectrum:
    def __init__(self, const):
        self.U10 = const["wind.speed"][0]
        x = const["spectrum.nonDimWindFetch"][0]
        self.band = const["band"][0]
        KT = const["spectrum.edge"][0]
        self.g = const["gravityAcceleration"][0]
        self.swell = const["spectrum.swell"][0]
        self.wind_waves = const["spectrum.windWaves"][0]


        # коэффициент gamma (см. спектр JONSWAP)
        self.__gamma = self.Gamma(x)
        # коэффициент alpha (см. спектр JONSWAP)
        self.__alpha = self.Alpha(x)
        # координата пика спектра по частоте
        self.omega_m = self.Omega(x) * self.g/self.U10
        # координата пика спектра по волновому числу
        self.k_m = self.k_max( self.omega_m )
        # длина доминантной волны
        self.lambda_m = 2 * np.pi / self.k_m

        self.sigma_sqr = self.__alpha * self.__gamma * self.g**2 *np.exp(-0.05)/(6*self.omega_m**4)

        self.k_edge = {} 

        self.k_edge['Ku'] = (
            68.13 + 72.9*self.k_m + 12.9*self.k_m**2*np.log(self.k_m) - 
            -0.396*np.log(self.k_m)/self.k_m - 0.42/self.k_m
            )
        self.k_edge['C'] = (
            2.74 - 2.26*self.k_m + 15.498*np.sqrt(self.k_m) + 1.7/np.sqrt(self.k_m) -
            0.00099*np.log(self.k_m)/self.k_m**2
            )
        self.k_edge['X'] = ( 25.82 + 25.43*self.k_m - 16.43*self.k_m*np.log(self.k_m) + 1.983/np.sqrt(self.k_m)                
            + 0.0996/self.k_m**1.5
            )

        # print(self.k_edge)
        
        # массив с границами моделируемого спектра.
        self.KT = np.array([self.k_m/4, self.k_edge[self.band]])
        if KT != None:
            self.KT = np.array(KT)
        # k0 -- густая сетка, нужна для интегрирования и интерполирования
        self.k0= np.logspace(np.log10(self.KT[0]), np.log10(self.KT[-1]), 10**4)

    def get_spectrum(self):
        # интерполируем смоделированный спектр
        self.spectrum = self.interpolate(self.full_spectrum)
        return self.spectrum

    def find_decision(self,omega):
        P = 9.8 * 1000.0/0.074
        Q = -1000.0*omega**2/0.074
        x1= -Q/2.0 + np.sqrt( (Q/2)**2 + (P/3)**3)
        x2= -Q/2.0 - np.sqrt( (Q/2)**2 + (P/3)**3)
        k=x1**(1/3)-(-x2)**(1/3)
        return k

    def det(self,k):
    #        Функция возвращает Якобиан при переходе от частоты к
    #    волновым числам по полному дисперсионному уравнению
        det=(self.g + 3*k**2*0.074/1000 )/(2*np.sqrt(self.g*k+k**3*0.074/1000) )
        return det

    def k_max(self,omega_max):
        # k_max -- координата пика спектра
        k_max=omega_max**2/self.g
        return k_max

    def omega_k(self,k):
        #    Пересчет волнового числа в частоту по полному дисперсионному
        # уравнению
        omega_k=(self.g*k+0.074*k**3/1000)**(1/2)
        return omega_k

    def JONSWAP(self,k):
        if k<=self.k_m:
            sigma=0.074
        else:
            sigma=0.09
        Sw=(
            self.__alpha/2*k**(-3)*np.exp(-1.25*(self.k_m/k)**2 )*
            self.__gamma**(np.exp(- ( np.sqrt(k/self.k_m)-1)**2 / (2*sigma**2) ))
           )
        return Sw

    # Безразмерный коэффициент Gamma
    def Gamma(self,x):
        if x>=20170:
            return 1
        gamma = (
               +5.253660929
               +0.000107622*x
               -0.03778776*np.sqrt(x)
               -162.9834653/np.sqrt(x)
               +253251.456472*x**(-3/2)
                )
        return gamma

    # Безразмерный коэффициент Alpha
    def Alpha(self,x):
        if x >= 20170:
            return 0.0081
        alpha = np.array( [],dtype = 'float64')
        alpha = [(
               +0.0311937
               -0.00232774 * np.log(x)
               -8367.8678786/x**2
               +4.5114599e+300*np.exp(-x)*1e+300*1e+17
    #            +4.5114599e+17*exp(-x)
              )]
        return alpha[0]

    #Вычисление безразмерной частоты Omega по безразмерному разгону x
    def Omega(self,x):
        if x>=20170:
            return 0.835
        omega_tilde=(0.61826357843576103
                     + 3.52883010586243843e-06*x
                     - 0.00197508032233982112*np.sqrt(x)
                     + 62.5540113059129759/np.sqrt(x)
                     - 290.214120684236224/x
        )
        return omega_tilde

    def spectrum0(self,n,k,spectrum_type = 'Karaev'):
        if spectrum_type == 'Karaev':
            power = [0,4,5,2.7,5]
        if n==0:
            return self.JONSWAP(k)
        else:
            omega0 = self.omega_k(self.limit_k[n-1])
            beta0  = self.spectrum0(n-1,self.limit_k[n-1]) * \
                        omega0**power[n]/self.det(self.limit_k[n-1])
            omega0 = self.omega_k(k)
            return beta0/omega0**power[n]*self.det(k)


    def full_spectrum(self,k,x=20170):

        try:
            full_spectrum = np.zeros(k.size)
        except:
            full_spectrum = [0]
            k = [k]

        if self.wind_waves:
            limit_1 = 1.2
            self.limit_2 = (
                    + 0.371347584096022408
                    + 0.290241610467870486 * self.U10
                    + 0.290178032985796564 / self.U10
                        )
            self.limit_3 = self.omega_k(270.0)
            self.limit_4 = self.omega_k(1020.0)
            self.limit_k = np.zeros(4)
            self.limit_k[0] = self.find_decision(limit_1 * self.omega_m)
            self.limit_k[1] = self.find_decision(self.limit_2 * self.omega_m)
            self.limit_k[2] = 270.0
            self.limit_k[3] = 1020.0


            for i in range(k.size):
                if k[i] <= self.limit_k[0]:
                    full_spectrum[i] =  self.spectrum0(0,k[i])
                elif k[i] <= self.limit_k[1]:
                    full_spectrum[i] = self.spectrum0(1,k[i])
                elif k[i] <= self.limit_k[2]:
                    full_spectrum[i] = self.spectrum0(2,k[i])
                elif k[i] <= self.limit_k[3]:
                    full_spectrum[i] = self.spectrum0(3,k[i])
                else:
                    full_spectrum[i] = self.spectrum0(4,k[i])



        if self.swell:
            for i in range(k.size):
                full_spectrum[i] += self.swell_spectrum(k[i]) 
        return full_spectrum


    def interpolate(self, spectrum):
        spectrum = interpolate.interp1d(self.k0, spectrum(self.k0))
        return spectrum
    
    def swell_spectrum(self, k):
        W = np.power(self.omega_m/self.omega_k(k), 5)
        sigma_sqr = 0.0081 * self.g**2 * np.exp(-0.05) / (6 * self.omega_m**4)
        spectrum = 6 * sigma_sqr * W / self.omega_k(k) * np.exp(-1.2 * W)   * self.det(k)
        return spectrum


if  __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import argparse

    ap = argparse.ArgumentParser(
        description="This script may plot Karaev (2002) spectrum for Ku, C and X band at different wind speeds.",
    )
    ap.add_argument("-b", "--band", nargs='+', type=str, required=False, 
                    default=['Ku', 'C'], 
                    help="(default: Ku C)")
    ap.add_argument("-w", "--windspeed", nargs='+', type=float, required=False, 
                    default=[5,7,10,15], 
                    help="(default: 5 7 10 15)",)
    ap.add_argument("-s", "--save", required=False, action='store_true')
    ap.add_argument("-c", "--config", required=False, default="rc.json")

    ap.add_argument("-d", "--difference", required=False, action='store_true')

    args = vars(ap.parse_args())


    from json import load
    with open(args["config"], "r") as f:
        const = load(f)
        

    U  = args["windspeed"]
    band = args["band"]
    for i in range(len(band)):
        if args["save"]:
            data = {}

        const["band"][0] = band[i]
        fig, ax = plt.subplots()

        for j in range(len(U)):
            const["wind.speed"][0] = U[j]
            if args["difference"]:
                bool1 = [False, True, True]
                bool2 = [True, False, True]
                for n in range(len(bool1)):
                    const["spectrum.swell"][0] = bool1[n]
                    const["spectrum.windWaves"][0] = bool2[n]
                    spectrum = Spectrum(const)
                    S = spectrum.get_spectrum()
                    k = spectrum.k0[0:-1:100]
                    ax.loglog(k, k**0*S(k),label="s=%s, w=%s" % (const["spectrum.swell"][0], const["spectrum.windWaves"][0]) )
            else:
                spectrum = Spectrum(const)
                S = spectrum.get_spectrum()
                k = spectrum.k0[0:-1:100]
                ax.loglog(k, k**0*S(k),label="$U=%.1f$" % (U[j]) )

            ax.set_title("%s-диапазон" % band[i])
            ax.set_ylabel("$S(\\kappa),   м^3 \\cdot рад^{-1}$")
            ax.set_xlabel("$\\kappa, рад \\cdot м^{-1}$")

            ax.set_xlim((5e-3,100))
            ax.legend()

            

            if args["save"]:
                import pandas as pd
                data.update({'k'+str(U[j]): k })
                data.update({'S'+str(U[j]): S(k) })
                plt.savefig("spectrum_" + band[i])
                data = pd.DataFrame(data)
                data.to_csv('spectrum_'+band[i]+'.csv', index = False, sep=';')





    plt.show()


