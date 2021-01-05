import numpy as np
from numpy import pi
from scipy import interpolate, integrate
import scipy as sp

from . import rc
from . import spectrum


import matplotlib.pyplot as plt
import os


CACHE_FOLDER = "__pycache__"

class Surface():

    def dispatcher(func):
        """
        Декоратор обновляет необходимые переменные при изменении
        разгона или скорости ветра
        """
        def wrapper(*args):
            self = args[0]
            x = rc.surface.x
            y = rc.surface.y
            gridSize = rc.surface.gridSize
            N = rc.surface.kSize
            M = rc.surface.phiSize

            if self._x.min() != x[0] or \
               self._x.max() != x[1] or \
               self._y.min() != y[0] or \
               self._y.max() != y[1] or \
               self._x.shape != gridSize:
                self.gridUpdate()
            
            if self.N != N or self.M != M:
                self.N, self.M = N, M
                self.amplUpdate()


            return func(*args)

        return wrapper
    def amplUpdate(self):
        self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)

        self.phi = np.linspace(-np.pi, np.pi,self.M + 1, endpoint=True)


        if rc.surface.randomPhases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))
        else:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))

    def gridUpdate(self):

        self._x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
        self._y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
        self._x, self._y = np.meshgrid(self._x, self._y)
        """
        self._z -- высоты морской поверхности
        self._zx -- наклоны X (dz/dx)
        self._zy -- наклоны Y (dz/dy)
        """
        self._z = np.zeros(rc.surface.gridSize)
        self._zx = np.zeros(rc.surface.gridSize)
        self._zy = np.zeros(rc.surface.gridSize)
        self._zz = np.zeros(rc.surface.gridSize)

        # Ссылка на область памяти, где хранятся координаты точек поверхности
        self._r = np.array([
            np.frombuffer(self._x),
            np.frombuffer(self._y),
            np.frombuffer(self._z),
        ], dtype="object")

        self._n = np.array([
            np.frombuffer(self._zx),
            np.frombuffer(self._zy),
            np.frombuffer(self._zz),
        ], dtype="object")

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + (np.ones((r.shape[1], 1)) @ r0).T)

    def __init__(self, spectrum, **kwargs):


        self.gridUpdate()

        self._R = None

        self._A = None
        self._F = None
        self._Psi = None

        self.N = rc.surface.kSize
        self.M = rc.surface.phiSize
        self.amplUpdate()
        # k_edge = spectrum.kEdge






        # print(self.direction)




        # self.k_m = spectrum.k_m

        # if kfrag == 'log':
        #     if k_edge == None:
        #         self.k = np.logspace(np.log10(self.k_m/4), np.log10(self.k_edge['Ku']), self.N + 1)
        #     else:
        #         self.k = np.logspace(np.log10(self.KT[0]), np.log10(self.KT[1]), self.N + 1)
        # elif kfrag == 'quad':
        #     self.k = np.zeros(self.N+1)
        #     for i in range(self.N+1):
        #         self.k[i] = self.k_m/4 + (self.k_edge['Ku']-self.k_m/4)/(self.N+1)**2*i**2

        # else:
        #     self.k = np.linspace(self.k_m/4, self.k_edge['Ku'], self.N + 1)

        # self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)


        # if random_phases == 0:
        #     self.psi = np.array([
        #             [0 for m in range(self.M) ] for n in range(self.N) ])
        # elif random_phases == 1:
        #     self.psi = np.array([
        #         [ np.random.uniform(0,2*pi) for m in range(self.M)]
        #                     for n in range(self.N) ])



        # self.A = self.amplitude(self.k)
        # self.F = self.angle(self.k,self.phi)

        self._mean = None
        self._sigmaxx = None
        self._sigmayy = None

        self._sigmaxy = None


        self._tmean = None
        self._tvar = None
        self._tsigmaxx = None
        self._tsigmayy = None
        self._tsigma = None
        self._tsigmaxy = None


    def _staticMoments(self, x0, y0, surface):

        corr = np.cov(surface[1], surface[2])
        moments = (np.mean(surface[0]), np.var(surface[0]), corr[0][0], corr[1][1], corr[0][0]+corr[1][1])
        print(moments)
        self._mean, self._var, self._sigmaxx, self._sigmayy  = moments[:-1]
        return moments

    def _theoryStaticMoments(self, band, stype="ryabkova"):


        Fx = lambda phi, k: self.Phi(k, phi)  * np.cos(phi)**2
        Fy = lambda phi, k: self.Phi(k, phi)  * np.sin(phi)**2
        Fxy = lambda phi, k: self.Phi(k, phi) * np.sin(phi)*np.cos(phi)
        Q = lambda phi, k: self.Phi(k, phi)

        # self._tvar = self.spectrum.quad(0, stype)

        # print("Before quad", spectrum.k_m, spectrum.limit_k, rc.surface.nonDimWindFetch)
        # self._tsigma = spectrum.quad(2, stype)

        # print("After quad", spectrum.k_m, spectrum.limit_k, rc.surface.nonDimWindFetch)
        self._tsigmaxx = spectrum.dblquad(Fx, stype)
        self._tsigmayy = spectrum.dblquad(Fy, stype)
        # self._tsigmaxy = self.spectrum.dblquad(Fxy, stype)
        # self._tsigma = self._tsigmaxx + self._tsigmayy
        # self._tvar = 0.0081/2 * integrate.quad(lambda x: S(x), KT[0], KT[1], epsabs=1e-6)[0]
        # moments = self._tvar, self._tsigmaxx, self._tsigmayy, self._tsigma, self._tsigma
        # moments = self._tsigma, self._tsigmaxx, self._tsigmayy
        # moments = spectrum.dblquad(Q, stype)
        moments = self._tsigmaxx, self._tsigmayy 
        print(spectrum.peak, rc.surface.nonDimWindFetch, moments)
        return moments
    
    @staticmethod
    def angle_correction(theta):
        # Поправка на угол падения с учётом кривизны Земли
        R = rc.constants.earthRadius
        z = rc.antenna.z
        theta =  np.arcsin( (R+z)/R * np.sin(theta) )
        return theta
    
    # def cwmCorrection(self, moments):
        

    @staticmethod
    def cross_section(theta, cov): 
        theta = Surface.angle_correction(theta)
        theta = theta[np.newaxis]
        # Коэффициент Френеля
        F = 0.8

        if len(cov.shape) <= 2:
            cov = np.array([cov])

        K = np.zeros(cov.shape[0])
        for i in range(K.size):
            K[i] = np.linalg.det(cov[i])

        sigma =  F**2/( 2*np.cos(theta.T)**4 * np.sqrt(K) )
        sigma *= np.exp( - np.tan(theta.T)**2 * cov[:, 1, 1]/(2*K))
        return sigma

    # @property
    # def nonDimWindFetch(self):
    #     return spectrum.nonDimWindFetch

    # @nonDimWindFetch.setter
    # def nonDimWindFetch(self, x):
    #     spectrum.nonDimWindFetch = x
    #     spectrum.peakUpdate(x=x)
    #     self.k_m = spectrum.k_m


    # @property
    # def windSpeed(self):
    #     return spectrum.windSpeed

    # @windSpeed.setter
    # def windSpeed(self, U):
    #     spectrum.windSpeed = U
    #     spectrum.peakUpdate()
    #     self.k_m = spectrum.k_m




    @property
    @dispatcher
    def meshgrid(self):
        r = np.array(self._r[0:2], dtype=float)
        return r.reshape(2, *rc.surface.gridSize)


    @meshgrid.setter
    def meshgrid(self, rho: tuple):
        for i in range(2):
            np.copyto(self._r[i], rho[i])

    @property
    def gridx(self):
        return self.reshape(self._r[0])

    @property
    def gridy(self):
        return self.reshape(self._r[1])

    @property
    def heights(self):
        return self.reshape(self._r[2])

    @heights.setter
    def heights(self, z):
        np.copyto(self._r[2], z)

    @property
    def coordinates(self):
        return self._r

    @coordinates.setter
    def coordinates(self, r):
        for i in range(3):
            np.copyto(self._r[i], r[i])

    @property
    def normal(self):
        return self._n

    @normal.setter
    def normal(self, n):
        for i in range(3):
            np.copyto(self._n[i], n[i])

    @property
    def amplitudes(self):
        return self._A

    @property
    def angleDistribution(self):
        return self._F

    @property
    def phases(self):
        return self._Psi

    @angleDistribution.setter
    def angleDistribution(self, F):
        self._F = F

    @amplitudes.setter
    def amplitudes(self, A):
        self._A = A

    @phases.setter
    def phases(self, Psi):
        self._Psi = Psi
    
    def reshape(self, arr):
        return np.array(arr, dtype=float).reshape(*rc.surface.gridSize)
    
    def plot(self, x, y, surf, label = "default"):

        surf = self.reshape(surf)
        x = self.reshape(x)
        y = self.reshape(y)

        fig, ax = plt.subplots()
        mappable = ax.contourf(x, y,surf, levels=100)
        ax.set_xlabel("$x,$ м")
        ax.set_ylabel("$y,$ м")
        bar = fig.colorbar(mappable=mappable, ax=ax)
        bar.set_label("высота, м")

        ax.set_title("$U_{10} = %.0f $ м/с" % (spectrum.U10) )
        ax.text(0.05,0.95,
            '\n'.join((
                    '$\\sigma^2_s=%.5f$' % (np.std(surf)**2/2),
                    '$\\sigma^2_{0s}=%.5f$' % (spectrum.sigma_sqr),
                    '$\\langle z \\rangle = %.5f$' % (np.mean(surf)),
            )),
            verticalalignment='top',transform=ax.transAxes,)

        fig.savefig("%s" % (label))




    # def B(self,k):

    #     if not isinstance(k, np.ndarray):
    #         k = np.array([k])

    #     km = spectrum.peak
    #     def b(k):
    #         k[np.where(k/km < 0.4)] = km * 0.4
    #         b=(
    #             -0.28+0.65*np.exp(-0.75*np.log(k/km))
    #             +0.01*np.exp(-0.2+0.7*np.log10(k/km))
    #         )
    #         return b

    #     B=10**b(k)
    #     return B

    # def Phi(self,k,phi):

    #     # Функция углового распределения

    #     if not isinstance(phi, np.ndarray):
    #         phi = np.array([phi])

    #     phi -= self.direction
    #     index = np.where(np.abs(phi) > np.pi)
    #     phi[index] = np.sign(phi[index])*(2*np.pi - np.abs(phi[index]))

    #     B0 = self.B(k)
    #     normalization = lambda B: B/np.arctan(np.sinh(2*pi*B))

    #     A0 = normalization(B0)

    #     phi = phi[np.newaxis]
    #     Phi = A0/np.cosh(2*B0*phi.T )
    #     return Phi.T


    def angle(self,k, phi):
        M = self.M
        N = self.N
        Phi = lambda phi,k: spectrum.azimuthal_distribution(k,phi)
        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                # integral[i][j] = integrate.quad( Phi, phi[j], phi[j+1], args=(k[i],) )[0]
                integral[i][j] = np.trapz( Phi( phi[j:j+2],k[i] ), phi[j:j+2])

        amplitude = np.sqrt(2*integral)
        # print(integral.shape)
        # if True:
        #     print(np.sum(integral)/(k[-1]-k[0]))

        return amplitude

    def amplitude(self, k):
        N = k.size
        S = spectrum.get_spectrum()
        integral = np.zeros(k.size-1)
        for i in range(1,N):
            integral[i-1] = integrate.quad(S,k[i-1], k[i])[0]

        # if True:
            # print(np.sum(integral), spectrum.quad(0))

        amplitude = np.sqrt(integral )
        return amplitude
    
    def ampld(self, k, phi):
        M = self.M
        N = self.N

        Phi = lambda phi, k: spectrum.azimuthal_distribution(k, phi)
        S = spectrum.get_spectrum()
        integral = np.zeros((N,M))

        for i in range(N):
            A = integrate.quad(S,k[i], k[i+1])[0]
            for j in range(M):
                if self.M > 2:
                    F = np.trapz( Phi( phi[j:j+2], k[i:i+1] ), phi[j:j+2])
                else:
                    F = 1

                integral[i][j] =  A * F

        amplitude = np.sqrt(2*integral)
        return amplitude


    # def export(self):
    #     spectrum.peakUpdate()
    #     # print(spectrum.KT)
    #     self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)
    #     # self.k_m = spectrum.k_m
    #     self.A = self.amplitude(self.k)
    #     self.F = self.angle(self.k, self.phi)

    #     # A = self.ampld(self.k, self.phi)
    #     # print( self.A[None].T*self.F.all() == A.all())

    #     # 1024 and 128 -- хватает
    #     # eps = 1e-6
    #     # var_theory  = spectrum.dblquad(0, 0, 0)
    #     # var_real = np.sum( (self.A[None].T * self.F)**2 )
    #     # if np.abs(var_theory - var_real) > eps:

    #     k = self.k[None].T * np.exp(1j*self.phi[None])


    #     return self.k, self.phi, self.A, self.F, self.psi

    @dispatcher
    def export(self):
        # spectrum.peakUpdate()
        # self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)

        srf = rc.surface
        # Aname = "A_%s_%s_%s_%s_%s.npy" % (srf.band, srf.kSize, srf.phiSize, rc.wind.speed, rc.wind.direction)
        # Apath = os.path.join( CACHE_FOLDER, Aname)

        k = self.k[None].T * np.exp(1j*self.phi[None])
        k = np.array(k[:-1, :-1])
        A0 = self.ampld(self.k, self.phi)

        return k, A0*np.exp(1j*self.psi)




    def integrate(self, x, y, i, j):
        dx = x[j] - x[i]
        if np.abs(dx) < np.abs(0.01*x[j]):
            integral = 0
        else:
            integral = (y[i] + y[j] )/2 * dx
        return integral


if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="This script may plot  2D realizations of rough water surface.")
    ap.add_argument("-c", "--config", required=False, default="rc.json", help = "path to custom configuration file (default: ./rc.json)")
    args = vars(ap.parse_args())

    surface = Surface()
    kernels = [kernel_default]

    for U in [5, 10, 15]:
        rc.wind.speed = U
        rc.surface.nonDimWindFetch = 900
        arr, X0, Y0 = run_kernels(kernels, surface)
        for i in range(len(kernels)):
            surface.plot(X0[i], Y0[i], arr[i][0], label="default%s" % (U))


    # surface.spectrum.plot()



    # plt.show()
