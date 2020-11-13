import numpy as np
from numpy import pi
from scipy import interpolate, integrate
import scipy as sp
from json import load,  dump

from tqdm import tqdm
from .spectrum import Spectrum
import matplotlib.pyplot as plt
import os

try:
    from numba import cuda
    from multiprocessing import Process, Array
    GPU = True
except:
    print("CUDA not installed")
    GPU = False 
import math

TPB=16
if GPU:
    @cuda.jit
    def kernel_cwm_offset(x, y, k, phi, A, F, psi):
        if i >= x.shape[0]:
            return

        for n in range(k.size): 
            for m in range(phi.size):
                kr = k[n]*(x[i]*math.cos(phi[m]) + y[i]*math.sin(phi[m]))      
                Af = A[n] * F[n][m]
                Sin =  math.sin(kr + psi[n][m]) * Af
                x[i] += -Sin * math.cos(phi[m])
                y[i] += -Sin * math.sin(phi[m])

    @cuda.jit
    def kernel_cwm(ans, x, y, k, phi, A, F, psi):
        i = cuda.grid(1)

        if i >= x.shape[0]:
            return

        for n in range(k.size): 
            for m in range(phi.size):
                    kr = k[n]*(x[i]*math.cos(phi[m]) + y[i]*math.sin(phi[m]))      
                    Af = A[n] * F[n][m]
                    Cos =  math.cos(kr + psi[n][m]) * Af
                    Sin =  math.sin(kr + psi[n][m]) * Af

                    kx = k[n] * math.cos(phi[m])
                    ky = k[n] * math.sin(phi[m])


                    # Высоты (z)
                    ans[0,i] +=  Cos 
                    # Наклоны X (dz/dx)
                    ans[1,i] +=  -Sin * kx
                    # Наклоны Y (dz/dy)
                    ans[2,i] +=  -Sin * ky

                    # CWM
                    x[i] += -100*Sin * math.cos(phi[m])
                    y[i] += -100*Sin * math.sin(phi[m])
                    ans[1,i] *= 1 - Cos * math.cos(phi[m]) * kx
                    ans[2,i] *= 1 - Cos * math.sin(phi[m]) * ky


    @cuda.jit
    def kernel_default(ans, x, y, k, phi, A, F, psi):
        i = cuda.grid(1)

        if i >= x.shape[0]:
            return

        for n in range(k.size): 
            for m in range(phi.size):
                    kr = k[n]*(x[i]*math.cos(phi[m]) + y[i]*math.sin(phi[m]))      
                    Af = A[n] * F[n][m]
                    Cos =  math.cos(kr + psi[n][m]) * Af
                    Sin =  math.sin(kr + psi[n][m]) * Af

                    kx = k[n] * math.cos(phi[m])
                    ky = k[n] * math.sin(phi[m])


                    # Высоты (z)
                    ans[0,i] +=  Cos 
                    # Наклоны X (dz/dx)
                    ans[1,i] +=  -Sin * kx
                    # Наклоны Y (dz/dy)
                    ans[2,i] +=  -Sin * ky


def init_kernel(kernel, arr, X, Y, host_constants):
    if GPU:
        cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
        threadsperblock = TPB 
        blockspergrid = math.ceil(X.size / threadsperblock)

        kernel[blockspergrid, threadsperblock](arr, X, Y, *cuda_constants)
    else:
        kernel(arr, X, Y, host_constants)



def run_kernels(kernel,  surface: object):
    N = surface.spectrum.KT.size - 1

    process = [ None for i in range(N)]
    arr = [ None for i in range(N)]
    X0 = [ None for i in range(N)]
    Y0 = [ None for i in range(N)]


    X, Y = surface.meshgrid
    model_coeffs = surface.export()
    k = model_coeffs[0]

    edge = surface.spectrum.KT

    edge = [ np.max(np.where(k <= edge[i] )) for i in range(1, edge.size)]
    edge = [0] + edge

    for j in range(N):
        # Срез массивов коэффициентов по оси K, условие !=1 для phi (он не зависит от band)
        host_constants = tuple([ model_coeffs[i][edge[j]:edge[j+1]] if i !=1 else model_coeffs[i] for i in range(len(model_coeffs))])
        # Create shared array
        arr_share = Array('d', 3*X.size )
        X_share = Array('d', X.size)
        Y_share = Array('d', Y.size)
        # arr_share and arr share the same memory
        arr[j] = np.frombuffer( arr_share.get_obj() ).reshape((3, X.size)) 
        X0[j] = np.frombuffer( X_share.get_obj() )
        Y0[j] = np.frombuffer( Y_share.get_obj() )
        np.copyto(X0[j], X.flatten())
        np.copyto(Y0[j], Y.flatten())

        process[j] = Process(target = init_kernel, args = (kernel, arr[j], X0[j], Y0[j], host_constants) )
        process[j].start()




    # wait until process funished
    for j in range(N):
        process[j].join()

    # # Экспериментальная часть 
    if (X.flatten() - X0[0]).any() != 0:
        for j in range(N):
            # Срез массивов коэффициентов по оси K, условие !=1 для phi (он не зависит от band)
            host_constants = tuple([ model_coeffs[i][edge[j]:edge[j+1]] if i !=1 else model_coeffs[i] for i in range(len(model_coeffs))])
            # Create shared array
            arr_share = Array('d', 3*X.size )
            X_share = Array('d', X.size)
            Y_share = Array('d', Y.size)
            # arr_share and arr share the same memory
            arr[j] = np.frombuffer( arr_share.get_obj() ).reshape((3, X.size)) 
            X0[j] = np.frombuffer( X_share.get_obj() )
            Y0[j] = np.frombuffer( Y_share.get_obj() )
            np.copyto(X0[j], X.flatten())
            np.copyto(Y0[j], Y.flatten())

            dx = X.flatten() - X0[j]
            process[j] = Process(target = init_kernel, args = (kernel, arr[j], X.flatten() - X0[j], 2*Y.flatten() - Y0[j], host_constants) )
            process[j].start()



        # wait until process funished
        for j in range(N):
            process[j].join()

    # print(arr[0][0], arr[1][0])

    # for i in range(len(kernels)):
    #     # wait until process funished
    #     surface.staticMoments = surface._staticMoments(X0[i], Y0[i], arr[i])

    return arr, X0, Y0


class Surface():

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + (np.ones((r.shape[1], 1)) @ r0).T)

    def __init__(self, **kwargs):

        config  = kwargs["config"] if "config" in kwargs else os.path.join(os.path.abspath(os.getcwd()), "rc.json")

        with open(config) as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["surface"])
        spectrum = Spectrum(**kwargs)
        self.spectrum = spectrum

        self._x = np.linspace(-self.x, self.x, self.gridSize)
        self._y = np.linspace(-self.y, self.y, self.gridSize)
        self._x, self._y = np.meshgrid(self._x, self._y)
        """
        self._z -- высоты морской поверхности
        self._zx -- наклоны X (dz/dx)
        self._zy -- наклоны Y (dz/dy)
        """
        self._z = np.zeros((self.gridSize, self.gridSize))
        self._zx = np.zeros((self.gridSize, self.gridSize))
        self._zy = np.zeros((self.gridSize, self.gridSize))
        self._zz = np.ones((self.gridSize, self.gridSize))

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

        self._R = None

        self._A = None
        self._F = None
        self._Psi = None

        self.N = self.kSize
        self.M = self.phiSize
        k_edge = self.kEdge

        random_phases = self.randomPhases
        kfrag = "log"
        self.grid_size = self.gridSize

        self.k = np.logspace(np.log10(self.spectrum.KT[0]), np.log10(self.spectrum.KT[-1]), self.N + 1)
        self.k_m = self.spectrum.k_m



        self.direction = []

        if spectrum.wind.enable:
            self.direction.append(np.deg2rad(spectrum.wind.direction))

        if spectrum.swell.enable:
            self.direction.append(np.deg2rad(spectrum.swell.direction))

        self.direction = np.array(self.direction)

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
        self.phi = np.linspace(-np.pi, np.pi,self.M + 1, endpoint=False)


        if random_phases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))
        else:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))

        if random_phases == 0:
            self.psi = np.array([
                    [0 for m in range(self.M) ] for n in range(self.N) ])
        elif random_phases == 1:
            self.psi = np.array([
                [ np.random.uniform(0,2*pi) for m in range(self.M)]
                            for n in range(self.N) ])



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
        self._mean, self._var, self._sigmaxx, self._sigmayy  = moments[:-1]
        return moments

    def _theoryStaticMoments(self, band, stype="ryabkova"):


        Fx = lambda phi, k: self.Phi(k, phi) * np.cos(phi)**2
        Fy = lambda phi, k: self.Phi(k, phi) * np.sin(phi)**2
        Fxy = lambda phi, k: self.Phi(k, phi) * np.sin(phi)*np.cos(phi)
        Q = lambda phi, k: F(phi, k)* S(k) * k**2 

        self._tvar = self.spectrum.quad(0, stype)
        self._tsigma = self.spectrum.quad(2, stype)
        self._tsigmaxx = self.spectrum.dblquad(Fx, stype)
        self._tsigmayy = self.spectrum.dblquad(Fy, stype)
        # self._tsigmaxy = self.spectrum.dblquad(Fxy, stype)
        # self._tsigma = self._tsigmaxx + self._tsigmayy
        # self._tvar = 0.0081/2 * integrate.quad(lambda x: S(x), KT[0], KT[1], epsabs=1e-6)[0]
        # moments = self._tvar, self._tsigmaxx, self._tsigmayy, self._tsigma, self._tsigmaxy
        # print(moments, self.spectrum.band)
        return moments
    
    def angleCorrection(self, theta):
        # Поправка на угол падения с учётом кривизны Земли
        R = self._rc["constants"]["earthRadius"]
        z = self._rc["antenna"]["z"]
        theta =  np.arcsin( (R+z)/R * np.sin(theta) )
        return theta
    
    # def cwmCorrection(self, moments):
        

    def crossSection(self, theta, moments, ): 
        var = moments[1:]
        # theta = self.angleCorrection(theta)
        # Коэффициент Френеля
        F = 0.8
        sigma =  F**2/( 2*np.cos(theta)**4 * np.sqrt(var[0]*var[1] - var[2]**2) )
        sigma *= np.exp( - np.tan(theta)**2 * var[1]/(2*var[0]*var[1] - 2*var[2]**2))
        return sigma

    @property
    def nonDimWindFetch(self):
        return self.spectrum.nonDimWindFetch

    @nonDimWindFetch.setter
    def nonDimWindFetch(self, x):
        self.spectrum.nonDimWindFetch = x
        self.spectrum.peakUpdate()
        self.k_m = self.spectrum.k_m


    @property
    def windSpeed(self):
        return self.spectrum.windSpeed

    @windSpeed.setter
    def windSpeed(self, U):
        self.spectrum.windSpeed = U
        self.spectrum.peakUpdate()
        self.k_m = self.spectrum.k_m



    @property
    def meshgrid(self):
        r = np.array(self._r[0:2], dtype=float)
        return r.reshape((2, self.gridSize, self.gridSize))

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
        return np.array(arr, dtype=float).reshape((self.gridSize, self.gridSize))
    
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

        ax.set_title("$U_{10} = %.0f $ м/с" % (self.spectrum.U10) )
        ax.text(0.05,0.95,
            '\n'.join((
                    '$\\sigma^2_s=%.5f$' % (np.std(surf)**2/2),
                    '$\\sigma^2_{0s}=%.5f$' % (self.spectrum.sigma_sqr),
                    '$\\langle z \\rangle = %.5f$' % (np.mean(surf)),
            )),
            verticalalignment='top',transform=ax.transAxes,)

        fig.savefig("%s" % (label))




    def B(self,k):

        if not isinstance(k, np.ndarray):
            k = np.array([k])

        def b(k):
            k[np.where(k/self.k_m < 0.4)] = self.k_m * 0.4
            b=(
                -0.28+0.65*np.exp(-0.75*np.log(k/self.k_m))
                +0.01*np.exp(-0.2+0.7*np.log10(k/self.k_m))
            )
            return b

        B=10**b(k)
        return B

    def Phi(self,k,phi):
        # Функция углового распределения
        phi = phi - self.direction[0]


        normalization = lambda B: B/np.arctan(np.sinh(2*pi*B))

        B0 = self.B(k)

        A0 = normalization(B0)
        phi = phi[np.newaxis]
        Phi = A0/np.cosh(2*B0*phi.T )
        return Phi.T


    def angle(self,k, phi):
        M = self.M
        N = self.N
        Phi = lambda phi,k: self.Phi(k,phi)
        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                # integral[i][j] = integrate.quad( Phi, phi[j], phi[j+1], args=(k[i],) )[0]
                integral[i][j] = np.trapz( Phi( phi[j:j+2],k[i] ), phi[j:j+2])

        amplitude = np.sqrt(2 *integral )
        return amplitude

    def amplitude(self, k):
        N = k.size
        S = self.spectrum.get_spectrum()
        integral = np.zeros(k.size-1)
        for i in range(1,N):
            integral[i-1] += integrate.quad(S,k[i-1],k[i])[0]
        amplitude = np.sqrt(integral )
        return np.array(amplitude)

    def export(self):

        self.k = np.logspace(np.log10(self.spectrum.KT[0]), np.log10(self.spectrum.KT[-1]), self.N + 1)
        self.k_m = self.spectrum.k_m
        self.A = self.amplitude(self.k)
        self.F = self.angle(self.k, self.phi)
        return self.k, self.phi, self.A, self.F, self.psi


    def integrate(self, x, y, i, j):
        dx = x[j] - x[i]
        if np.abs(dx) < np.abs(0.01*x[j]):
            integral = 0
        else:
            integral = (y[i] + y[j] )/2 * dx
        return integral

    def moment(self, x0, y0, surface, p=1):

        grid_size = self.grid_size
        x0 = x0.reshape((grid_size, grid_size))
        y0 = y0.reshape((grid_size, grid_size))
        z0 = surface.reshape((grid_size,grid_size))

        S = np.zeros((grid_size-1, grid_size-1))
        Z = np.zeros((grid_size-1, grid_size-1))

        for m in range(grid_size-1):
            for n in range(grid_size-1):
                x = np.array([x0[i,j] for i in range(m,2+m) for j in range(n,n+2)])
                y = np.array([y0[i,j] for i in range(m,2+m) for j in range(n,n+2)])
                z = np.array([z0[i,j] for i in range(m,2+m) for j in range(n,n+2)])

                walk = [0,2,3,1]

                x = x[walk]
                y = y[walk]
                z = z[walk]

                s = lambda i,j: self.integrate(x, y, i, j)
                Z[m,n] = np.mean(z)
                S[m,n] = np.abs(+ s(0,1) + s(1,2) + s(2,3)  + s(3,1))

        return (np.sum(S*Z**p)/np.sum(S))




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
        surface.windSpeed = U
        surface.nonDimWindFetch = 900
        arr, X0, Y0 = run_kernels(kernels, surface)
        for i in range(len(kernels)):
            surface.plot(X0[i], Y0[i], arr[i][0], label="default%s" % (U))


    # surface.spectrum.plot()



    # plt.show()
