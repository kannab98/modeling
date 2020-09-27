import numpy as np
from numpy import pi
from scipy import interpolate, integrate

from tqdm import tqdm
from spectrum import Spectrum


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
                    x[i] += -Sin * math.cos(phi[m])
                    y[i] += -Sin * math.sin(phi[m])
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

else:
    def kernel_cwm(ans, x, y, k, phi, A, F, psi):
        for n in range(k.size): 
            for m in range(phi.size):
                    kr = k[n]*(x*math.cos(phi[m]) + y*math.sin(phi[m]))      
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
                    x += -Sin * math.cos(phi[m])
                    y += -Sin * math.sin(phi[m])
                    ans[1,i] *= 1 - Cos * math.cos(phi[m]) * kx
                    ans[2,i] *= 1 - Cos * math.sin(phi[m]) * ky

        return ans,x,y

    def kernel_default(ans, x, y, k, phi, A, F, psi):
        for n in range(k.size): 
            for m in range(phi.size):
                    kr = k[n]*(x*math.cos(phi[m]) + y*math.sin(phi[m]))      
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

        return ans,x,y



def init_kernel(kernel, arr, X, Y, host_constants):
    if GPU:
        cuda_constants = (cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
        threadsperblock = TPB 
        blockspergrid = math.ceil(X.size / threadsperblock)
        kernel[blockspergrid, threadsperblock](arr, X, Y, *cuda_constants)
    else:
        kernel(arr, X, Y, host_constants)



def run_kernels(kernels,  X, Y, host_constants):
    process = [ None for i in range( len(kernels) )]
    arr = [ None for i in range( len(kernels) )]
    X0 = [ None for i in range( len(kernels) )]
    Y0 = [ None for i in range( len(kernels) )]

    for j, kernel in enumerate(kernels):
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

    for i in range(len(kernels)):
        # wait until process funished
        process[i].join()

    return arr, X0, Y0


class Surface(Spectrum):
    def __init__(self,const):

        self.N = const["surface"]["kSize"][0]
        k_edge = const["surface"]["kEdge"][0]
        self.M = const["surface"]["phiSize"][0]
        random_phases = const["surface"]["randomPhases"][0]
        kfrag = "log"
        self.grid_size = const["surface"]["gridSize"][0]



        self.direction = []

        if const["wind"]["enable"][0]:
            self.direction.append(np.deg2rad(const["wind"]["direction"][0]))

        if const["swell"]["enable"][0]:
            self.direction.append(np.deg2rad(const["swell"]["direction"][0]))

        self.direction = np.array(self.direction)

        # print(self.direction)

        Spectrum.__init__(self, const)
        self.spectrum = self.get_spectrum()



        if kfrag == 'log':
            if k_edge == None:
                self.k = np.logspace(np.log10(self.k_m/4), np.log10(self.k_edge['Ku']), self.N + 1)
            else:
                self.k = np.logspace(np.log10(self.KT[0]), np.log10(self.KT[1]), self.N + 1)

        elif kfrag == 'quad':
            self.k = np.zeros(self.N+1)
            for i in range(self.N+1):
                self.k[i] = self.k_m/4 + (self.k_edge['Ku']-self.k_m/4)/(self.N+1)**2*i**2

        else:
            self.k = np.linspace(self.k_m/4, self.k_edge['Ku'], self.N + 1)

        self.k_ku = self.k[ np.where(self.k <= self.k_edge['Ku']) ]
        self.k_c = self.k[ np.where(self.k <= self.k_edge['C']) ]


        if const["debug"] == True:
            print(\
                "Параметры модели:\n\
                    N={},\n\
                    M={},\n\
                    U={} м/с,\n\
                    Band={}\n\
                    k=[{:.2f},{:.1f}]\n mean=0".format(self.N,self.M,self.U10,self.band,self.k.min(), self.k.max())
                )

        self.phi = np.linspace(-np.pi, np.pi,self.M + 1)

        spectrum = self.spectrum

        self.A = self.amplitude(self.k, spectrum)
        self.F = self.angle(self.k, self.phi, direction=self.direction[0])


        if random_phases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))
        else:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))



    def B(self,k):
          def b(k):
              b=(
                  -0.28+0.65*np.exp(-0.75*np.log(k/self.k_m))
                  +0.01*np.exp(-0.2+0.7*np.log10(k/self.k_m))
                )
              return b
          B=10**b(k)
          return B

    def Phi(self,k,phi, direction = 0):
        # Функция углового распределения
        phi = phi - direction
        normalization = lambda B: B/np.arctan(np.sinh(2* (pi)*B))
        B0 = self.B(k)
        A0 = normalization(B0)
        Phi = A0/np.cosh(2*B0*(phi) )
        return Phi


    def angle(self,k, phi, direction):
        M = self.M
        N = self.N


        Phi = lambda phi,k, direction: self.Phi(k, phi, direction)
        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                # integral[i][j] = integrate.quad( Phi, phi[j], phi[j+1], args=(k[i],) )[0]
                integral[i][j] += np.trapz( Phi( phi[j:j+2],k[i], direction=direction), phi[j:j+2], )
        amplitude = np.sqrt(2 *integral )
        return amplitude

    def amplitude(self, k, spectrum):
        N = k.size

        integral = np.zeros(k.size-1)
        # for S in self.spectrum:
        for i in range(1,N):
            integral[i-1] += integrate.quad(spectrum,k[i-1],k[i])[0]

        amplitude = np.sqrt(2 *integral )
        return np.array(amplitude)

    def export(self):
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

    def get_moments(self, x0, y0, surface):
        moments = np.zeros(3)
        params = ["mean", "sigmaxx", "sigmayy", ]
        for i, param in enumerate(params):
            if param == "mean":
                moments[i] = self.moment(x0, y0, surface[i], p=1)
                # moments[i] = np.mean(surface[i])
            else:
                moments[i] = np.abs(self.moment(x0, y0, surface[i], p=2) -  moments[0]**2 )
                # moments[i] = np.std(surface[i])**2 

        return moments


if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt


    ap = argparse.ArgumentParser(description="This script may plot  2D realizations of rough water surface.")


    ap.add_argument("-c", "--config", required=False, default="rc.json", help = "path to custom configuration file (default: ./rc.json)")
    args = vars(ap.parse_args())

    from json import load
    with open(args["config"], "r") as f:
        const = load(f)


    x_size = const["surface"]["x"][0]
    y_size = const["surface"]["y"][0]
    grid_size = const["surface"]["gridSize"][0]


    args = vars(ap.parse_args())


    x = np.linspace(-x_size, x_size, grid_size)
    y = np.linspace(-x_size, x_size, grid_size)
    t = np.arange(0, 900, 0.5)

    X, Y = np.meshgrid(x, y)
    surface = Surface(const)

    host_constants = surface.export()





    if True:
        kernels = [kernel_default]
        labels = ["default", "cwm"] 

        process = [ None for i in range( len(kernels) )]
        arr = [ None for i in range( len(kernels) )]
        X0 = [ None for i in range( len(kernels) )]
        Y0 = [ None for i in range( len(kernels) )]

        for j, kernel in enumerate(kernels):
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


            process[j] = Process(target = init_kernel, args = (kernel, arr[j], X0[j], Y0[j], host_constants))
            process[j].start()
        




        for i in range(len(kernels)):
            # wait until process funished
            process[i].join()
        

        fig1d, ax1d = plt.subplots()
        for i in range(len(kernels)):
            fig, ax = plt.subplots()
            surf = arr[i][0].reshape((grid_size, grid_size))
            x = X0[i].reshape((grid_size, grid_size))
            y = Y0[i].reshape((grid_size, grid_size))

            mappable = ax.contourf(x, y,surf, levels=100)
            ax.set_xlabel("$x,$ м")
            ax.set_ylabel("$y,$ м")
            bar = fig.colorbar(mappable=mappable, ax=ax)
            bar.set_label("высота, м")

            ax.set_title("$U_{10} = %.0f $ м/с" % (surface.U10) )
            ax.text(0.05,0.95,
                '\n'.join((
                        '$\\sigma^2_s=%.5f$' % (np.std(surf)**2),
                        '$\\langle z \\rangle = %.5f$' % (np.mean(surf)),
                )),
                verticalalignment='top',transform=ax.transAxes,)

            fig.savefig("%s" % (labels[i]))

            ax1d.plot(x[0], surf[0])
        
        fig1d.savefig("compare")



    # plt.show()
