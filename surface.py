import numpy as np
from numpy import pi
from scipy import interpolate,integrate
from tqdm import tqdm
from spectrum import Spectrum




class Surface(Spectrum):
    def __init__(self,const):

        self.N = const["surface"]["kSize"][0]
        k_edge = const["surface"]["kEdge"][0]
        self.M = const["surface"]["phiSize"][0]
        random_phases = const["surface"]["randomPhases"][0]
        kfrag = "log"



        self.direction = []

        if const["wind"]["enable"][0]:
            self.direction.append(np.deg2rad(const["wind"]["direction"][0]))

        if const["swell"]["enable"][0]:
            self.direction.append(np.deg2rad(const["swell"]["direction"][0]))

        self.direction = np.array(self.direction)

        print(self.direction)

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
        # self.A = np.empty(shape=(self.N, self.M))
        # self.F = np.copy(self.A)

        # for i, spectrum in enumerate(self.spectrum):
        spectrum = self.spectrum
        self.A = self.amplitude(self.k, spectrum)
        self.F = self.angle(self.k,self.phi, direction=self.direction[0])
            # if i == 0:
                # self.A = np.array([A])
                # self.F = np.array([F])
            # else:
                # self.A = np.vstack((self.A, np.array([A])))
                # self.F = np.vstack((self.F, np.array([F])))


        if random_phases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.direction.size, self.N, self.M))
        else:
            self.psi = np.random.uniform(0, 2*pi, size=(self.direction.size, self.N, self.M))



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
    
    def Surface_space(self,r):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi
        self.surface = [0 for i in range(A.shape[0])]
        # print(A.shape[0])
        progress_bar = tqdm( total = A.shape[0]*N*M,  leave = False , desc="Surface calc")
        for s in range(A.shape[0]):
            for n in range(N):
                for m in range(M):
                    kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                    tmp = A[s][n] * \
                        np.cos( kr + psi[s][n][m]) * F[s][n][m]

                    progress_bar.update(1)
                    self.surface[s] += tmp
            # print(self.surface.shape)
            # print(self.surface)


        progress_bar.close()
        progress_bar.clear()
        heights = self.surface
        return heights

    def Surface_time(self,t):
        N = self.N
        M= self.M
        k = self.k
        A = self.A
        F = self.F
        psi = self.psi
        self.surface = 0
        progress_bar = tqdm( total = N*M,  leave = False , desc="Surface calc")
        for n in range(N):
            for m in range(M):
                tmp = A[n] * \
                    np.cos( 
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]


                self.surface += tmp

                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        heights = self.surface
        return heights

    def surfaces_band(self,r,t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        s = 0
        s_xx = 0
        s_yy = 0

        s1 = 0
        s_xx1 = 0
        s_yy1 = 0

        progress_bar = tqdm( total = N*M,  leave = False )

        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = A[n] * \
                    np.cos( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                tmp1 = -A[n] * \
                    np.sin( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                if k[n] <= self.k_c.max():
                    s += tmp
                    s_xx += k[n]*np.cos(phi[m])*tmp1
                    s_yy += k[n]*np.sin(phi[m])*tmp1 

                else :
                    s1 += tmp
                    s_xx1 += k[n]*np.cos(phi[m])*tmp1
                    s_yy1 += k[n]*np.sin(phi[m])*tmp1

                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        band_c = [s, s_xx, s_yy]
        band_ku = [s+s1, s_xx + s_xx1, s_yy + s_yy1]
        return band_c, band_ku

    def choppy_wave_space(self, r):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        self.cwm_x = 0
        self.cwm_y = 0

        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = -A[n] * \
                    np.sin( 
                        +kr
                        +psi[n][m]) \
                        * F[n][m]

                self.cwm_x += tmp * np.cos(phi[m])
                self.cwm_y += tmp * np.sin(phi[m])
        return [self.cwm_x, self.cwm_y]


    def choppy_wave_time(self, t):
        N = self.N
        M= self.M
        k = self.k
        A = self.A
        F = self.F
        psi = self.psi

        self.cwm_t = 0
        for n in range(N):
            for m in range(M):
                tmp = -A[n] *  np.sin( self.omega_k(k[n])*t  +psi[n][m])  * F[n][m]
                self.cwm_t += tmp 

        return self.cwm_t


    def choppy_wave_jac(self, r, t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        self.cwm_x_dot = 0
        self.cwm_y_dot = 0
        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = -A[n] * \
                    np.cos( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                self.cwm_x_dot += tmp * k[n]*np.cos(phi[m])**2
                self.cwm_y_dot += tmp * k[n]*np.sin(phi[m])**2
        return [self.cwm_x_dot, self.cwm_y_dot]

    
if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt


    ap = argparse.ArgumentParser(description="This script may plot 1D or/and 2D realizations of rough water surface.")


    ap.add_argument("-c", "--config", required=False, default="rc.json", help = "path to custom configuration file (default: ./rc.json)")
    ap.add_argument("-s", "--save", required=False, action='store_true', help = "save all plots in script in pdf file and import data in csv file")
    ap.add_argument("-t", "--timeplot", required=False, action='store_true', help = "plot evolution of surface at a point (0, 0) in 15 min")
    ap.add_argument("-x", "--spaceplot", required=False, action='store_true', help = "plot snapshot of surface")
    args = vars(ap.parse_args())

    from json import load
    with open(args["config"], "r") as f:
        const = load(f)

    if not args["spaceplot"] and not args["timeplot"]:
            ap.print_help()
            sys.exit(1)

    x_size = const["surface"]["x"][0]
    y_size = const["surface"]["y"][0]
    grid_size = const["surface"]["gridSize"][0]


    args = vars(ap.parse_args())


    x = np.linspace(-x_size, x_size, grid_size)
    y = np.linspace(-x_size, x_size, grid_size)
    t = np.arange(0, 900, 0.5)

    X, Y = np.meshgrid(x, y)
    surface = Surface(const)


    if args["spaceplot"]:
        Heights = surface.Surface_space([X,Y]) 
        for heights in Heights:
            fig,ax = plt.subplots()
            mappable = ax.contourf(X,Y,heights, levels=100)
            ax.set_xlabel("$x,$ м")
            ax.set_ylabel("$y,$ м")
            bar = fig.colorbar(mappable=mappable,ax=ax)
            bar.set_label("высота, м")


            ax.set_title("$U_{10} = %.0f $ м/с" % (surface.U10) )
            ax.text(0.05,0.95,
                '\n'.join((
                        '$\\sigma^2_s=%.2f$' % (np.std(heights)**2), 
                        '$\\langle z \\rangle = %.2f$' % (np.mean(heights)),
                )),
                verticalalignment='top',transform=ax.transAxes,)

    if args["timeplot"]:
            heights = surface.Surface_time(t) 
            fig,ax = plt.subplots()
            ax.plot(t,heights)
            ax.set_xlabel("$t,$ с")
            ax.set_ylabel("высота, м")

            ax.set_title("$U_{10} = %.0f $ м/с" % (surface.U10) )
            ax.text(0.05,0.95,
                '\n'.join((
                        '$\\sigma^2_s=%.2f$' % (np.std(heights)**2), 
                        '$\\langle z \\rangle = %.2f$' % (np.mean(heights)),
                )),
                verticalalignment='top',transform=ax.transAxes,)



    plt.show()