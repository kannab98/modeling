import numpy as np

class Pulse():
    def __init__(self, surface, x, y, const):
        # if len(x.shape) < 2:
        #     self.x, self.y = np.meshgrid(x,y)
        # else:
        self.x, self.y = x, y

        self.r = np.vstack((
                            self.x.flatten(),
                            self.y.flatten(),
                            surface[0].flatten()
                        ))

        self.z0 = const["antenna"]["z"][0] + surface[0].max()
        r0 = [ const["antenna"]["x"][0], const["antenna"]["y"][0], const["antenna"]["z"][0]]

        self.r0 = np.vstack((
                      +r0[0]*np.zeros(self.x.size),
                      +r0[1]*np.zeros(self.y.size),
                      -r0[2]*np.ones(surface[0].size)
                    ))

        self.n = np.vstack((
                        surface[1].flatten(),
                        surface[2].flatten(),
                        np.ones(surface[0].size)
                    ))


        self.timp = const["antenna"]["impulseDuration"][0]
        self.c =  const["constants"]["lightSpeed"][0]
        self.R = self.r - self.r0

        self.Rabs = self.Rabs_calc(self.R)
        self.Nabs = self.Nabs_calc(self.n)
        self.theta = self.theta_calc(self.R, self.Rabs)
        self.theta0 = self.theta0_calc(self.R, self.n, self.Rabs, self.Nabs)

        #!$gane\_width \equiv \theta_{3dB}$!
        gane_width = np.deg2rad(const["antenna"]["gainWidth"][0]) # Ширина диаграммы направленности в радианах
        self.gamma = 2*np.sin(gane_width/2)**2/np.log(2)

        sigmaxx = const["surface"]["sigmaxx"][0]
        sigmayy = const["surface"]["sigmayy"][0]
        print(sigmaxx)

        # print(np.sum(self.sigma))

        gridsize = const["surface"]["gridSize"][0]
        # self.sigma = self.sigma.reshape((gridsize, gridsize))

    def main(self):
        # self.theta  = self.theta_calc(self.R, self.Rabs)
        self.theta0 = self.theta0_calc(self.R, self.n, self.Rabs, self.Nabs)
        self.index = self.mirror_sort(self.r, self.r0, self.n, self.theta0)
        return self.theta0


    def G0(self, xi=0, phi=0, G0=1,):
            # G -- диаграмма направленности
            # theta -- угол падения
            # phi = const["antenna"]["polarAngle"]


            xi = np.deg2rad(xi)
            phi = np.deg2rad(phi)

            X = self.R[0,:] * np.cos(phi) + self.R[1,:] * np.sin(phi)
            Y = self.R[0,:] * np.sin(phi) - self.R[1,:] * np.cos(phi)
            Z = self.z0

            rho = np.sqrt(X**2 + Y**2)
            phi = np.arccos(X/rho)

            Rabs = np.sqrt(X**2 + Y**2 + Z**2)

            theta = (Z*np.cos(xi) + rho*np.sin(xi)*np.cos(phi))
            theta *= 1/Rabs
            theta = np.arccos(theta)



            return G0*np.exp(-2/self.gamma * np.sin(theta)**2)

    def cross_section(theta, sigmaxx, sigmayy):
        F = 0.5
        sigma =  F**2/( 2*np.cos(theta)**4 * np.sqrt(sigmaxx * sigmayy) )
        sigma *= np.exp( np.tan(theta)/(2*sigmaxx) )
        return sigma

    def G(self, theta, G0=1):
            # G -- диаграмма направленности
            # theta -- угол падения
            return G0*np.exp(-2/self.gamma * np.sin(theta)**2)

    def Rabs_calc(self, R):
        R = np.array(R)
        if len(R.shape)>1:
            Rabs = np.sqrt(np.sum(R**2,axis=0))
        else:
            Rabs = np.sqrt(np.sum(R**2))
        return Rabs

    def Nabs_calc(self,n):
        N = np.sqrt(np.sum(n**2,axis=0))
        return N

    def theta_calc(self, R, Rabs):
        theta = R[-1,:]/Rabs
        return np.arccos(theta)

    def theta0_calc(self,R,n,Rabs,Nabs):
        theta0 = np.einsum('ij, ij -> j', R, n)
        theta0 *= 1/Rabs/Nabs
        return np.arccos(theta0)

    def mirror_sort(self, err = 1):
        index = np.where(self.theta0 < np.deg2rad(err))
        return index

    # def sort(self, arr):
    #     arr = arr.flatten()
    #     return arr[self.index]

    def power(self, t, omega , timp,  R,  theta):

        c = self.c
        G = self.G
        tau = R/c
        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]
        theta = theta[index]
        R= R[index]
        tau = tau[index]
        # Путь к поверхности
        #! $\omega\tau\cos(\theta) = kR$!
        E0 = G(theta)/R
        # e0 = np.exp(1j*omega*(t  - tau*np.cos(theta)) )
        e0 = 1
        # Путь от поверхности
        E0 = E0*e0*G(theta)/R
        # e0 = np.exp(1j*omega*(tau + tau*np.cos(theta)) )
        e0 = 1


        E = np.sum(E0*e0)
        return np.abs(E)**2/2

    def power1(self, t,):

        Rabs = self.Rabs
        theta = self.theta
        theta0 = self.theta0

        c = self.c
        timp = self.timp



        # if t >= 0 :
        #     rmax = np.sqrt(t**2 + 2*t*self.z0/c)*c
        # if t <= 0:
        # if t >= timp:
        #     rmin = np.sqrt((t-timp)**2 + 2*(t-timp)*self.z0/c)*c
        # else:
        #     rmin = 0

        # z0 = self.z0
        # Rmax = np.sqrt(z0**2 + rmax**2)
        # Rmin = np.sqrt(z0**2 + rmin**2)

        tau = Rabs/c
        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]
        # Rmin = 0
        # index = [ i for i in range(Rabs.size) if Rmin <= Rabs[i] <= Rmax ]

        theta = theta[index]
        Rabs = Rabs[index]
        theta0 = theta0[index]

        index = np.where(theta0 < np.deg2rad(1))
        theta = theta[index]
        Rabs = Rabs[index]
        # print(Rabs.size)
        # print(index)

        # print(theta.size)

        G = self.G(theta)
        E0 = (G/Rabs)**2
        e0 = 1

        E = np.sum(E0**2*e0)
        # return np.abs(E)**2/2
        return E

    def sort(self, x, y):
        theta0 = self.theta0_calc(self.R, self.n, self.Rabs, self.Nabs)
        index1 = np.where(theta0 < np.deg2rad(1))
        return x[index1], y[index1]

        # for i in range()
        # index2 = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]


if __name__ == "__main__":

    import sys, argparse
    import matplotlib.pyplot as plt
    from json import load
    with open("rc.json", "r") as f:
        const = load(f)

    const["surface"]["x"][0] = 125e3
    xmax = const["surface"]["x"][0]
    gridsize = const["surface"]["gridSize"][0]
    x0 = np.linspace(-xmax, xmax, gridsize)
    y0 = np.linspace(-xmax, xmax, gridsize)

    x,y = np.meshgrid(x0,y0)

    surface = [np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)]



    fig, ax = plt.subplots()
    pulse = Pulse(surface, x, y, const, )
    G = np.zeros(gridsize*gridsize)

    for xi in np.arange(-17, 17, 1.4):
        G += pulse.G0(phi = 90, xi = xi)
    plt.contourf(x,y,G.reshape((gridsize,gridsize)))
    # timp = const["antenna"]["impulseDuration"][0]


    # t = np.linspace(-1*timp, 4*timp, 512)
    # P = np.zeros(t.size)
    # theta = pulse.theta

    # for i in range(t.size):
    #     P[i] = pulse.power1(t[i])
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # pulse = Pulse(surface, r*np.cos(phi), r*np.sin(phi), const, )
    # theta0 = pulse.G(pulse.theta)
    # im = ax.contourf(phi, r, theta0.reshape(x.shape))


    plt.savefig("G0")

    # plt.show()
