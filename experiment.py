import numpy as np
from tools.rc import Config, Surface, Sattelite

class Experiment(Config):

    def __init__(self, ):

        super().__init__()
    
        self._R = self.position(self.surface.coordinates, self.sattelite.coordinates)
        self._n = self.surface.normal



        self._gamma = 2*np.sin(self.sattelite.gainWidth/2)**2/np.log(2)
        self._G = self.G(self._R, 
                            self.sattelite.polarAngle, 
                            self.sattelite.deviation, self._gamma)

        self._nabs = self.abs(self._n)
        self._Rabs = self.abs(self._R)
        print(self._Rabs)
        self.tau = self._Rabs / self.constants.c
        self.t0 = self._Rabs.min() / self.constants.c

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.sqrt(np.sum(vec**2,axis=0))
        # return np.sqrt(np.diag(vec.T@vec))

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + ( np.ones((r.shape[1], 1) ) @ r0 ).T)


    
    @staticmethod
    def G(r, phi, xi, gamma):
        x = r[0]
        y = r[1]
        z = r[2]


        # Поворот системы координат на угол phi
        X = np.array(x * np.cos(phi) + y * np.sin(phi), dtype=float)
        Y = np.array(x * np.sin(phi) - y * np.cos(phi), dtype=float)
        Z = np.array(z, dtype=float)

        # Проекция вектора (X,Y,Z) на плоскость XY
        rho =  np.sqrt(np.power(X, 2) + np.power(Y, 2))

        # Полярный угол в плоскости XY
        psi = np.arccos(X/rho)


        theta = (Z*np.cos(xi) + rho*np.sin(xi)*np.cos(psi))
        theta *= 1/np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(theta)

        return np.exp(-2/gamma * np.sin(theta)**2)

    @property
    def surface_coordinates(self):
        return self.surface.coordinates

    @surface_coordinates.setter
    def surface_coordinates(self, r):
        for i in range(3):
            np.copyto(self.surface.coordinates[i], r[i])

        self._R = self.position(self.surface.coordinates, self.sattelite.coordinates)
        self._Rabs = self.abs(self._R)

        self._n = self.position(self.surface.coordinates, self.sattelite.coordinates)
        self._Nabs = self.abs(self._n)
        
    @property
    def sattelite_coordinates(self):
        return self.surface.coordinates

    @sattelite_coordinates.setter
    def sattelite_coordinates(self, r):
        for i in range(3):
            np.copyto(self.sattelite.coordinates[i], r[i])

        self._R = self.position(self.surface.coordinates, self.sattelite.coordinates)
        self._Rabs = self.abs(self._R)






    @property
    def incidence(self):
        z = np.array(self._R[-1], dtype=float)
        R = np.array(self._Rabs, dtype=float)
        return np.arccos(z/R)

    @property
    def localIncidence(self):
        R = np.array(self._R/self._Rabs, dtype=float)
        n = np.array(self._n/self._nabs, dtype=float)
        theta0 = np.einsum('ij, ij -> j', R, n)
        return np.arccos(theta0)

    @staticmethod
    def sort(incidence, err = 1):
        index = np.where(incidence < np.deg2rad(err))
        return index

    
    @property
    def gain(self):
        size = self.surface.gridSize
        self._G = self.G(self._R, 
                            self.sattelite.polarAngle, 
                            self.sattelite.deviation, self._gamma)
        return self._G.reshape((size, size))
    
    def power(self, t):
        tau = self._Rabs / self.constants.c
        timp = self.sattelite.impulseDuration
        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]

        theta =self.incidence[index]
        Rabs = self._Rabs[index]
        R = self._R[:,index]
        theta0 = self.localIncidence[index]

        index = self.sort(theta0)
        theta = theta[index]
        Rabs = Rabs[index]
        R = R[:,index]



        G = self.G(R, self.sattelite.polarAngle, self.sattelite.deviation, self._gamma)
        print(index)

        E0 = (G/Rabs)**2
        P = np.sum(E0**2/2)
        return P




# print(rc.surface.abs(rc.surface.coordinates))

ex = Experiment()
timp = 3e-9
t = np.linspace(ex.t0 -1*timp, ex.t0 + timp)
p = np.empty_like(t)


for i in range(t.size):
    p[i] = ex.power(t=t[i])

import matplotlib.pyplot as plt
plt.plot(t,p)
plt.show()

# rc.pickle()
# pulse = Pulse(rc)

# rc = pulse.rc
# rc.surface.gridSize = 252
# print(rc.surface.gridSize) 


# import matplotlib.pyplot as plt


# G = np.zeros(pulse.gain.shape)
# pulse.polarAngle = 90
# x = pulse.x
# y = pulse.y



# for i in range(8):
#     for xi in np.arange(-10,10,3):
#         r0 = pulse.sattelite_position
#         pulse.sattelite_position = np.array([r0[0]+5000,r0[1], r0[2]])
#         pulse.deviation = xi 
#         G  += pulse.gain

# plt.contourf(x,y,G)
# plt.savefig("kek")
    

    

    
