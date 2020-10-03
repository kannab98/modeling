import numpy as np
from tools.rc import Config, Surface, Sattelite

class Pulse():

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + ( np.ones((r.shape[1], 1) ) @ r0 ).T)

    @staticmethod
    def theta(self, R, Rabs):
        theta = R[-1,:]/Rabs
        return np.arccos(theta)
    
    @property
    def distance(self):
        return self._Rabs

    @property
    def coordinates(self):
        return self.position(rc.surface.coordinates, rc.sattelite.coordinates)



    
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



    def __init__(self,  rc):



        self.rc = rc
    
        self._R = self.position(rc.surface.coordinates, rc.sattelite.coordinates)
        self._n = rc.surface.normal

        self._gamma = 2*np.sin(rc.sattelite.gainWidth/2)**2/np.log(2)
        self._G = self.G(self._R, 
                            rc.sattelite.polarAngle, 
                            rc.sattelite.deviation, self._gamma)

        self._nabs = self.abs(self._n)
        self._Rabs = self.abs(self._R)



    
    @property
    def gain(self):
        size = rc.surface.gridSize
        self._G = self.G(self._R, 
                            rc.sattelite.polarAngle, 
                            rc.sattelite.deviation, self._gamma)
        return self._G.reshape((size, size))


rc = Config()
print(rc.surface.abs(rc.surface.coordinates))
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
    

    

    
