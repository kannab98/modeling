import numpy as np
from .rc import Config, Constants
from .surface import Surface
from .surface import kernel_default as kernel_default
import matplotlib.pyplot as plt
import datacontrol

from json import load,  dump

class Experiment(Constants):

    def __init__(self, surface):

        self.surface = surface

        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["antenna"])

        self.constants = Constants()
    
        self._R = self.position(self.surface_coordinates, self.sattelite_coordinates)
        self._n = self.surface.normal



        self._gamma = 2*np.sin(self.gainWidth/2)**2/np.log(2)
        self._G = self.G(self._R, 
                            self.polarAngle, 
                            self.deviation, self._gamma)

        self._nabs = self.abs(self._n)
        self._Rabs = self.abs(self._R)
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
        return np.array(self.surface.coordinates, dtype=float)

    @surface_coordinates.setter
    def surface_coordinates(self, r):
        for i in range(3):
            np.copyto(self.surface.coordinates[i], r[i])

        self._R = self.position(self.surface_coordinates, self.sattelite_coordinates)
        self._Rabs = self.abs(self._R)

    @property
    def surface_normal(self):
        return np.array(self.surface.normal, dtype=float)

    @surface_normal.setter
    def surface_normal(self, r):
        for i in range(3):
            np.copyto(self.surface.normal[i], r[i])
        self._nabs = self.abs(self._surface.normal)


        
    @property
    def sattelite_coordinates(self):
        return np.array([self.x, self.y, self.z])

    @sattelite_coordinates.setter
    def sattelite_coordinates(self, r):
        self.x, self.y, self.z = r
        self._R = self.position(self.surface_coordinates, self.sattelite_coordinates)
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

    
    # @property
    # def x(self):
    #     size = self.surface.gridSize
    #     return self._R[0].reshape((size, size))

    # @property
    # def y(self):
    #     size = self.surface.gridSize
    #     return self._R[1].reshape((size, size))
    # @property
    # def z(self):
    #     size = self.surface.gridSize
    #     return self._R[2].reshape((size, size))
    
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

        E0 = (G/Rabs)**2
        P = np.sum(E0**2/2)
        return P




kernels = [kernel_default]
labels = ["default", "cwm"] 

surface = Surface()

ex = Experiment(surface)
z = ex.sattelite_coordinates[-1]

U = surface.windSpeed
g = ex.constants.g



surface.spectrum.nonDimWindFetch = 20170
surface.nonDimWindFetch = 20170
xi = np.arctan(5000/z)
Xi = np.deg2rad(np.linspace(-17, 17, 49))

# rc = surface._rc
# z = rc["antenna"]["z"]
# R = rc["constants"]["earthRadius"]



# plt.plot(np.rad2deg(Xi), np.rad2deg(f(Xi)))
# plt.show()
# sigma0 = np.zeros(Xi.size)

# fig, ax = plt.subplots()
# X = U**2 * 20170 / g 

# for i, xi in enumerate(Xi):
#     arr, X0, Y0 = srf.run_kernels(kernels, surface)
#     sigma0[i] = surface.crossSection(xi)


# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0/sigma0.max())

# kernels = [srf.kernel_default]





# wind = np.linspace(3,15, Xsize)
# fetch = np.linspace(5000, 20170, Xsize)
# direction = np.linspace(-np.pi/2, np.pi/2, 180)
# Xsize = direction.size
# sigma0 = np.zeros((Xi.size, Xsize))

# Xb, Yb, Zb = ex.surface_coordinates
# for i, xi in enumerate(Xi):
#     X = Xb
#     Y = Yb + z*np.tan(xi)
#     for j in range(Xsize):
#         X, Y, Z = ex.surface_coordinates
#         # surface.nonDimWindFetch= fetch[j]
#         # surface.windSpeed = wind[j]
#         surface.direction[0] = direction[j]
#         ex.surface_coordinates = (X,Y,Z)
#         arr, X0, Y0 = srf.run_kernels(kernels, surface)
#         sigma0[i][j] = surface.crossSection(xi)
#         X += 5000




# y = np.array([z*np.tan(xi) for xi in Xi])
# x = np.array([5000*i for i in range(Xsize)])
# x, y = np.meshgrid(x,y)

# import pandas as pd
# df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'sigma': sigma0.flatten()})

# df.to_csv('direction2.tsv' , sep='\t', float_format='%.2f')

# plt.contourf(sigma0, levels=100)
# # plt.savefig("track_fetch")

# plt.savefig("direction2" )

# X1, Y1 = np.meshgrid(X1, Y1)
# plt.contourf(X1, Y1, sigma0.T, levels=100)

# plt.imshow(sigma0)

# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0.max())

# plt.savefig("sigma0")
    # for i in range(len(kernels)):
    #     surface.plot(X0[i], Y0[i], arr[i][0], label="default%s" % (U))


# plt.figure()
# for i in range(t.size):
    # p[i] = ex.power(t=t[i])

# plt.plot(t,p)
# plt.show()

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
    

    

    
