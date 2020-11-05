from json import load,  dump
import numpy as np
import os


class Config():
    def __init__(self):


        with open(os.path.join(os.path.abspath(os.getcwd()), "rc.json"), "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        self._surface = Surface(self._rc)
        self._sattelite = Sattelite(self._rc)
        self._constants = Constants(self._rc)
        self._swell = Swell(self._rc)
        self._windWaves = Wind(self._rc)

    @property
    def surface(self):
        return self._surface

    @property
    def swell(self):
        return self._swell

    @property
    def windWaves(self):
        return self._windWaves

    @property
    def sattelite(self):
        return self._sattelite

    @property
    def constants(self):
        return self._constants

    def export(self):
        with open('rc.obj', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def init_from():
        with open('rc.obj', 'rb') as f:
            return pickle.load(f)


class Surface():

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + (np.ones((r.shape[1], 1)) @ r0).T)

    def __init__(self, rc="rc.json"):

        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["surface"])
            





    # @property
    # def gridSize(self):
    #     return self._rc["gridSize"]

    # @gridSize.setter
    # def gridSize(self, value):
    #     self._rc["gridSize"] = value

    # @property
    # def x(self):
    #     return self._rc["x"]

    # @property
    # def y(self):
    #     return self._rc["y"]

    # @property
    # def randomPhases(self):
    #     return self._rc["randomPhases"]

    # @property
    # def phiSize(self):
    #     return self._rc["phiSize"]

    # @property
    # def kSize(self):
    #     return self._rc["kSize"]

    # @property
    # def kEdge(self):
    #     return self._rc["kEdge"]

    # @property
    # def nonDimWindFetch(self):
    #     return self._rc["nonDimWindFetch"]

    # @property
    # def grid(self):
    #     return self._r[0:2]

    # @grid.setter
    # def grid(self, rho: tuple):
    #     for i in range(2):
    #         np.copyto(self._r[i], rho[i])

    # @property
    # def gridx(self):
    #     return self._r[0].reshape((self.gridSize, self.gridSize))

    # @property
    # def gridy(self):
    #     return self._r[1].reshape((self.gridSize, self.gridSize))

    # @property
    # def heights(self):
    #     return self._r[2].reshape((self.gridSize, self.gridSize))

    # @heights.setter
    # def heights(self, z):
    #     np.copyto(self._r[2], z)

    # @property
    # def coordinates(self):
    #     return self._r

    # @coordinates.setter
    # def coordinates(self, r):
    #     for i in range(3):
    #         np.copyto(self._r[i], r[i])

    # @property
    # def normal(self):
    #     return self._n

    # @normal.setter
    # def normal(self, n):
    #     for i in range(3):
    #         np.copyto(self._n[i], n[i])

    # @property
    # def amplitudes(self):
    #     return self._A

    # @property
    # def angleDistribution(self):
    #     return self._F

    # @property
    # def phases(self):
    #     return self._Psi

    # @angleDistribution.setter
    # def angleDistribution(self, F):
    #     self._F = F

    # @amplitudes.setter
    # def amplitudes(self, A):
    #     self._A = A

    # @phases.setter
    # def phases(self, Psi):
    #     self._Psi = Psi

    # @property
    # def distance_to(self, ):
    #     return self._R

    # @distance_to.setter
    # def distance_to(self, obj):
    #     self._R = self.position(self.coordinates, obj.coordinates)


class Sattelite():
    def __init__(self, rc):
        self._rc = rc["antenna"]

    @property
    def x(self):
        return self._rc["x"]

    @property
    def y(self):
        return self._rc["y"]

    @property
    def z(self):
        return self._rc["z"]

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @property
    def deviation(self):
        return self._rc["deviation"]

    @deviation.setter
    def deviation(self, xi):
        self._xi = np.deg2rad(xi)

    @property
    def polarAngle(self):
        return self._rc["polarAngle"]

    @polarAngle.setter
    def polarAngle(self, phi):
        self._rc["polarAngle"] = np.deg2rad(phi)

    @property
    def impulseDuration(self):
        return self._rc["impulseDuration"]

    @property
    def band(self):
        return self._rc["band"]

    @property
    def gainWidth(self):
        return np.deg2rad(self._rc["gainWidth"])


class Constants:

    def __init__(self, rc="rc.json"):

        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["constants"])
        self.c = 299792458
        self.R = 6370e3
        self.g = 9.81

    # @property
    # def gravityAcceleration(self):
    #     return self._rc["gravityAcceleration"]

    # @property
    # def earthRadius(self):
    #     return self._rc["earthRadius"]

    # @property
    # def lightSpeed(self):
    #     return self._rc["lightSpeed"]


class Swell:
    def __init__(self, rc="rc.json"):

        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["swell"])

    # @property
    # def enable(self):
    #     return self._rc["enable"]

    # @property
    # def direction(self):
    #     return self._rc["direction"]

    # @property
    # def speed(self):
    #     return self._rc["speed"]


class Wind:
    def __init__(self, rc="rc.json"):

        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        vars(self).update(self._rc["wind"])


    # @property
    # def enable(self):
    #     return self._rc["enable"]

    # @property
    # def direction(self):
    #     return self._rc["direction"]

    # @property
    # def speed(self):
    #     return self._rc["speed"]
config = Config()