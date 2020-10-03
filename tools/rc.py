from json import load, dump
import numpy as np
import pickle

class Config():
    def __init__(self):
        with open("rc.json", "r", encoding='utf-8') as f:
            self._rc = load(f)

        for Key, Value in self._rc.items():
            for key, value in Value.items():
                self._rc[Key][key] = value[0]

        self._surface = Surface(self._rc)
        self._sattelite = Sattelite(self._rc)
        self._constants = Constants()
        

    @property
    def surface(self):
        return self._surface


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
        return np.array(r + ( np.ones((r.shape[1], 1) ) @ r0 ).T)


    def __init__(self, rc, ):
        self._rc = rc["surface"]



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
    
        




    @property
    def gridSize(self):
        return self._rc["gridSize"]

    @gridSize.setter
    def gridSize(self, value):
        self._rc["gridSize"] = value

    @property
    def x(self):
        return self._rc["x"]

    @property
    def y(self):
        return self._rc["y"]

    @property
    def randomPhases(self):
        return self._rc["randomPhases"]

    @property
    def kSize(self):
        return self._rc["kSize"]

    @property
    def nonDimWindFetch(self):
        return self._rc["nonDimWindFetch"]
    
    @property
    def grid(self):
        return self._r[0:2]

    @grid.setter
    def grid(self, rho: tuple):
        for i in range(2):
            np.copyto(self._r[i], rho[i])

    @property
    def gridx(self):
        return self._r[0] 

    @property
    def gridy(self):
        return self._r[1] 

    
    @property
    def heights(self):
        return self._r[2]
    
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
    
    @property
    def distance_to(self, ):
        return self._R

    @distance_to.setter
    def distance_to(self, obj):
        self._R = self.position(self.coordinates, obj.coordinates)



    


    
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
        self._xi= np.deg2rad(xi)

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
    def __init__(self):
        self.c = 299792458
        self.R = 6370e3
        self.g = 9.81

config = Config()