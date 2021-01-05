

from modeling import kernel, surface
from numba import cuda
import math
import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp



TPB = 16
a = 20.1
b = 100
X = np.array([a])
Y = np.array([b])
host_constants = surface.export()
cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
threadsperblock = TPB 
blockspergrid = math.ceil(X.size / threadsperblock)

X0 = X.flatten()
Y0 = Y.flatten()

def grid_cwm(r):
    x0 = np.array([r[0]])
    y0 = np.array([r[1]])

    kernel.cwm_offset[blockspergrid, threadsperblock](x0, y0, *cuda_constants)

    return np.array([x0, y0])

def jac(r):
    ans = np.zeros((2, 2, r[0].size))
    ans[0,0,:] = 1
    ans[1,1,:] = 1
    x0 = np.array([r[0]])
    y0 = np.array([r[1]])

    kernel.cwm_jac[blockspergrid, threadsperblock](ans, x0, y0, *cuda_constants)
    ans[1,0] = ans[0,1]

    return ans[:,:,0]

from scipy.optimize import root

R = np.array([[20.1], [100]])
def func(r):
    f = grid_cwm(r) - R 
    return f.flatten()




# sol = root(func, jac=jac, x0=[1, 1])
# print(sol.x, sol.x.shape)
r = np.array([a, b])
r0 = r
print(r0)

for i in range(10):
    F = func(r0)
    J = np.linalg.inv(jac(r0))
    r0[0] -= F[0]*J[0,0]  + F[1]*J[0,1]
    r0[1] -= F[0]*J[1,0]  + F[1]*J[1,1]

print(r0)
r = grid_cwm(r0)
print(r.flatten())




