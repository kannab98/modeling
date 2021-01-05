from modeling import kernel, surface
from numba import cuda, float32
import math
from cmath import exp, phase
import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp


@cuda.jit
def linear_grid(x, y, jac, k, A):
    i = cuda.grid(1)

    if i >= x.shape[0]:
        return

    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
            kr = k[n,m].real*x[i] + k[n,m].imag*y[i]
            e = A[n,m] * exp(1j*kr) 

            x[i] += -e.imag * k[n,m].real/abs(k[n,m])
            y[i] += -e.imag * k[n,m].imag/abs(k[n,m])

            jac[0, 0, i] -=  e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])
            jac[0, 1, i] -=  e.real * (k[n,m].real * k[n,m].imag)/abs(k[n,m])
            jac[1, 1, i] -=  e.real * (k[n,m].imag * k[n,m].imag)/abs(k[n,m])

    jac[1, 0, i] = jac[0, 1, i]
    


    


def pick_nonzero_row(m, k):
    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k

def inverse(matrix_origin):

    I = np.identity(matrix_origin.shape[0], dtype=float)

    m = matrix_origin
    # mnp.matrix(np.diag([1.0 for i in range(matrix_origin.shape[0])]))))

    # forward trace
    n = matrix_origin.shape[0]
    for k in range(n):
        # 1) Swap k-row with one of the underlying if m[k, k] = 0
        # swap_row = pick_nonzero_row(m, k)
        # print(swap_row)
        # if swap_row != k:
        #     m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        # 2) Make diagonal element equals to 1
        if m[k, k] != 1:
            tmp =  m[k, k]
            for j in range(n):
                I[k, j] *= 1 / tmp
                m[k, j] *= 1 / tmp
            # print(I[k, :])

        # print(I, '\n')
        # 3) Make all underlying elements in column equal to zero
        for row in range(k + 1, n):
            tmp = m[row, k]

            for j in range(n):
                I[row, j] -= I[k, j] * tmp
            for j in range(n):
                m[row, j] -= m[k, j] * tmp


    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                # 1) Make all overlying elements equal to zero in the former identity matrix
                tmp = m[row, k]
                for j in range(n):
                    I[row, j] -= I[k, j] * m[row, k]
                for j in range(n):
                    m[row, j] -= m[k, j] * m[row, k]

    return I

TPB = 16
X = np.array([10.1])
Y = np.array([20.1])


host_constants = surface.export()
cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
threadsperblock = TPB 
blockspergrid = math.ceil(X.size / threadsperblock)




X0 = X.flatten()
Y0 = Y.flatten()

x0 = X.flatten()
y0 = Y.flatten()



for i in range(2):
    jac = np.zeros((2, 2, X.size))
    jac[1,1,:] = 1
    jac[0,0,:] = 1

    # Вычисление ФУНКЦИИ
    Fx = np.zeros(x0.size)
    Fy = np.zeros(x0.size)
    Fx[:] = x0[:]
    Fy[:] = y0[:]
    linear_grid[blockspergrid, threadsperblock](Fx, Fy,  jac, *cuda_constants)

    I = inverse(jac[:,:,0])

    x0 -= (Fx-X0)*I[0, 0] + (Fy-Y0)*I[0, 1]
    y0 -= (Fx-X0)*I[1, 0] + (Fy-Y0)*I[1, 1]
    # print(X, x, x0)

# linear_grid[blockspergrid, threadsperblock](x, y,  jac, *cuda_constants)

linear_grid[blockspergrid, threadsperblock](x0, y0,  jac, *cuda_constants)
print(X0, Fx, x0)
print(Y0, Fy, y0)




# matrix_origin = np.random.rand(2,2)
# print(np.linalg.inv(matrix_origin))
# inv = inverse(matrix_origin)
# print(inv, "\n")