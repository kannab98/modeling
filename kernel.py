
from numba import  cuda
import numpy as np
import math

TPB=16


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

@cuda.jit
def kernel_polar(ans, x, y, k, phi, A, F, psi):
    i = cuda.grid(1)

    if i >= x.size:
        return

    for n in range(k.size): 
        for m in range(phi.size):
                kr = k[n]*(x[i]*math.cos(phi[m]) + y[i]*math.sin(phi[m]))      
                Af = A[n] * F[n][m]
                Cos =  math.cos(kr + psi[n][m]) * Af
                Sin = - math.sin(kr + psi[n][m]) * Af

                Cosx = k[n] * math.cos(phi[m])
                Siny = k[n] * math.sin(phi[m])


                ans[0,i] +=  Cos 
                ans[1,i] +=  Cos * Cosx   
                ans[2,i] +=  Sin * Siny

                # CWM
                ans[1,i] *= 1 - Cos * Cosx
                ans[2,i] *= 1 - Sin * Siny

                x[i] += Sin * math.cos(phi[m])
                y[i] += Sin * math.sin(phi[m])
