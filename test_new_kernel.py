from kernel import *
from surface import Surface
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse

from json import load, dump
with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

xmax = 1000
# xmax = 100
grid_size = const["surface.gridSize"][0] 
# grid_size = 200
c = const["light.speed"][0]
z0 = const["antenna.z"][0]



def pdf(z):
    N = []
    Z = []

    while z.size > 0:
        index = np.where( np.abs(z - z[0])  <= 0.01 )[0]
        Z.append(z[0])
        z = np.delete(z, index)
        N.append(index.size)

    Z = np.array(Z)
    N = np.array(N)
    N = N[Z.argsort()]
    Z = np.sort(Z)

    Nint = np.zeros(Z.size)
    for i in range(Z.size):
        Nint[i] = np.trapz(N[0:i], Z[0:i])
    
    norm = np.trapz(N, Z)
    return Z, N/norm, Nint/norm


def integrate(x, y, i, j):
    dx = x[j] - x[i]
    if np.abs(dx) < np.abs(0.01*x[j]):
        integral = 0
    else:
        integral = (y[i] + y[j] )/2 * dx
    return integral


def weighted_mean(x0, y0, band2, p=1):
    x0 = x0.reshape((grid_size, grid_size))
    y0 = y0.reshape((grid_size, grid_size))
    z0 = band2[0].reshape((grid_size,grid_size))

    S = np.zeros((grid_size-1, grid_size-1))
    Z = np.zeros((grid_size-1, grid_size-1))

    for m in range(grid_size-1):
        for n in range(grid_size-1):
            x = np.array([x0[i,j] for i in range(m,2+m) for j in range(n,n+2)])
            y = np.array([y0[i,j] for i in range(m,2+m) for j in range(n,n+2)])
            z = np.array([z0[i,j] for i in range(m,2+m) for j in range(n,n+2)])

            walk = [0,2,3,1]

            x = x[walk]
            y = y[walk]
            z = z[walk]

            s = lambda i,j: integrate(x, y, i, j)
            Z[m,n] = np.mean(z)
            S[m,n] = np.abs(+ s(0,1) + s(1,2) + s(2,3)  + s(3,1))

    return (np.sum(S*Z**p)/np.sum(S))


fig_pulse, ax_pulse = plt.subplots() 
fig_pdf, ax_pdf = plt.subplots() 
fig_cdf, ax_cdf = plt.subplots() 



stream = cuda.stream()
surface = Surface(const)
host_constants = surface.export()
cuda_constants = (cuda.to_device(host_constants[i], stream = stream) for i in range(len(host_constants)))
k, phi, A, F, psi = (cuda.to_device(host_constants[i], stream = stream) for i in range(len(host_constants)))

x0 = np.linspace(-xmax, xmax, grid_size)
y0 = np.linspace(-xmax, xmax, grid_size)
x0, y0 = np.meshgrid(x0,y0)
x0 = x0.flatten()
y0 = y0.flatten()


threadsperblock = (TPB, TPB, TPB)
blockspergrid_x = math.ceil(x0.size / threadsperblock)
kernels = [kernel_default, kernel_test]
labels = ["default", "cwm"] 
T0 = (z0 - 5)/c

for j, kernel in enumerate(kernels):

    surf = np.zeros((3, x0.size))
    kernel[blockspergrid, threadsperblock, stream](surf, x0, y0, k, phi, A, F, psi)
