from kernel import *
from surface import Surface
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse

from json import load, dump
for U10 in [5,10,15]:
    with open("rc.json", "r", encoding='utf-8') as f:
        const = load(f)

    xmax = 1000
    # xmax = 100
    grid_size = const["surface.gridSize"][0] 
    # grid_size = 200
    c = const["light.speed"][0]
    z0 = const["antenna.z"][0]
    const["wind.speed"][0]=U10



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


    threadsperblock = TPB 
    blockspergrid = math.ceil(x0.size / threadsperblock)
    kernels = [kernel_default, kernel_cwm]
    labels = ["default", "cwm"] 


    data_p = pd.Series(dtype='object')
    data_s = {}
    data_m = pd.Series(dtype='object')

    mn = []
    disp = []

    T0 = (z0 - 10)/c

    for j, kernel in enumerate(kernels):

        T = np.linspace(T0, np.sqrt((z0+10)**2+xmax**2)/c, 512)
        P = np.zeros(T.size)

        surf = np.zeros((3, x0.size))
        kernel[blockspergrid, threadsperblock, stream](surf, x0, y0, k, phi, A, F, psi)
        mn.append(weighted_mean(x0, y0, surf))
        disp.append(weighted_mean(x0, y0, surf, p=2) - mn[-1])
        Hs = 4*np.sqrt(disp[-1])

        pulse = Pulse(surf, x0, y0, const)

        index = pulse.mirror_sort()
        z = surf[0][index].flatten()
        Z, W, f = pdf(z)


        for i in range(T.size):
            P[i] = pulse.power1(T[i])


        dT = pd.Series(T - T0, name = 't_%s' % (labels[j]))
        dP = pd.Series(P/P.max(), name = 'P_%s' % (labels[j]))
        data_p = pd.concat([data_p, dT, dP], axis=1)

        ax_cdf.plot(Z, f)
        ax_pdf.plot(Z, W)
        ax_pulse.plot(T-T0, P/P.max())

        data_s.update({ 'x_%s' % (labels[j]): x0})
        data_s.update({ 'y_%s' % (labels[j]): y0})
        data_s.update({ 'z_%s' % (labels[j]): surf[0]})
        data_s.update({ 'zxx_%s' % (labels[j]): surf[1]})
        data_s.update({ 'zyy_%s' % (labels[j]): surf[2]})

        Z = pd.Series(Z, name = 'Z_%s' % (labels[j]))
        W = pd.Series(W, name = 'W_%s' % (labels[j]))
        f = pd.Series(f, name = 'f_%s' % (labels[j]))
        data_m = pd.concat([data_m, Z, W, f], axis=1)



    now = datetime.datetime.now().strftime("%m%d_%H%M")
    os.mkdir(str(now))
    data_p = pd.DataFrame(data_p)
    data_s= pd.DataFrame(data_s)
    data_m = pd.DataFrame(data_m)

    data_p.to_csv('%s/impulse%d_%s.csv' % (now, const["wind.speed"][0], now), index = False, sep=';')
    data_m.to_csv('%s/impulse%d_%s_pdf.csv' % (now, const["wind.speed"][0], now), index = False, sep=';')

    with  open('%s/impulse%d_%s_data.csv' 
                % (now, const["wind.speed"][0], now), 'a') as f:
            
        for i in range(len(mn)):
            f.write('#mean_%s = %.8f\n' % (labels[i], mn[i]))
            f.write('#dispersion heights_%s = %.8f\n' % (labels[i], disp[i]))

        data_s.to_csv(f, index = False, sep=';')
        f.close()

    with open('%s/rc.json' % (now), 'w', encoding="utf-8") as f:
        dump(const, f, indent=4)




    plt.show()
