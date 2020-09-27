
import surface as srf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pulse import Pulse
from surface import Surface
from multiprocessing import Process, Array
from json import load, dump

"""
Для изменения параметров моделирования необходимо изменить файл rc.json

Наиболее часто используемый параметр surface.gridsize - позволяет выбрать
точность моделирования поверхности. 

Размер площадки в данном скрипте выбирается автоматически 
на основе высоты значительного волнения, известной из спектра волнения. 
"""

with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

grid_size = const["surface"]["gridSize"][0] 
c = const["constants"]["lightSpeed"][0]
z0 = const["antenna"]["z"][0]



surface = Surface(const)
host_constants = surface.export()



Hs = 4 * np.sqrt(surface.sigma_sqr)
T0 = (z0 - Hs)/c
tau = 75e-9
print(Hs)
xmax = np.sqrt(4*Hs*z0 + c*tau*z0/2)
print(xmax)
T = np.linspace(T0-Hs/c, np.sqrt(z0**2+xmax**2)/c, 104)


x = np.linspace(-xmax, xmax, grid_size)
y = np.linspace(-xmax, xmax, grid_size)


X, Y = np.meshgrid(x, y)


kernels = [srf.kernel_default, srf.kernel_cwm]



labels = ["default", "cwm"] 

process = [ None for i in range( len(kernels) )]
arr = [ None for i in range( len(kernels) )]
X0 = [ None for i in range( len(kernels) )]
Y0 = [ None for i in range( len(kernels) )]

for j, kernel in enumerate(kernels):
    # Create shared array
    arr_share = Array('d', 3*X.size )
    X_share = Array('d', X.size)
    Y_share = Array('d', Y.size)
    # arr_share and arr share the same memory
    arr[j] = np.frombuffer( arr_share.get_obj() ).reshape((3, X.size)) 
    X0[j] = np.frombuffer( X_share.get_obj() )
    Y0[j] = np.frombuffer( Y_share.get_obj() )
    np.copyto(X0[j], X.flatten())
    np.copyto(Y0[j], Y.flatten())


    process[j] = Process(target = srf.init_kernel, args = (kernel, arr[j], X0[j], Y0[j], host_constants))
    process[j].start()

data_p = pd.Series(dtype='object')
data_m = pd.Series(dtype='object')

fig_p, ax_p = plt.subplots()
fig_m, ax_m = plt.subplots()
for i in range(len(kernels)):
    # wait until process funished
    process[i].join()
    surf = arr[i]
    x = X0[i]
    y = Y0[i]
    pulse = Pulse(surf, x, y, const)
    # Calculate Probality Dencity Function of mirrored dots
    Z, W, f = pulse.pdf(surf)

    P = np.zeros(T.size)
    # Calculate impulse form
    for j in range(T.size):
        P[j] += pulse.power1(T[j])
    
    ax_p.plot(T,P, label=labels[i])
    dT = pd.Series(T - z0/c, name = 't_%s' % (labels[i]))
    dP = pd.Series(P/P.max(), name = 'P_%s' % (labels[i]))
    data_p = pd.concat([data_p, dT, dP], axis=1)

    Z = pd.Series(Z, name = 'Z_%s' % (labels[i]))
    W = pd.Series(W, name = 'W_%s' % (labels[i]))
    f = pd.Series(f, name = 'f_%s' % (labels[i]))

    ax_m.plot(Z,f, label=labels[i])
    data_m = pd.concat([data_m, Z, W, f], axis=1)




now = datetime.datetime.now().strftime("%m%d_%H%M%S")


os.mkdir(str(now))

# Save impulses forms and PDF to png files 
fig_p.savefig("%s/impulses" % now)
fig_m.savefig("%s/pdf" % now)
data_p = pd.DataFrame(data_p)
data_m = pd.DataFrame(data_m)

# Save impulses forms and PDF to csv files 
data_p.to_csv('%s/impulses.csv' % (now), index = False, sep=';')
data_m.to_csv('%s/impulses_pdf.csv' % (now), index = False, sep=';')

# Save 
with open('%s/rc.json' % (now), 'w', encoding="utf-8") as f:
    dump(const, f, indent=4, ensure_ascii=False,)

plt.plot(T, P)
plt.show()


