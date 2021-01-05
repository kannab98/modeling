from modeling.tools import plotSurface
from modeling import rc, kernel, surface, spectrum, cuda
import numpy as np
from multiprocessing import Process, Array
import matplotlib.pyplot as plt
import numba
from modeling.tools import correlate

# logging.basicConfig(filename='example.log', level=logging.INFO)
# spectrum.plot()



# create the logging file handler


# add handler to logger object
# spectrum.plot()
# from ftplib import FTP 
# import os

# ftp = FTP('localhost')
# ftp.login('ftp', '*TeKwT')
# ftp.cwd('WaterSurfaceModeling')


# filename = 'rc.json'
# with open(filename, 'rb') as f:
#     ftp.storlines('STOR %s'  % os.path.basename(filename), f)




# print(surface[1].max(), surface[2].max() )
# print(surface[1].min(), surface[2].min() )

# # plotSurface()




from scipy import signal

# X, Y = surface.meshgrid
# X = X.flatten()
# Y = Y.flatten()
# arr = np.zeros((3, X.size))
# K = np.zeros((10, 2*X.size-1))




# TPB = 16
# import math
# host_constants = surface.export()
# k = numba.cuda.to_device(host_constants[0])
# A = host_constants[1]
# threadsperblock = TPB 
# blockspergrid = math.ceil(X.size / threadsperblock)


# x,y = surface.meshgrid
# srf = kernel.simple_launch(cuda.default)
# Z = srf[0]
# K = signal.correlate2d(Z, Z)

# plt.imshow(K)
# plt.savefig("kek")
# plt.show()

correlate.test()
