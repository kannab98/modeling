import numpy as np
import os.path
import json
import pandas as pd
import logging
import modeling.cuda
from . import rc
from . import spectrum
from . import surface

import datetime
import math

logger = logging.getLogger(__name__)



try:
    from numba import cuda, float32
    from multiprocessing import Process, Array
    GPU = True
except:
    print("CUDA not installed")
    GPU = False 

band = np.array(["C", "X", "Ku", "Ka"])

TPB=16

def init(kernel, arr, X, Y, host_constants):
    cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
    threadsperblock = TPB 
    blockspergrid = math.ceil(X.size / threadsperblock)

    X0 = X
    Y0 = Y
    if kernel.__dict__['py_func'].__name__ == "cwm":

        logger.debug("Irregular meshgrid. Initializing regularization")
        X0 = X0.flatten()
        Y0 = Y0.flatten()
        modeling.cuda.linearized_cwm_grid[blockspergrid, threadsperblock](X0, Y0, *cuda_constants)



    if (X.flatten().all() == X0.all()):

        logger.debug("Use regular meshgrid")
        X0 = cuda.to_device(X0)
        Y0 = cuda.to_device(Y0)

    else:
        logger.warning("Can't complete regularization. Use irregular meshgrid")

    kernel[blockspergrid, threadsperblock](arr, X0, Y0, *cuda_constants)




    return arr, X, Y

def simple_launch(kernel):
    logger.info("Use %s kernel" % kernel.__dict__['py_func'].__name__)

    model_coeffs = surface.export()

    X, Y = surface.meshgrid
    X = X.flatten()
    Y = Y.flatten()
    arr = np.zeros((3, X.size))

    arr, X0, Y0 = init(kernel, arr, X, Y, model_coeffs)
    X0 = np.array([X0])
    Y0 = np.array([Y0])

    size = rc.surface.gridSize

    var0 = spectrum.quad(0)
    var0_s = spectrum.quad(2)

    var = np.var(arr[0])
    var_s = np.var(arr[1]+arr[2])

    logger.info('Practice variance of heights sigma^2_h=%.6f' % var)
    logger.info('Practice variance of heights sigma^2=%.6f' % var_s)

    if 0.5*var0 <= var <= 1.5*var0:
        logger.info("Practice variance of heights sigma^2_h matches with theory")
    else:
        logger.warning("Practice variance of heights sigma^2_h not matches with theory")

    if 0.5*var0_s <= var_s <= 1.5*var0_s:
        logger.info("Practice variance of full slopes sigma^2 matches with theory")
    else:
        logger.warning("Practice variance of full slopes sigma^2 not matches with theory")

    return arr.reshape(3, *size)


# def cache_file(kernel):
#     srf = rc.surface
#     wind = rc.wind
#     return "%s_%s_%s_%s_%s_%s_%s" % (kernel.__name__, srf.band, srf.gridSize, wind.speed, wind.direction)






def __kernel_per_band__(kernel, X, Y, model_coeffs):


    k = np.abs(model_coeffs[0][:, 0])
    N = spectrum.KT.size - 1

    process = [ None for i in range(N)]
    arr = [ None for i in range(N)]
    X0 = [ None for i in range(N)]
    Y0 = [ None for i in range(N)]


    edge = spectrum.KT
    edge = [ np.max(np.where(k <= edge[i] )) for i in range(1, edge.size)]
    edge = [0] + edge

    def __multiple_kernels_(kernel, X, Y, model_coeffs):

        for j in range(N):
            # Срез массивов коэффициентов по оси K, условие !=1 для phi (он не зависит от band)
            host_constants = tuple([ model_coeffs[i][edge[j]:edge[j+1]]  for i in range(len(model_coeffs))])
            # Create shared array
            arr_share = Array('d', 3*X.size )
            X_share = Array('d', X.size)
            Y_share = Array('d', Y.size)
            # arr_share and arr share the same memory
            arr[j] = np.frombuffer( arr_share.get_obj() ).reshape((3, X.size)) 
            X0 = np.frombuffer( X_share.get_obj() )
            Y0 = np.frombuffer( Y_share.get_obj() )

            np.copyto(X0, X.flatten())
            np.copyto(Y0, Y.flatten())

            process[j] = Process(target = init, args = (kernel, arr[j], X0, Y0, host_constants) )
            process[j].start()

        for j in range(N):
            process[j].join()

        return arr, X0, Y0
            
    arr, X0, Y0 = __multiple_kernels_(kernel, X, Y, model_coeffs)
        
    return arr, X0, Y0


def launch(kernel):

    print(kernel.__dict__['py_func'].__name__)

    model_coeffs = surface.export()

    X, Y = surface.meshgrid
    arr, X0, Y0 = __kernel_per_band__(kernel, X, Y, model_coeffs)
    X0 = np.array([X0])
    Y0 = np.array([Y0])


    size = (rc.surface.gridSize, rc.surface.gridSize)

    return arr, X0.reshape(size), Y0.reshape(size)


def only_last_band(arr):
    ind = np.where(band == rc.surface.band)[0][0]
    return np.sum(arr[:ind], axis=0).reshape((3, rc.surface.gridSize, rc.surface.gridSize))

def reshape_meshgrid(X0, Y0):
    X0 = np.array(X0)
    Y0 = np.array(Y0)
    X0 = X0.reshape((3, rc.surface.gridSize, rc.surface.gridSize))
    Y0 = Y0.reshape((3, rc.surface.gridSize, rc.surface.gridSize))
    return X0, Y0
    
def dump(arr0, X0, Y0):


    ind = len(arr0)
    band = ["C", "X", "Ku", "Ka"]
    band = band[:ind]
    size = rc.surface.gridSize
    arr = np.zeros((ind, 3, size*size))
    arr[0] = arr0[0] 
    for i in range(1, ind):
        arr[i] = np.sum(arr0[:ind], axis=0)

    arr = np.array(arr)

    data = np.array([arr, X0, Y0], dtype=object)
    name = ["surface", "rc"]
    ext = [".pkl", ".json"]


    field = ["heights", "sigmaxx", "sigmayy"]

    iterables = [band, field]

    columns = pd.MultiIndex.from_arrays([["x"],[""],[""]])
    df = pd.DataFrame({"x": X0.flatten(), "y": Y0.flatten()} )
    columns = pd.MultiIndex.from_product(iterables) 
    df0 = pd.DataFrame(columns=columns, index=df.index)
    df = pd.concat([df, df0], axis=1)

    # label = ["$Z$, м", "$arctan(\\frac {\\partial Z}{\\partial X}), град$", "$arctan(\\frac {\\partial Z}{\\partial Y}), град$"]

    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    file = [''.join(name[i]+suffix+ext[i]) for i in range(len(name)) ]

    for i, b in enumerate(band):
        for j, m in enumerate(field):
            df[(b, m)] = arr[i][j].flatten()
    
    df.to_pickle(file[0])

    rc.surface.varPreCalc = np.sum((surface.ampld(surface.k, surface.phi)**2/2))
    rc.surface.varReal = np.var(arr[-1][0])
    # rc.surface.varTheory = spectrum.dblquad(0,0,0)

    rc.surface.covReal = np.cov(arr[-1][1],arr[-1][2]).tolist()
    # rc.surface.covTheory = spectrum.cov().tolist()
    rc.surface.kEdge = spectrum.KT.tolist()

    with open(file[1], 'w') as f:
        dt = {}
        for Key, Value in vars(rc).items():
            if type(Value) ==  type(rc.surface):
                dt.update({Key: {}})
                for key, value in Value.__dict__.items():
                    if key[0] != "_":
                        dt[Key].update({key: value})

        json.dump(dt, f, indent=4)


