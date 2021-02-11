import pandas as pd
import numpy as np
from modeling import rc 
from modeling import surface, kernel, cuda
from modeling import spectrum 
import matplotlib.pyplot as plt
from scipy.integrate import quad

U = np.linspace(5, 15, 5)
X,Y = surface.meshgrid

for j in range(U.size):
    rc.wind.speed = U[j]
    arr = kernel.launch(cuda.default)
    arrKa = kernel.convert_to_band(arr, 'Ka')
    arrKu = kernel.convert_to_band(arr, 'Ku')

    arr0 = kernel.launch(cuda.cwm)
    arr0Ka = kernel.convert_to_band(arr, 'Ka')
    arr0Ku = kernel.convert_to_band(arr, 'Ku')

    moments = surface.staticMoments(X,Y, arr)
    moments0 = surface.staticMoments(X,Y, arr)

with open("data/ModelMoments.xlsx", "wb") as f:
            df = df.to_excel(f)

