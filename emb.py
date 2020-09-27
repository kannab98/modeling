from surface import Surface
from retracking import Retracking
import matplotlib.pyplot as plt
import numpy as np

from json import load
with open("rc.json","r") as f:
    const = load(f)

U10 = np.linspace(5, 15, 10)
swh = np.zeros_like(U10)
for i in range(U10.size):
    const["wind"]["speed"][0] = U10[i]
    surface = Surface(const)
    retracking  = Retracking(const)
    swh[i] = 4*np.sqrt(surface.sigma_sqr)

dtypes = ["Rostov", "Chelton", "Ray"]
for dtype in dtypes:
    print(dtype)
    y = retracking.emb(swh=swh, U10=U10, dtype=dtype)
    plt.plot(U10, y)

