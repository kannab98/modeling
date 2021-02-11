from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rc.surface.x = [-25, 25]
rc.surface.y = [-25, 25]
rc.surface.gridSize = [4, 4]
rc.surface.phiSize = 128
rc.surface.kSize = 1024
print(rc.surface.band)

srf = kernel.launch(cuda.default)

srf0 = srf[0]+srf[1]+srf[2]
srf1 = srf[0]+srf[1]+srf[2]+srf[3]
print(srf1[1])
print(srf0[1])
# srf0 = kernel.convert_to_band(srf, 'Ka')
# srf1 = kernel.convert_to_band(srf, 'Ku')

print(srf0.all() == srf1.all())