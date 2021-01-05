    
import numpy as np
from modeling import rc 
from modeling.surface import *
from modeling import surface 
from modeling import spectrum as spec
import matplotlib.pyplot as plt
from scipy.integrate import quad

    
kernel = kernel_default
arr, X0, Y0 = run_kernels(kernel, surface)
arr = np.array([arr[0],
                arr[0]+arr[1],
                arr[0]+arr[1]+arr[2],
                arr[0]+arr[1]+arr[2]+arr[3]])

# for i in range(len(band)):
    # moments = self.surface._staticMoments(X0[0], Y0[0], arr[i], )
    # sigma[j,i] = moments


# with open("data/ModelMoments.xlsx", "wb") as f:
            # df = df.to_excel(f)

