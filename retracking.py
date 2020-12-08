import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from os import path

from modeling import rc
from modeling.retracking import Retracking

rc.constants.lightSpeed = 1500
rc.antenna.impulseDuration = 40e-6




retracking = Retracking()
df0, df = retracking.from_file(path.join("impulses", ".*.txt"))
print(df, "\n",  df0)

          # df[f, "t"] = sr.iloc[:, 0]
            # df[f, "P"] = sr.iloc[:, 1]