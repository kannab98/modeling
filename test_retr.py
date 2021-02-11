from modeling import rc
from modeling.retracking import retracking
import matplotlib.pyplot as plt
import numpy as np
from os import path


rc.constants.lightSpeed = 1500 # м/с
rc.antenna.impulseDuration = 40e-6 # с

df0, df = retracking.from_file(path.join("impulses", ".*.txt"))

print(df)
