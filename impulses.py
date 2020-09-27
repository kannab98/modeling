import pandas as pd 
import numpy as np
import os
from retracking import Retracking
import matplotlib.pyplot as plt
import re
"""
Чтение из файла формы импульса и проведение процедуры ретрекинга 
"""




impulses = []
dates = re.compile('0917_16*')
for root, dirs, files in os.walk("./"):
  for dir in dirs :
    if dates.match(dir):
        impulses.append('%s/impulses' % (dir))


from json import load
with open("rc.json", "r") as f:
    const = load(f)

retracking = Retracking(const)
# fig, ax = plt.subplots()


for file in impulses:
    labels = ["default", "cwm"]
    df = pd.read_csv(file + '.csv' , sep=';',header=0).T
    t1 = df.iloc[1].values
    p1 = df.iloc[2].values

    t1 = t1[~np.isnan(t1)] 
    p1 = p1[~np.isnan(p1)]
    popt1,func1 = retracking.trailing_edge(t1,p1)
    # ax.plot(t1, func1(t1, *popt1))


    t2 = df.iloc[3].values
    p2 = df.iloc[4].values

    popt2,func2 = retracking.trailing_edge(t2,p2)
    # ax.plot(t2, func1(t2, *popt2))
    # print(popt1[-2], popt2[-2])
    print("высота", retracking.height(popt1[2]) - retracking.height(popt2[2]))
    print("swh", retracking.swh(popt1[-2]) - retracking.swh(popt2[-2]))


    fig, ax = plt.subplots()
    ax.plot(t1,p1, label = labels[0])
    ax.plot(t2,p2, label = labels[1])

    ax.set_xlabel('t')
    ax.set_ylabel('P')
    plt.legend()
    plt.savefig(file + '.png')





    fig, ax = plt.subplots()
    df = pd.read_csv(file + '_pdf.csv' , sep=';',header=0).T
    z1 = df.iloc[1].values
    f1 = df.iloc[3].values
    z2 = df.iloc[4].values
    f2 = df.iloc[6].values
    ax.plot(z1,f1, label = labels[0])
    ax.plot(z2,f2, label = labels[1])
    plt.legend()
    ax.set_xlabel('z')
    ax.set_ylabel('F, функция распределения')

    plt.savefig(file + '_pdf.png')