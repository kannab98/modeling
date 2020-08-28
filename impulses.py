import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
date = "0827_0023"
file = '%s/impulse15_%s' % (date, date)

for i in range(1):
    fig, ax = plt.subplots()
    labels = ["default", "cwm"]
    df = pd.read_csv(file + '.csv' , sep=';',header=0).T
    t1 = df.iloc[1].values
    p1 = df.iloc[2].values
    t2 = df.iloc[3].values
    p2 = df.iloc[4].values
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

# plt.show()
