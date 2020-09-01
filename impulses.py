import pandas as pd 
import numpy as np
from retracking import Retracking
import matplotlib.pyplot as plt
date = "0901_0112"

file = 'impulse5_%s' % (date)

from json import load
with open("rc.json", "r") as f:
    const = load(f)

retracking = Retracking(const)
fig, ax = plt.subplots()

labels = ["default", "cwm"]
df = pd.read_csv(file + '.csv' , sep=';',header=0).T
t1 = df.iloc[1].values
p1 = df.iloc[2].values

t1 = t1[~np.isnan(t1)] 
p1 = p1[~np.isnan(p1)]
popt1,func1 = retracking.trailing_edge(t1,p1)
ax.plot(t1, func1(t1, *popt1))


t2 = df.iloc[3].values
p2 = df.iloc[4].values

popt2,func2 = retracking.trailing_edge(t2,p2)
ax.plot(t2, func1(t2, *popt2))
# print(popt1[-2], popt2[-2])
# print(retracking.height(popt1[-2])-retracking.height(popt2[-2]))
print(popt1[2]*retracking.c - popt2[2]*retracking.c)
print(popt1[2]*retracking.c, popt2[2]*retracking.c)

fig, ax = plt.subplots()
ax.plot(t1,p1, label = labels[0])
ax.plot(t2,p2, label = labels[1])

print(retracking.emb(5))
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

plt.show()
