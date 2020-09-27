import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from numpy import pi
from surface import Surface
from retracking import Retracking

from json import load, dump
with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

const["surface"]["kSize"][0] = 1
const["surface"]["phiSize"][0] = 1


# U10 = np.linspace(5, 15, 10)

def parmom(surface, a,b,c):
    S = lambda k: surface.spectrum(k)
    Phi =  lambda k,phi: surface.Phi(k, phi)
    k = surface.k0
    phiint = np.zeros(k.size)
    phi = np.linspace(-np.pi, np.pi, 10**3)
    for i in range(k.size):
        phiint[i] = np.trapz( Phi(k[i],phi)*np.cos(phi)**c*np.sin(phi)**b, phi)
    integral =  np.trapz( np.power(k,a+b-c) * S(k) * phiint, k)
    return integral

def mom(surface, n):
    S = lambda k: surface.spectrum(k)
    k = surface.k0
    integral =  np.trapz( np.power(k,n) * S(k), k)
    return integral


# MEDIAN = np.zeros_like(U10)
# MEAN = np.zeros_like(U10)
# swh = np.zeros_like(U10)

# const["wind"]["speed"][0] = U10
surface = Surface(const)
sigma = surface.sigma_sqr
Hs = 4 * np.sqrt(sigma)
x = np.linspace(-Hs,Hs, 10000)
W1 = 1/np.sqrt(2*np.pi*sigma)*np.exp(-x**2/(2*sigma))
Sigma = parmom(surface, 1,1,1)**2 - parmom(surface, 2,0,1)*parmom(surface, 0,2,1)
W = W1*(1 + Sigma/mom(surface, 0) - mom(surface, 1)/mom(surface, 0)*x - Sigma/mom(surface, 0)**2 * x**2)

F = np.zeros(W.size)
F1 = np.zeros(W.size)
for i in range(W.size):
    F[i] = np.trapz(W[0:i], x[0:i])
    F1[i] = np.trapz(W1[0:i], x[0:i])

eps = 1e-3
median = [] 
median1 = [] 
for i in range(F.size):
    if np.abs(F[i] - 0.5) < eps:
        median.append(x[i])

MEDIAN = np.mean(median)
MEAN = -mom(surface, 1)
swh = 4*np.sqrt(surface.sigma_sqr)

plt.figure(figsize=(3.6, 3.6))
plt.plot(x, W1, label="default")
plt.plot(x, W, label="cwm")
plt.legend()

plt.xlabel("z, м")
plt.ylabel("W, плотность вероятности")
plt.savefig("pdf_cwm", bbox_inches="tight")


plt.figure(figsize=(3.6, 3.6))
plt.plot(x, F1, label="default")
plt.plot(x, F, label="cwm")
plt.legend()

plt.xlabel("z, м")
plt.ylabel("F, функция распределения")
plt.savefig("f_cwm", bbox_inches="tight")





# fig, ax = plt.subplots()


# retracking  = Retracking(const)
# dtypes = ["Rostov"]
# for dtype in dtypes:
#     y = retracking.emb(swh=swh, U10=U10, dtype=dtype)
#     ax.plot(swh, y, label=dtype)


# print(MEAN)
# print(MEDIAN)

# ax.plot(swh, MEDIAN, label="IPFRAN 1")
# ax.plot(swh, MEAN, label="IPFRAN 2")
# plt.legend()
# plt.savefig("compare_emb")
# plt.xlabel("U10")
# plt.xlabel("EMB")


# plt.show()
