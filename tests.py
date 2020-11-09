from modeling.surface import Surface
from modeling.surface import kernel_default as kernel
from modeling.surface import run_kernels

from modeling.spectrum import Spectrum
import unittest
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import matplotlib


class TestSpectrum(unittest.TestCase, Spectrum):

    def test_init(self):
        spectrum = Spectrum()

    def __init__(self, *args, **kwargs):
        super(TestSpectrum, self).__init__(*args, **kwargs)
        self.spectrum = Spectrum()

    def test_slick(self):

        k = self.spectrum.k0 
        S = self.spectrum.with_slick(k)
        # plt.loglog(k, S)
        self.spectrum.plot(stype="slick")
        self.spectrum.plot(stype="ryabkova")

        df = pd.read_csv("data/RYABK_1N.DAT", sep = "\s+", header=0)
        plt.plot(df["KT"], df["OUR_kt"])


        plt.savefig("kek")



@unittest.skip
class TestMoments(unittest.TestCase, Surface):


    def test_init(self):
        surface = Surface()

    def __init__(self, *args, **kwargs):
        super(TestMoments, self).__init__(*args, **kwargs)
        self.surface = Surface(config="data/TestTheoreticalMoments.json")

    @unittest.skipIf(os.path.isfile("data/TheoreticalMoments.xlsx") == True, "test already complete")
    def test_theoretical(self):
        U = np.linspace(3, 20, 180)
        label = ["var", "xx", "yy", "xx+yy",  "xy"]
        t = ["theory"]
        band = ["C", "X", "Ku", "Ka"]

        columns = pd.MultiIndex.from_arrays([["U"],[""],[""]])
        df = pd.DataFrame(U, columns=columns )
        iterables = [t, band, label]
        columns = pd.MultiIndex.from_product(iterables) 
        df0 = pd.DataFrame(columns=columns, index=df.index)
        df = pd.concat([df, df0], axis=1)
    

        sigma = np.zeros((U.size, len(band), len(label)))

        for j in range(U.size):
            self.surface.spectrum.windSpeed =  U[j]
            print(j)
            for i in range(len(band)):
                moments = self.surface._theoryStaticMoments(band=band[i])
                sigma[j,i] = moments


        for i, b in enumerate(band):
            for j,m in enumerate(label):
                df.loc[:,("theory", b, m) ] = sigma[:,i,j]
        
        with open("data/TheoreticalMoments.xlsx", "wb") as f:
            df = df.to_excel(f)

    @unittest.skipIf(os.path.isfile("data/ModelMoments.xlsx") == True, "test already complete")
    def test_model(self):
        U = np.linspace(3, 20, 180)
        label = ["var", "xx", "yy", "xx+yy",  "xy"]
        t = ["model"]
        band = ["C", "X", "Ku", "Ka"]

        columns = pd.MultiIndex.from_arrays([["U"],[""],[""]])
        df = pd.DataFrame(U, columns=columns )
        iterables = [t, band, label]
        columns = pd.MultiIndex.from_product(iterables) 
        df0 = pd.DataFrame(columns=columns, index=df.index)
        df = pd.concat([df, df0], axis=1)
    

        sigma = np.zeros((U.size, len(band), len(label)))

        for j in range(U.size):
            self.surface.spectrum.windSpeed =  U[j]
            print(j)


            arr, X0, Y0 = run_kernels(kernel, self.surface)
            arr = np.array([arr[0],
                            arr[0]+arr[1],
                            arr[0]+arr[1]+arr[2],
                            arr[0]+arr[1]+arr[2]+arr[3]])

            for i in range(len(band)):
                moments = self.surface._staticMoments(X0[0], Y0[0], arr[i], )
                sigma[j,i] = moments


        for i, b in enumerate(band):
            for j,m in enumerate(label):
                df.loc[:,("model", b, m) ] = sigma[:,i,j]
        
        with open("data/ModelMoments.xlsx", "wb") as f:
            df = df.to_excel(f)




if __name__ == '__main__':
    unittest.main()
