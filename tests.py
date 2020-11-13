from modeling.surface import Surface
from modeling.surface import kernel_cwm as kernel
from modeling.surface import run_kernels

from modeling.spectrum import Spectrum
import unittest
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib



@unittest.skip
class TestSurface(unittest.TestCase, Surface):

    def __init__(self, *args, **kwargs):
        super(TestSurface, self).__init__(*args, **kwargs)
        self.surface = Surface()

    def test_model_moments(self):
        with open("data/Moments.xlsx", "rb") as f:
            self.df = pd.read_excel(f, header=[0,1,2], index_col=[0])
            print(self.df.columns)
            self.U = self.df["U"].values
            self.df = self.df["ryabkova"]

    
    def test_plot_crosssec(self):
        surface = self.surface
        moments = self.df["Ku"]
        print(moments)
        theta = np.linspace(-17, 17, 49)
        sigma = surface.crossSection(theta, moments)
        

@unittest.skip
class TestSpectrum(unittest.TestCase, Surface):

    def test_init(self):
        surface = Surface()

    def __init__(self, *args, **kwargs):
        super(TestSpectrum, self).__init__(*args, **kwargs)
        self.surface = Surface()

    def test_theoretical(self):
        U = np.linspace(5, 10, 2)
        label = ["var", "xx", "yy", "xx+yy"]
        t = ["ryabkova"]
        band = ["Ku"]
        validate = [[0.02102094, 0.013770, 0.008372, 0.022142],
                    [0.33895775, 0.020408, 0.013256, 0.033664]
                    ]
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
                self.surface.spectrum.band = band[i]
                moments = self.surface._theoryStaticMoments(band=band[i])
                sigma[j,i] = moments
                # print("Я ", np.round(moments, 6) )
                # print("Караев ", np.round(validate[j], 6) )


        # for i, b in enumerate(band):
        #     for j,m in enumerate(label):
        #         df.loc[:,("ryabkova", b, m) ] = sigma[:,i,j]

    
        # with open("data/Moments.xlsx", "wb") as f:
        #     df = df.to_excel(f)

        # k = self.spectrum.k0 
        # S = self.spectrum.get_spectrum()
        # plt.loglog(k, S)
        # self.spectrum.plot(stype="slick")
        # self.spectrum.plot(stype="ryabkova")


        # S = self.spectrum.get_spectrum("slick")
        # k = self.spectrum.k0
        # plt.loglog(k, S(k)*k**0)

        # S = self.spectrum.get_spectrum("ryabkova")
        # k = self.spectrum.k0
        # plt.loglog(k, S(k)*k**0)
        # # plt.ylim([1e-5, 1e-2])

        # # df = pd.read_csv("data/RYABK_1N.DAT", sep = "\s+", header=0)
        # # plt.plot(df["KT"], df["OUR_kt"], label="slick")
        # # plt.legend()


        # plt.savefig("kek")



# @unittest.skip
class TestMoments(unittest.TestCase, Surface):


    def test_init(self):
        surface = Surface()

    def __init__(self, *args, **kwargs):
        super(TestMoments, self).__init__(*args, **kwargs)
        self.surface = Surface()
    
    @unittest.skip
    def test_cwm(self):
        arr, X0, Y0 = run_kernels(kernel, self.surface)
        arr = np.array([arr[0],
                        arr[0]+arr[1],
                        arr[0]+arr[1]+arr[2],
                        arr[0]+arr[1]+arr[2]+arr[3]])

    
    

    @unittest.skipIf(os.path.isfile("data/TheoreticalMoments.xlsx") == True, "test already complete")
    def test_theoretical(self):
        U = np.linspace(5, 10, 2)
        label = ["var", "xx", "yy"]
        t = ["theory"]
        band = ["C", "X", "Ku", "Ka"]

        validate = [[0.02102094, 0.013770, 0.008372, 0.022142],
                    [0.33895775, 0.020408, 0.013256, 0.033664]
                    ]

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
        
        # with open("data/TheoreticalMoments.xlsx", "wb") as f:
            # df = df.to_excel(f)

    def test_mean(self):
        U = np.linspace(5, 10, 2)
        for i in range(U.size):
            self.surface.spectrum.windSpeed =  U[i]
            print(self.surface.spectrum.quad(1))

    # @unittest.skipIf(os.path.isfile("data/ModelMoments.xlsx") == True, "test already complete")
    def test_model(self):
        U = np.linspace(5, 10, 2)
        label = ["mean", "var", "xx", "yy", "xx+yy"]
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
        
        print(df)
        # with open("data/ModelMoments.xlsx", "wb") as f:
            # df = df.to_excel(f)






if __name__ == '__main__':
    unittest.main()
