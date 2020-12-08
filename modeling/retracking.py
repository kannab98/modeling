import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd

from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas import read_csv
from modeling import rc
from modeling.spectrum import Spectrum
from modeling.surface import Surface
import numpy


class Retracking():
    """
    Самый простой способ использовать этот класс, после вызова его конструктора
    это использовать метод класса from_file. 
    Перед этим имеет смысл указать параметры конкретной системы радиолокации.
    Для класса пока что нужны только два параметра:
        1. Скорость света (звука)
        2. Длительность импульса

    Задать их можно с помощью  объекта rc:
    >>> from modeling import rc
    >>> rc.constants.lightSpeed = 1500 # м/с
    >>> rc.antenna.impulseDuration = 40e-6 # с

    Или же изменить файл rc.json и положить его в рабочую директорию.

    Пример простого использования:
    # Импорт модуля
    >>> from modeling import rc
    >>> from modeling.retracking import Retracking 
    # Конструктор класса
    >>> retracking = Retracking()
    # Ретрекинг для всех файлов, заканчивающихся на .txt в директории impulses
    >>> df0, df = retracking.from_file(path.join("impulses", ".*.txt"))

    

    """
    def __init__(self, **kwargs):
        # Скорость света/звука
        self.c = rc.constants.lightSpeed
        # Длительность импульса в секундах
        self.T = rc.antenna.impulseDuration


    def from_file(self, file):
        """
        Поиск импульсов в файлах по регулярному выражению. 

        Вычисление для всех найденных коэффициентов 
        аппроксимации формулы ICE. 
        
        Оценка SWH и высоты до поверхности.

        Экспорт данных из найденных файлов в output.xlsx в лист raw

        Эспорт обработанных данных в output.xlsx в лист brown

        """
        
        path, file = os.path.split(file)

        path = os.path.abspath(path)
        rx = re.compile(file)


        _files_ = []
        for root, dirs, files in os.walk(path):
            for file in files:
                _files_ += rx.findall(file)

        columns = pd.MultiIndex.from_product([ _files_, ["t", "P"] ], names=["file", "data"])
        df0 = pd.DataFrame(columns=columns)

        df = pd.DataFrame(columns=["SWH", "H", "Amplitude", "Alpha", "Epoch", "Sigma", "Noise"], index=_files_)

        for i, f in enumerate(_files_):
            sr = pd.read_csv(os.path.join(path, f), sep="\s+", comment="#")
            df0[f, "t"] = sr.iloc[:, 0]
            df0[f, "P"] = sr.iloc[:, 1]

            popt = self.pulse(sr.iloc[:, 0].values, sr.iloc[:, 1].values)

            df.iloc[i][2:] = popt
            df.iloc[i][0] = self.swh(df.iloc[i]["Sigma"])
            df.iloc[i][1] = self.height(df.iloc[i]["Epoch"])

        excel_name = "output.xlsx"

        df.to_excel(excel_name, sheet_name='brown')

        with pd.ExcelWriter(excel_name, mode='a') as writer:  
            df0.to_excel(writer, sheet_name='raw')


        return df0, df
        

    @staticmethod
    def leading_edge(t, pulse, dtype="needed"):
        """
        Аппроксимация экспонентой заднего фронта импульса. 
        dtype = "full" -- возвращает все коэффициенты аппроксимации
        dtype = "needed" -- возвращает коэффициенты аппроксимации,
                            необходимые для формулы Брауна

        """
        # Оценили положение максимума импульса
        n = np.argmax(pulse)
        # Обрезали импульс начиная с положения максимума
        pulse = np.log(pulse[n:])
        t = t[n:]
        line = lambda t,alpha,b: -alpha*t + b   
        # Аппроксимация
        popt = curve_fit(line, 
                            xdata=t,
                            ydata=pulse,
                            p0=[1e6,0],
                        )[0]

        if dtype == "full":
            return popt
        elif dtype == "needed":
            return popt[0]

    @staticmethod 
    def trailing_edge(t, pulse):
        """
        Аппроксимация функией ошибок переднего фронта импульса. 

        """

        # Оценили амплитуду импульса
        A0 = (max(pulse) - min(pulse))/2

        # Оценили положение максимума импульса
        n = np.argmax(pulse)

        # Обрезали импульс по максимум

        pulse = pulse[0:n]
        t = t[0:n]


        func = lambda t, A, tau, sigma_l, b:   A * (1 + erf( (t-tau)/sigma_l )) + b

        # Аппроксимация
        popt = curve_fit(func, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A0, (t.max() + t.min())/2, (t[-1]-t[0])/t.size, 0])[0]

                            

        return popt


    
    @staticmethod
    def ice(t, A,alpha,tau,sigma_l,T):
        """
        Точная аппроксимация формулы Брауна.
        В отличии от Брауна не привязяна к абсолютному времени. 
        См. отчет по Ростову за 2020 год

        """
        return A * np.exp( -alpha * (t-tau) ) * (1 + erf( (t-tau)/sigma_l ) ) + T

    def pulse(self, t, pulse):
        alpha = self.leading_edge(t, pulse, dtype="needed")
        A, tau, sigma_l, b = self.trailing_edge(t, pulse)

        popt = curve_fit(self.ice, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A, alpha, tau, sigma_l, b],
                            bounds = [0, np.inf]
                        )[0]
        return popt 
    
    def swh(self, sigma_l):
        """
        Вычисление высоты значительного волнения
        """
        # Скорость света/звука [м/с]
        c = rc.constants.lightSpeed
        # Длительность импульса [с]
        T = rc.antenna.impulseDuration

        sigma_p = 0.425 * T
        sigma_c = sigma_l/np.sqrt(2)
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2))*c/2
        return 4*sigma_s

    def height(self, tau):
        """
        Вычисление высоты от антенны до поверхности воды
        """

        # Скорость света/звука [м/с]
        c = rc.constants.lightSpeed
        return tau*self.c/2

    def emb(self, swh, U10, dtype = "Rostov"):
        """
        Поправка на состояние морской поверхности (ЭМ-смещение)
        """
        if dtype ==  "Rostov":
            emb = swh * (- 0.019 + 0.0027 * swh - 0.0037 * U10 + 0.00014 * U10**2)
            return emb

        elif dtype == "Chelton":
            coeff = np.array([0.0029, -0.0038, 0.000155 ])
            emb = [coeff[i]*U10**i for i in range(coeff.size)]
            EMB = 0
            for i in range(coeff.size):
                EMB += emb[i]
            return  -abs(EMB)


        elif dtype == "Ray":
            coeff = np.array([0.00666,  0.0015])
            emb = [coeff[i]*U10**i for i in range(coeff.size)]
            EMB = 0
            for i in range(coeff.size):
                EMB += emb[i]
            return  -abs(EMB)
        
        return None
    

class Brown():

    def __init__(self):

        theta = np.deg2rad(rc.antenna.gainWidth)
        self.Gamma = self.gamma(theta)

    def t(self):
        T = rc.antenna.impulseDuration
        return np.linspace(-10*T, 25*T, 1000)

    @staticmethod
    def H(h):
        R = rc.constants.earthRadius
        return h * ( 1 + h/R )
    
    @staticmethod
    def A(gamma, A0=1.):
        xi = np.deg2rad(rc.antenna.deviation)
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    @staticmethod
    def u(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = rc.constants.lightSpeed
        return (t - alpha * sigma_c**2 - cwm_mean/c) / (np.sqrt(2) * sigma_c)

    @staticmethod
    def v(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = rc.constants.lightSpeed
        return alpha * (t - alpha/2 * sigma_c**2 - cwm_mean/c)

    @staticmethod
    def alpha(beta,delta):
        return delta - beta**2/4

    def delta(self, gamma):
        c = rc.constants.lightSpeed
        xi = np.deg2rad(rc.antenna.deviation)
        h = rc.antenna.z
        return 4/gamma * c/self.H(h) * np.cos(2 * xi)
    
    @staticmethod
    def gamma(theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self, gamma):
        c = rc.constants.lightSpeed
        xi = np.deg2rad(rc.antenna.deviation)
        h = rc.antenna.z
        return 4/gamma * np.sqrt( c/self.H(h) ) * np.sin( 2*xi )

    @staticmethod
    def sigma_c(sigma_s):
        T = rc.antenna.impulseDuration
        c = rc.constants.lightSpeed
        sigma_p = 0.425 * T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/c)**2 )

    def pulse(self, t, dim = 1, cwm=False):

        self.dim = dim
        gamma = self.Gamma
        delta = self.delta(gamma)
        beta  = self.beta(gamma)

        if dim == 1:
            alpha = self.alpha(beta, delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2), delta)


        spec = Spectrum() 
        surf = Surface()
        sigma_s = spec.quad(0)
        sigma_c = self.sigma_c(sigma_s)

        cwm_mean = 0

        if cwm == True:

            cwm_mean = spec.quad(1)
            sigma_s = spec.quad(0) - cwm_mean
            sigma_c = self.sigma_c(sigma_s)


        u = self.u(t, alpha, sigma_c, cwm_mean=cwm_mean)
        v = self.v(t, alpha, sigma_c, cwm_mean=cwm_mean)

        A = self.A(gamma)
        pulse = A * np.exp(-v) * ( 1 + erf(u) )

        if self.dim == 2:
            alpha = gamma
            u = self.u(t, alpha, sigma_c)
            v = self.v(t, alpha, sigma_c)
            pulse -= A/2 * np.exp(-v) * ( 1 + erf(u) )

        return pulse

