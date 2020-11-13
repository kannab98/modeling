import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas import read_csv
from modeling import rc

class Retracking():
    def __init__(self, **kwargs):

        # Скорость света/звука
        self.c = rc.constants.lightSpeed
        # Длительность импульса в секундах
        self.T = rc.antenna.impulseDuration
        

    def leading_edge(self,t,pulse):
        """
        Аппроксимация экспонентой заднего фронта импульса. 

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

        self.alpha = popt[0]
        return popt

    
    def trailing_edge(self, t, pulse):
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
        popt = curve_fit(func, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A0, (t.max() + t.min())/2, (t[-1]-t[0])/t.size, 0])[0]
                            
        self.A = popt[0]
        self.tau = popt[1]
        self.sigma_l = popt[2]
        self.b = popt[3]
        return popt,func

    def pulse(self, t, pulse):
        self.leading_edge(t, pulse)
        self.trailing_edge(t, pulse)

        ice = lambda t, A,alpha,tau,sigma_l,T:  A * np.exp( -alpha * (t-tau) ) * (1 + erf( (t-tau)/sigma_l ) ) + T
        popt = curve_fit(ice, 
                            xdata=t,
                            ydata=pulse,
                            p0=[self.A, self.alpha, self.tau, self.sigma_l, self.b],
                        )[0]
        return popt, ice 


    def swh(self, sigma_l):
        sigma_p = 0.425 * self.T
        sigma_c = sigma_l/np.sqrt(2)
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2))*self.c/2
        return 4*sigma_s

    def height(self, tau):
        return tau*self.c/2

    def emb(self, swh, U10, dtype = "Rostov"):
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
    def __init__(self, sigma_s, const):
        self.R = const["constants"]["earthRadius"][0]
        self.c = const["constants"]["lightSpeed"][0]

        self.xi = np.deg2rad(const["antenna"]["deviation"][0]) 
        self.theta = np.deg2rad(const["antenna"]["gainWidth"][0]) 
        self.Gamma = self.gamma(self.theta)

        self.h = const["antenna"]["z"][0]
        self.sigma_s = sigma_s 
        self.T = const["antenna"]["impulseDuration"][0]


    def H(self,h):
        return h*( 1+ h/self.R )
    
    def A(self,gamma,xi,A0=1.):
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    def u(self,t,alpha,sigma_c):
        return (t - alpha * sigma_c**2) / (np.sqrt(2) * sigma_c)

    def v(self,t,alpha,sigma_c):
        return alpha*(t - alpha/2 * sigma_c**2)

    def alpha(self,beta,delta):
        return delta - beta**2/4

    def delta(self,gamma,xi,h):
        return 4/gamma * self.c/self.H(h) * np.cos(2 * xi)
    
    def gamma(self,theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self,gamma,xi,h):
        return 4/gamma * np.sqrt(self.c/self.H(h)) * np.sin(2*xi)


    def sigma_c(self,sigma_s):
        sigma_p = 0.425 * self.T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/self.c)**2 )

    def pulse(self,t, dim = 1):

        self.dim = dim
        gamma = self.Gamma
        delta = self.delta(gamma,self.xi,self.h)
        beta  = self.beta(gamma,self.xi,self.h)

        if dim == 1:
            alpha = self.alpha(beta,delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2),delta)

        sigma_c = self.sigma_c(self.sigma_s)

        u = self.u(t, alpha, sigma_c)
        v = self.v(t, alpha, sigma_c)

        A = self.A(gamma,self.xi)
        pulse = A*np.exp(-v)*( 1 + erf(u) )
        
        if self.dim == 2:
            alpha = gamma
            u = self.u(t, alpha, sigma_c)
            v = self.v(t, alpha, sigma_c)
            pulse -= A/2*np.exp(-v)*( 1 + erf(u) )

        return pulse
