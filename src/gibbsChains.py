import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math


class GibbsChain(ABC):
    """
    A chain that could sample from a gibbs distribution
    """
    def __init__(self, beta = 0, startpoint = None):
        """ set beta and startpoint , if startpoint is not given, set it """
        self.current = startpoint
        self.beta = beta
        self.offset = self.get_offset()
        if (self.current == None):
            self.set_startpoint()

    def restart_and_sample(self, tvd = None, steps = None):
        if (tvd == None and steps == None):
            print("error in restart and sample!")
        self.set_startpoint()
        if tvd != None:
            mixtime = self.compute_mixingtime(tvd)
        else:
            mixtime = steps
        for _ in range(int(mixtime)):
            self.step()

    def get_uniform_mixing(self):
        """ return uniform mixing time """
        Lambda = self.get_Lambda()
        pai_min = self.get_lower_paimin()
        return int(np.ceil(np.log(1/pai_min)/np.log(1/Lambda)))

    def compute_mixingtime(self, tvd):
        """ compute mixing time given tvd """
        Lambda = self.get_Lambda()
        spectral_gap = 1-Lambda
        pai_min = self.get_lower_paimin()
        return int(np.ceil((np.log(1/pai_min)/2 - np.log(2 * tvd)) / spectral_gap))

    def get_upper_Q(self):
        """ return an upper bound on Q """
        Hbar = self.get_Hbar()
        return np.exp(Hbar)

    def get_lower_paimin(self):
        """ return a lower bound on pai_min """
        Hbar = self.get_Hbar()
        Hmin = self.get_Hmin()
        Hmax = self.get_Hmax()
        rho = (Hbar - Hmin) / (Hmax - Hmin)
        upper_invQ = (1 - rho) * np.exp(-self.beta * Hmax) + rho * np.exp(-self.beta * Hmin)
        Zmin = self.get_Zmin()
        return np.exp(-self.beta * Hmax) / Zmin/ upper_invQ

    @abstractmethod
    def set_startpoint(self):
        """ helper function to set the startpoint """
        pass

    @abstractmethod
    def get_Hmax(self):
        pass

    @abstractmethod
    def get_Hmin(self):
        pass

    @abstractmethod
    def get_Hbar(self):
        pass

    @abstractmethod
    def get_Lambda(self):
        pass

    @abstractmethod
    def get_Zmin(self):
        pass

    @abstractmethod
    def step(self):
        """ take a single step of the chain, update the value for *current* """
        pass

    @abstractmethod
    def get_Hamiltonian(self):
        """ compute the hamiltonian value of current sample """
        pass

    @abstractmethod
    def get_offset(self):
        return 0



class VotingChain(GibbsChain):
    def __init__(self, beta = 0, startpoint = None, n = 10, w = None, wt = None, wf = None):
        self.n = n
        self.w = w
        self.wt = wt
        self.wf = wf
        super().__init__(beta, startpoint)

    def get_Lambda(self):
        n_variable = 2*self.n+1
        hw = 2*n_variable+1
        return 1 - np.exp(-3*hw*self.beta*2)/n_variable 

    def get_Hmax(self):
        return self.offset + sum([x for x in self.wt if x>0]) + sum([x for x in self.wf if x>0]) + np.abs(self.w)*self.n
    
    def get_Hmin(self):
        return self.offset + sum([x for x in self.wt if x<0]) + sum([x for x in self.wf if x<0]) - np.abs(self.w)*self.n

    def get_Hbar(self):
        return sum(self.wf)/2 + sum(self.wt)/2 + self.offset

    def get_Zmin(self):
        return 2**(2*self.n+1)

    def set_startpoint(self):
        Q = rng.randint(0, 2)*2-1 # {-1, 1}
        Ts = rng.randint(0, 2, self.n)
        Fs = rng.randint(0, 2, self.n)
        self.current = {"Q": Q, "Ts": Ts, "Fs": Fs}

    def step(self):
        Q = self.current["Q"]
        Ts = self.current["Ts"]
        Fs = self.current["Fs"]
        rv = rng.randint(0, self.n * 2 + 1) # pick the changed variable uniformly 
        if rv == 0:
            ratio = np.exp(-self.beta * (-Q-Q) * (sum(Ts) - sum(Fs)))
        elif rv <= self.n:
            ratio = np.exp(-self.beta * (-Ts[rv-1]+0.5)*2)
        else:
            ratio = np.exp(-self.beta * (-Fs[rv-self.n-1]+0.5)*2)
        if rng.random() < ratio/(1+ratio): 
            if rv == 0:
                self.current["Q"] = - self.current["Q"]
            elif rv <= self.n:
                self.current["Ts"][rv-1] = 1 - self.current["Ts"][rv-1]
            else:
                self.current["Fs"][rv-self.n-1] = 1 - self.current["Fs"][rv-self.n-1]

    def get_Hamiltonian(self, X):
        Q = X["Q"]
        Ts = X["Ts"]
        Fs = X["Fs"]
        Hx = self.w * Q * (sum(Ts) - sum(Fs))
        for i in range(self.n):
            Hx += self.wt[i]*Ts[i] + self.wf[i]*Fs[i]
        Hx += self.offset # this is to make sure Hx >= 1
        return Hx

    def get_offset(self):
        return sum([-x for x in self.wt if x<0]) + sum([-x for x in self.wf if x<0]) + np.abs(self.w)*self.n + 1

    
class VotingChainLogical(GibbsChain):
    def __init__(self, beta = 0, startpoint = None, n = 10, w = None, wt = None, wf = None):
        self.n = n
        self.w = w
        self.wt = wt
        self.wf = wf
        super().__init__(beta, startpoint)

    def get_Lambda(self):
        n_variable = 2*self.n+1
        hw = 3
        return 1 - np.exp(-3*hw*self.beta*2)/n_variable 

    def get_Hmax(self):
        return self.offset+ np.abs(self.w)+ sum([x for x in self.wt if x>0]) + sum([x for x in self.wf if x>0])

    def get_Hmin(self):
        return self.offset- np.abs(self.w)+ sum([x for x in self.wt if x<0]) + sum([x for x in self.wf if x<0])

    def get_Hbar(self):
        return sum(self.wf)/2 + sum(self.wt)/2 + self.offset

    def get_Zmin(self):
        return 2**(2*self.n+1) 

    def set_startpoint(self):
        Q = rng.randint(0, 2)*2-1 # {-1, 1}
        Ts = rng.randint(0, 2, self.n)
        Fs = rng.randint(0, 2, self.n)
        self.current = {"Q": Q, "Ts": Ts, "Fs": Fs}

    def step(self):
        Ham1 = self.get_Hamiltonian(self.current)
        rv = rng.randint(0, self.n * 2 + 1) # pick the changed variable uniformly 
        if rv == 0:
            self.current["Q"] = -self.current["Q"]
        elif rv <= self.n:
            self.current["Ts"][rv-1] = 1 - self.current["Ts"][rv-1]
        else:
            self.current["Fs"][rv-self.n-1] = 1 - self.current["Fs"][rv-self.n-1]
        ratio = np.exp(-self.beta * (self.get_Hamiltonian(self.current) - Ham1))
        if rng.uniform() >= ratio/(1+ratio): 
            # change back
            if rv == 0:
                self.current["Q"] = -self.current["Q"]
            elif rv <= self.n:
                self.current["Ts"][rv-1] = 1 - self.current["Ts"][rv-1]
            else:
                self.current["Fs"][rv-self.n-1] = 1 - self.current["Fs"][rv-self.n-1]

    def get_Hamiltonian(self, X):
        Q = X["Q"]
        Ts = X["Ts"]
        Fs = X["Fs"]
        Hx = self.w * Q * (max(Ts) - max(Fs))
        for i in range(self.n):
            Hx += self.wt[i]*Ts[i] + self.wf[i]*Fs[i]
        Hx += self.offset # this is to make sure Hx >= 1
        return Hx

    def get_offset(self):
        return sum([-x for x in self.wt if x<0]) + sum([-x for x in self.wf if x<0]) + np.abs(self.w) + 1



class IsingChainLattice(GibbsChain):
    def __init__(self, beta = 0, startpoint = None, n = 10):
        self.n = n
        super().__init__(beta, startpoint)

    def get_Lambda(self):
        return 1-1/(10*self.n**2*2*np.log(self.n))

    def get_Hmax(self):
        return self.offset + (self.n-1)*self.n*2
    
    def get_Hmin(self):
        return self.offset

    def get_Hbar(self):
        return self.offset + (self.n-1)*self.n

    def get_offset(self):
        return 1

    def set_startpoint(self):
        self.current = rng.randint(0,2,size=self.n**2)

    def step(self):
        rv = rng.randint(0, self.n**2)
        neighbors = [rv-1, rv+1, rv-self.n, rv+self.n]
        match = 0
        for neighbor in neighbors:
            if 0<=neighbor<self.n**2:
                match += int(self.current[rv] == self.current[neighbor])
        diff = 4-match*2
        ratio = np.exp(-self.beta*diff)
        if rng.uniform() < ratio/(1+ratio):
            self.current[rv] = 1 - self.current[rv]

    def get_Hamiltonian(self, X):
        match = 0
        for p in range(self.n**2):
            neighbors = [p+1, p+self.n]
            for neighbor in neighbors:
                if 0<=neighbor<self.n**2 and X[p] == X[neighbor]:
                    match += 1
                # print(X, p, neighbor, match)
        return self.offset + match

    def get_upper_Q(self):
        Hbar = self.get_Hbar()
        Hmin = self.get_Hmin()
        Hmax = self.get_Hmax()
        rho = (Hbar - Hmin) / (Hmax - Hmin)
        return (1 - rho) * np.exp(-self.beta * Hmax) + rho * np.exp(-self.beta * Hmin)

    def get_lower_paimin(self):
        Hmin = self.get_Hmin()
        upper_Q = self.get_upper_Q()
        Zmax = 2**(self.n**2)
        return np.exp(-self.beta*Hmin)/Zmax/upper_Q

    def get_Zmin(self):
        # this should not be called
        pass



class ProductGibbsChain():
    def __init__(self, gibbsChain, betas = [0, 1], startpoint = None):
        """ set beta and startpoint , if startpoint is not given, set it """
        self.current = startpoint
        self.chains = [deepcopy(gibbsChain) for b in betas]
        for i in range(len(betas)):
            self.chains[i].beta = betas[i]
        if (self.current == None):
            self.set_startpoint()

    def get_Lambda(self):
        return 1 - (1-self.chains[-1].get_Lambda())/ len(self.chains)

    def get_lower_paimin(self):
        paimins = [chain.get_lower_paimin() for chain in self.chains]
        paimin = 1.0
        for p in paimins:
            paimin *= p
        return paimin

    def get_uniform_mixing(self):
        Lambda = self.get_Lambda()
        pai_min = self.get_lower_paimin()
        return int(np.ceil(np.log(1/pai_min)/np.log(1/Lambda)))

    def compute_mixingtime(self, tvd):
        Lambda = self.get_Lambda()
        spectral_gap = 1-Lambda
        pai_min = self.get_lower_paimin()
        return int(np.ceil((np.log(1/pai_min)/2 - np.log(2 * tvd)) / spectral_gap))

    def set_startpoint(self):
        """ helper function to set the startpoint """
        for i in range(len(self.chains)):
            self.chains[i].set_startpoint()
        self.current = [chain.current for chain in self.chains]

    def step(self):
        """ take a single step of the chain, update the value for *current* """
        n = len(self.chains)
        i = rng.randint(0, n)
        self.chains[i].step()
        self.current[i] = self.chains[i].current

    def restart_and_sample(self, tvd = None, steps = None):
        if (tvd == None and steps == None):
            print("error in restart and sample!")
        self.set_startpoint()
        if tvd != None:
            mixtime = self.compute_mixingtime(tvd)
        else:
            mixtime = steps
        for _ in range(int(mixtime)):
            self.step()
