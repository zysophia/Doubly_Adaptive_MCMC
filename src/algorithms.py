import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math
from gibbsChains import *
from meanEstimator import *
from tpa import *


def kolmogorov(e, kappa, gibbsChain, bmin, bmax, d, compute_z = False):
    print("running kolmogorov sampling...")
    print(f"e = {e}, kappa = {kappa}, bmin = {bmin}, bmax = {bmax}, d = {d}")
    tao_dict = {256: 1.260, 128: 1.372, 64:1.539, 32: 1.794, 16: 2.197, 8: 2.86, 4:4.0}
    sample_complexity = 0

    etilt = 1 - 1/np.sqrt((1+e))
    r = int(np.ceil(2/etilt**2))
    gamma = 0.24
    m = tao_dict[d]/2/np.log(1+ gamma) * np.log(gibbsChain.get_Hmax())
    k = int(m*d)

    gibbsChain.beta = bmax
    q = np.log(gibbsChain.get_upper_Q())
    tvd = kappa/ (k*q + m*q*r + 3*r + 1)
    # sample complexity for TPA
    res = TPA_k_d(bmin = bmin, bmax=bmax, k = k, d = d, gibbsChain = gibbsChain, tvd = tvd)
    schedule, TPAsteps = res["schedule"], res["steps"]

    sample_complexity += TPAsteps
    # sample complexity for paired_product
    for beta in schedule[:-1]:
        gibbsChain.beta = beta
        sample_complexity += gibbsChain.compute_mixingtime(tvd = tvd) * r
    for beta in schedule[1:]:
        gibbsChain.beta = beta
        sample_complexity += gibbsChain.compute_mixingtime(tvd = tvd) * r
    if (compute_z == False):
        return sample_complexity, TPAsteps, None

    # in this case, we actually run KOL to compute z
    zz = [0 for _ in range(r)]
    for rr in range(r):
        z = 1.0
        for i in range(len(schedule)-1):
            gap = schedule[i+1]-schedule[i]
            gibbsChain.beta = schedule[i]
            gibbsChain.set_startpoint()
            func_f = lambda x: np.exp(-gap/2*gibbsChain.get_Hamiltonian(x)) 
            for s in range(gibbsChain.compute_mixingtime(tvd = tvd)):
                gibbsChain.step()
            z /= func_f(gibbsChain.current)
            gibbsChain.beta = schedule[i+1]
            gibbsChain.set_startpoint()
            func_g = lambda x: np.exp(gap/2*gibbsChain.get_Hamiltonian(x)) 
            for s in range(gibbsChain.compute_mixingtime(tvd = tvd)):
                gibbsChain.step()
            z *= func_g(gibbsChain.current)
        zz[rr] = z
    return sample_complexity, TPAsteps, np.mean(zz)




def parallelGibbs(schedule = None, TPAsteps = 0, bmin = 0, bmax = 1, gibbsChain = None, eps = 0.1, delta = 0.25, kappa = 0.2, d = 64, trace = True):

    print("running Parallel Gibbs...")
    print(f"l = {len(schedule)}, trace = {trace}, e = {eps}, delta = {delta}, kappa = {kappa}, bmin = {bmin}, bmax = {bmax}, d = {d}")

    z = 1.0
    w = 1.0
    v = 1.0
    sample_complexity = 0
    Hmax = gibbsChain.get_Hmax()
    Hmin = gibbsChain.get_Hmin()
    sample_complexity += TPAsteps
    l = len(schedule)
    
    # get mean-estimator params
    epsprime = ((1+eps)**(1/l)-1) / ((1+eps)**(1/l)+1)
    delprime = delta/2/l

    for i in range(l-1):
        print(i)
        gap = schedule[i+1]-schedule[i]

        gibbsChain.beta = schedule[i]
        gibbsChain.set_startpoint()
        func_f = lambda x: np.exp(-gap/2*gibbsChain.get_Hamiltonian(x))
        a = np.exp(-gap/2*Hmax)
        b = np.exp(-gap/2*Hmin)
        res1 = mean_estimator(gibbsChain, func_f, epsprime, delprime, a, b, use_trace=trace)
        wi = res1["mean_value"]
        sample_complexity += res1["steps"]

        gibbsChain.beta = schedule[i+1]
        gibbsChain.set_startpoint()
        func_g = lambda x: np.exp(gap/2*gibbsChain.get_Hamiltonian(x)) 
        a = np.exp(gap/2*Hmin)
        b = np.exp(gap/2*Hmax)
        res2 = mean_estimator(gibbsChain, func_g, epsprime, delprime, a, b, use_trace=trace)
        vi = res2["mean_value"]
        sample_complexity += res2["steps"]

        w *= wi
        v *= vi
    z = v/w

    return z, sample_complexity



def superGibbs(schedule = None, TPAsteps = 0, bmin = 0, bmax = 1, gibbsChain = None, eps = 0.1, delta = 0.25, kappa = 0.2, d = 64, trace = True):
    
    print("running Super Gibbs...")
    print(f"l = {len(schedule)}, e = {eps}, trace = {trace}, delta = {delta}, kappa = {kappa}, bmin = {bmin}, bmax = {bmax}, d = {d}")

    z = 1.0
    sample_complexity = 0
    Hmax = gibbsChain.get_Hmax()
    Hmin = gibbsChain.get_Hmin()
    sample_complexity += TPAsteps
    l = len(schedule)
   
    # get mean-estimator params
    epsprime = eps/(2+eps)
    delprime = delta/2

    # prepare product chain
    prodChain1 = ProductGibbsChain(gibbsChain, betas = schedule[:-1])
    prodChain2 = ProductGibbsChain(gibbsChain, betas = schedule[1:])

    gaps = [schedule[i+1]-schedule[i] for i in range(len(schedule)-1)]

    # prepare functions
    def funcf(x):
        assert(len(x) == len(gaps))
        ret = 1.0
        for i in range(len(x)):
            ret *= np.exp( - gaps[i]/2 * gibbsChain.get_Hamiltonian(x[i]))
        return ret

    def funcg(x):
        assert(len(x) == len(gaps))
        ret = 1.0
        for i in range(len(x)):
            ret *= np.exp( gaps[i]/2 * gibbsChain.get_Hamiltonian(x[i]))
        return ret

    a = np.exp(-(bmax-bmin)/2*Hmax) 
    b = np.exp(-(bmax-bmin)/2*Hmin) 
    res1 = mean_estimator(prodChain1, funcf, epsprime, delprime, a, b, use_trace = trace)
    ww = res1["mean_value"]
    sample_complexity += res1["steps"]
    a = np.exp((bmax-bmin)/2*Hmin) 
    b = np.exp((bmax-bmin)/2*Hmax) 
    res2 = mean_estimator(prodChain2, funcg, epsprime, delprime, a, b, use_trace = trace)
    vv = res2["mean_value"]
    sample_complexity += res2["steps"]
    z *= vv/ww
    
    return z, sample_complexity



    
