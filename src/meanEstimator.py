import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math


def mean_estimator(gibbsChain, f, e, d, a, b, use_trace):
    Lambda = gibbsChain.get_Lambda()
    uniform_mixing = gibbsChain.get_uniform_mixing()
    R = b-a
    gamma = 1.1
    # print("b/a", b/a)
    print("\nR = ", R, "lambda = ", Lambda)
    if use_trace:
        T = int(np.ceil((1+Lambda)/(1-Lambda) * np.log(2) / 2))
    else:
        T = 1
    print("T = ", T)
    Lambda = Lambda**T
    # print("now Lambda", Lambda, "T", T)
    I = max(1, int(math.log(b*R/2/a**2 *(1-e)**2/e/(1+e), gamma)))
    # I = 25
    ln3Id = np.log(3*I/d)
    alpha = (1+Lambda)*R*ln3Id*(1+e)/ b/(1-Lambda)/e

    gibbsChain1 = deepcopy(gibbsChain)
    gibbsChain2 = deepcopy(gibbsChain)
    gibbsChain1.restart_and_sample(steps = uniform_mixing) 
    gibbsChain2.restart_and_sample(steps = uniform_mixing)

    pchain1vals = []
    pchain2vals = []
    m = 0

    for i in range(1, I+1):
        mi = int(np.ceil(gamma**i*alpha))
        pchain1vals += [0 for _ in range(mi-m)]
        pchain2vals += [0 for _ in range(mi-m)]
        for j in range(m, mi):
            chain1vals = [0 for _ in range(T)]
            chain2vals = [0 for _ in range(T)]
            for t in range(T):
                gibbsChain1.step()
                gibbsChain2.step()
                chain1vals[t] = f(gibbsChain1.current)
                chain2vals[t] = f(gibbsChain2.current)
            pchain1vals[j] = np.mean(chain1vals)
            pchain2vals[j] = np.mean(chain2vals)
        empirical_mean = np.sum(pchain1vals + pchain2vals) / 2/mi
        empirical_tv = np.sum([(pchain1vals[k]-pchain2vals[k])**2 for k in range(mi)]) / 2/mi
        variance_upper = empirical_tv + (11+np.sqrt(21))* (1+Lambda/np.sqrt(21)) * R**2 * ln3Id / (1-Lambda)/mi +\
            R * np.sqrt( (1+Lambda)/(1-Lambda) * empirical_tv * ln3Id / mi )
        bernstein_bound = 10*R*ln3Id/(1-Lambda)/mi + np.sqrt( (1+Lambda)/(1-Lambda) * variance_upper * ln3Id / mi )
        correction_mean = (max(a, empirical_mean-bernstein_bound) + min(b, empirical_mean+bernstein_bound))/2
        correction_error = (min(b, empirical_mean+bernstein_bound) - max(a, empirical_mean-bernstein_bound))/2/correction_mean # relative error
        # correction_error = (min(b, empirical_mean+bernstein_bound) - max(a, empirical_mean-bernstein_bound))/2# additive error
        m = mi
        print("mi=", mi, "tv =", empirical_tv, "variance_upper", variance_upper, "bernstein", bernstein_bound)
        if i==I or correction_error<=e:
            print("done, I =", I, "i =", i, e, correction_error, correction_mean)
            return {"mean_value" : correction_mean, "steps": T*mi}

