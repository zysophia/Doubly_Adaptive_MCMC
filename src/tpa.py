import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math


def TPA_1(bmin, bmax, gibbsChain, tvd):
    """
    Get a cooling schedule from one run of TPA, assuming H(x) >= 0
    params: 
        bmin, bmax -> range of beta
        gibbsChain -> oracle for approximate sampling
        tvd        -> tvd constraint on sampling
    """
    b = bmin
    schedule = []
    steps = 0
    while b < bmax:
        u = rng.random()
        gibbsChain.beta = b
        mixingtime = gibbsChain.compute_mixingtime(tvd = tvd)
        gibbsChain.restart_and_sample(steps=mixingtime)
        steps += int(mixingtime)
        Hx = gibbsChain.get_Hamiltonian(gibbsChain.current)
        if Hx == 0: break
        b -= np.log(u)/Hx
        if bmin <= b <= bmax:
            schedule.append(b)
    return {"steps": steps, "schedule": schedule}

def TPA_k_d(bmin, bmax, k, d, gibbsChain, tvd):
    """
    Get a cooling schedule from k run of TPA, assuming H(x) >= 0, distance of beta sampling is d
    params: 
        bmin, bmax -> range of beta
        k          -> # of TPA runs
        gibbsChain -> oracle for approximate sampling
        tvd        -> tvd constraint on sampling
    """
    o_schedule = []
    steps = 0
    for i in range(k):
        res = TPA_1(bmin, bmax, gibbsChain, tvd)
        subschedule, substeps = res["schedule"], res["steps"]
        o_schedule += subschedule #[1:-1]
        steps += substeps
    o_schedule.sort()
    pt = rng.randint(0, d)
    schedule = [bmin]
    while pt < len(o_schedule):
        schedule.append(o_schedule[pt])
        pt += d
    schedule.append(bmax)

    return {"steps": steps, "schedule": schedule}

