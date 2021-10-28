import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math
from gibbsChains import *
from algorithms import *
from meanEstimator import *
from tpa import *


log_filename = "../logs/isingoutput_v1.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode="a", encoding=None, delay=False)
logging.basicConfig(handlers=[file_handler], level=logging.DEBUG)

logging.info("----- This is a new run -----")
n = 4
chain = IsingChainLattice(n = n)
logging.info("Ising model, n = %d", n)

bmin = -0.02
bmax = 0.0
eps = 0.005
delta = 0.25
kappa = 0.1
dist = 64

chain.beta = bmin
dp = [[]]
for i in range(n**2):
    for j in range(len(dp)):
        d = dp.pop(0)
        dp.append(d+[0])
        dp.append(d+[1])
z = 0
for d in dp:
    hh = chain.get_Hamiltonian(d)
    z += np.exp(-bmin*hh)
real_z = z/2**(n**2)
print("real z:", real_z)
logging.info("real value z = %.20f", real_z)

# TPA for parallel and super Gibbs
tao_dict = {256: 1.260, 128: 1.372, 64:1.539, 32: 1.794, 16: 2.197, 8: 2.86, 4:4.0}
Hmax = chain.get_Hmax()
Hmin = chain.get_Hmin()
gamma = 0.24
tao = tao_dict[dist]
m = tao/2/np.log(1+ gamma) * np.log(Hmax)
k = int(m*dist)
print("k = ", k)
chain.beta = bmin
q = np.log(chain.get_upper_Q())
tvd = kappa/ (k*q)
res = TPA_k_d(bmin, bmax, k, dist, chain, tvd)
schedule, TPAsteps = res["schedule"], res["steps"]

for eps in [0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005]:
    
    logging.info("parameters: bmin = %.2f, bmax = %.2f, eps = %.4f, delta = %.2f, kappa = %.2f", \
        bmin, bmax, eps, delta, kappa)
    logging.info("real value z = %.20f", real_z)

    # run Kolmogorov
    compute = False
    print(f"Running Kolmogorov (compute z {compute})...")
    steps, kolTPAsteps, est_z = kolmogorov(e = eps, kappa = kappa, gibbsChain = chain, bmin = bmin, bmax = bmax, d = dist, compute_z = compute)
    print(f"kolmogorov takes {steps} steps, while z = {est_z}, and TPA takes {kolTPAsteps} steps")
    if (compute):
        logging.info("Kolmogorov (compute z) takes %d steps, while z = %.20f, and TPA takes %d steps, ", steps, est_z, kolTPAsteps)
    else:
        logging.info("Kolmogorov takes %d steps, and TPA takes %d steps", steps, kolTPAsteps)

    z, steps = parallelGibbs(schedule = schedule, TPAsteps = TPAsteps, bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    print("parallelGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm parallelGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)
        
    z, steps = superGibbs(schedule = schedule, TPAsteps = TPAsteps,bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    print("superGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm superGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)

        