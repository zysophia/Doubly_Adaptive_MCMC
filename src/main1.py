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


log_filename = "../logs/output.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode="a", encoding=None, delay=False)
logging.basicConfig(handlers=[file_handler], level=logging.DEBUG)

n = 3
w = 0.9# int(rng.uniform(-1, 1)*10)/10#0.9 # rng.random()
wt = [0.5]*n# [int(x)/10 for x in rng.uniform(-1.0, 1.0, size = n//2)*10] #[0.2, 0.5, 0.1]#[0.1]*n #[0.9, 0.9, -0.1, -0.1, -0.5] # rng.random(n)
wf = [-0.4]*n# [int(x)/10 for x in rng.uniform(-1.0, 1.0, size = n//2)*10] #[-0.8, -0.2, -0.9]#[-0.8]*n #[-0.8, -0.8, -0.3, -0.3, -0.9] #rng.random(n)

logging.info("----- This is a new run -----")

chain = VotingChainLogical(n = n, w = w, wt = wt, wf = wf)
logging.info("Voting model -- logical, n = %d, w = %s", n, " ".join(str(x) for x in [w, ";"]+wt+[";"]+wf))

# chain = VotingChain(n = n, w = w, wt = wt, wf = wf)
# logging.info("Voting model -- linear, n = %d, w = %s", n, " ".join(str(x) for x in [w, ";"]+wt+[";"]+wf))

bmin = 0.0
bmax = 0.1
eps = 0.0025
delta = 0.25
kappa = 0.1
dist = 64

# TPA for parallel and super Gibbs
tao_dict = {256: 1.260, 128: 1.372, 64:1.539, 32: 1.794, 16: 2.197, 8: 2.86, 4:4.0}
Hmax = chain.get_Hmax()
Hmin = chain.get_Hmin()
gamma = 0.24
tao = tao_dict[dist]
m = tao/2/np.log(1+ gamma) * np.log(Hmax)
k = int(m*dist)
print("k = ", k)
chain.beta = bmax
q = np.log(chain.get_upper_Q())
tvd = kappa/ (k*q)
res = TPA_k_d(bmin, bmax, k, dist, chain, tvd)
schedule, TPAsteps = res["schedule"], res["steps"]

for eps in [0.25, 0.1, 0.075, 0.05, 0.04, 0.025, 0.01]:#, 0.0075, 0.005, 0.0025, 0.001, 0.0005]:

    logging.info("parameters: bmin = %.2f, bmax = %.2f, eps = %.4f, delta = %.2f, kappa = %.2f", \
        bmin, bmax, eps, delta, kappa)

    # get real z
    dp = [[]]
    for i in range(n):
        for j in range(len(dp)):
            d = dp.pop(0)
            dp.append(d+[0])
            dp.append(d+[1])

    Ts = deepcopy(dp)
    Fs = deepcopy(dp)
    z = 0
    for Q in [-1, 1]:
        for T in Ts:
            for F in Fs:
                hh = chain.get_Hamiltonian({"Q":Q, "Ts": T, "Fs": F})
                z += np.exp(-bmax*hh)
    real_z = 2**(2*n+1)/z
    print("real value z = ", real_z)
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

    # parallel Gibbs (with McMcPro)
    z, steps = parallelGibbs(schedule = schedule, TPAsteps = TPAsteps, bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = False)
    print("parallelGibbs (McMcPro) takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm parallelGibbs (McMcPro), takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)
    # parallel Gibbs (with trace)
    z, steps = parallelGibbs(schedule = schedule, TPAsteps = TPAsteps, bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    print("parallelGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm parallelGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)
    # super Gibbs (with McMcPro)
    z, steps = superGibbs(schedule = schedule, TPAsteps = TPAsteps,bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = False)
    print("superGibbs (McMcPro) takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm superGibbs (McMcPro), takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)
    # super Gibbs (with trace)
    z, steps = superGibbs(schedule = schedule, TPAsteps = TPAsteps,bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    print("superGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm superGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    logging.info("multiplicative error is %.7f", z/real_z - 1)
