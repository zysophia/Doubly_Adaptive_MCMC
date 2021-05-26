# Fast Doubly-Adaptive MCMC to Estimate the GibbsPartition Function with Weak Mixing Time Bounds

## Introduction
We present a novel method for reducing the sample complexity of rigorously estimating the partition functionsof Gibbs (or Boltzmann) distributions, which arise ubiquitously in probabilistic graphical models. 

## Reproduce the experiments

- All the code is available in the directory `src/`
- Required packages and dependencies are in `src/requirements.txt`
- run `python3 ./src/main1.py` to get experimental results for Voting models
- run `python3 ./src/main2.py` to get experimental results for Ising models
- run `python3 ./src/log_process.py` to process the logging info
- run `python3 ./src/log_visualize.py` to obtain the corresponding plots
