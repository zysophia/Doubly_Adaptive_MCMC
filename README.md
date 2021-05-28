# Fast Doubly-Adaptive MCMC to Estimate the GibbsPartition Function with Weak Mixing Time Bounds

## Introduction
We present a novel method for reducing the computational complexity of rigorously estimating the partition functions of Gibbs (or Boltzmann) distributions, which arise ubiquitously in probabilistic graphical models. A major obstacle to applying the Gibbs distribution in practice is the need to estimate their partition function (normalizing constant).  The state of the art in addressing this problem is multi-stage algorithms which consist of a cooling schedule and a mean estimator in each step of the schedule.  While the cooling schedule in these algorithms is adaptive, the mean estimate computations use MCMC as a black-box to draw approximately-independent samples. Here we develop a doubly adaptive approach, combining the adaptive cooling schedule with an adaptive MCMC mean estimator, whose number of Markov chain steps adapts dynamically to the underlying chain. Through rigorous theoretical analysis, we prove that our method outperforms the state of the art algorithms in several factors: (1) The computational complexity of our method is smaller; (2) Our method is less sensitive to loose bounds on mixing times, an inherent components in these algorithms; and (3) The improvement obtained by our method is particularly significant in the most challenging regime of high precision estimates. We demonstrate the advantage of our method in experiments run on classic factor graphs, such as voting models and Ising models. 

## Reproduce the experiments

- All the code is available in the directory `src/`
- Required packages and dependencies are in `src/requirements.txt`
- run `python3 ./src/main1.py` to get experimental results for Voting models
- run `python3 ./src/main2.py` to get experimental results for Ising models
- run `python3 ./src/log_process.py` to process the logging info
- run `python3 ./src/log_visualize.py` to obtain the corresponding plots
