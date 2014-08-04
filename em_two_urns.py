'''
Example of fitting a mixture of two Bernoulli distributions
using EM.
'''

import numpy as np
from scipy.stats import binom

### Set program parameters ###
N = 100  # number of observations
M = 10  # for each observation, the number of draws to make from the selected urn
MAXITER = 100  # max number of iterations for EM to perform


### Set paramters for data generating process ###
theta0 = 0.20  # probability of success for distribution 0
theta1 = 0.70  # probability of success for distribution 1
tau = 0.45  # probability of choosing distribution 0


### Set initial guesses for the EM algorithm ###
theta0_guess = 0.15
theta1_guess = 0.90
tau_guess = 0.70


### Simulate data ###
# latent data
zdata = np.random.binomial(1,tau,N)

# observed data
xdata = np.empty((N,M), dtype=int)

for obs in range(N):
	if zdata[obs]==0:
		xdata[obs] = np.random.binomial(1,theta0,M)
	else:
		xdata[obs] = np.random.binomial(1,theta1,M)

# represent the observed data, where each observation is a
# sequence of Bernoulli draws, as Binomial distributed data
xBinData = [sum(x) for x in xdata]

### Estimate parameters with Expectation-Maximization Algorithm ###

# create vectors to store estimates at each iteration, and put 
# initial guess for parameters in position 0
theta0_estvec = [theta0_guess]
theta1_estvec = [theta1_guess]
tau_estvec = [tau_guess]

## EM Loop ##
for i in range(MAXITER):
	theta0_estlast = theta0_estvec[-1]
	theta1_estlast = theta1_estvec[-1]
	tau_estlast = tau_estvec[-1]

	## E Step
	# compute membership probabilities
	T = []  # let first dim be rows, second dim be prob 0 and prob 1
	# ...



	## M Step


### Display results ###

