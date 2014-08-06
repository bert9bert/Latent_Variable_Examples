'''
Example of fitting a mixture of two Bernoulli distributions
using EM.
'''

import numpy as np
from scipy.stats import binom
import pandas as pd

### Set program parameters ###
N = 1000  # number of observations
M = 10  # for each observation, the number of draws to make from the selected urn
MAXITER = 1000  # max number of iterations for EM to perform


### Set paramters for data generating process ###
tau = 0.35  # probability of choosing distribution 0
theta0 = 0.20  # probability of success for distribution 0
theta1 = 0.70  # probability of success for distribution 1


### Set initial guesses for the EM algorithm ###
tau_guess = 0.70
theta0_guess = 0.15
theta1_guess = 0.90


### Simulate data ###
# latent data
zdata = np.random.binomial(1,1-tau,N)

# observed data
xdata = np.empty((N,M), dtype=int)

for obs in range(N):
	if zdata[obs]==0:
		xdata[obs] = np.random.binomial(1,theta0,M)
	else:
		xdata[obs] = np.random.binomial(1,theta1,M)

# represent the observed data, where each observation is a
# sequence of Bernoulli draws, as Binomial distributed data
xBinData = np.sum(xdata, axis=1)

### Estimate parameters with Expectation-Maximization Algorithm ###

# create vectors to store estimates at each iteration, and put 
# initial guess for parameters in position 0
theta0_estvec = [theta0_guess]
theta1_estvec = [theta1_guess]
tau_estvec = [tau_guess]

## EM Loop ##
for i in range(MAXITER):
	## get the estimates of the last iteration
	theta0_estlast = theta0_estvec[-1]
	theta1_estlast = theta1_estvec[-1]
	tau_estlast = tau_estvec[-1]

	## E Step
	# compute membership probabilities
	# (from an application of Bayes' rule)
	pr_x_cond_z0 = binom.pmf(xBinData, M, theta0_estlast)
	pr_x_cond_z1 = binom.pmf(xBinData, M, theta1_estlast)

	pr_x_z0 = pr_x_cond_z0*tau_estlast
	pr_x_z1 = pr_x_cond_z1*(1-tau_estlast)

	pr_total = pr_x_z0 + pr_x_z1

	T0 = pr_x_z0/pr_total
	T1 = 1 - T0


	## M Step
	# (all the parameters are in separate linear terms of the quasi-log likelihood function, so 
	# all the parameters can be maximized separately)
	tau_estthis = np.average(T0)
	theta0_estthis = xBinData.dot(T0)/(M*np.sum(T0))
	theta1_estthis = xBinData.dot(T1)/(M*np.sum(T1))


	## store the estimates of this iteration
	theta0_estvec.append(theta0_estthis)
	theta1_estvec.append(theta1_estthis)
	tau_estvec.append(tau_estthis)


### Display results ###
em_results_df = pd.DataFrame(
	{'tau_hat': tau_estvec, 'theta0_hat': theta0_estvec, 'theta1_hat': theta1_estvec})
print(em_results_df[0:10])
print(em_results_df[-10:])
print('(tau, theta0, theta1) = (%f, %f, %f)' % (tau, theta0, theta1))


