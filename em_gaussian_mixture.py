'''
Gaussian mixture example
'''

import numpy as np
from scipy.stats import multivariate_normal

### set program parameters ###
N = 100  # number of observations
TOL = 0.001  # EM iteration stops once improvement in log likelihood is below this
MAXITER = 10  # EM iteration stops once this many iterations are performed

### set true parameters for Gaussian mixture ###
d = 2  # dimensions of Gaussian

# first Gaussian
mu_dist0 = np.array([2, 50])
sigma_dist0 = np.array([[15, 0], [0, 30]])

# second Gaussian
mu_dist1 = np.array([5, 90])
sigma_dist1 = np.array([[10, 0], [0, 12]])

assert mu_dist0.shape==(d,)
assert mu_dist1.shape==(d,)
assert sigma_dist0.shape==(d,d)
assert sigma_dist1.shape==(d,d)

# mixing probability (of getting the second distribution)
tau = 0.2


### Simulate data ###
latent_vec = np.random.binomial(1, tau, N)

x_vec = np.empty((N,d))
for i in range(N):
	if latent_vec[i]==0:
		x_vec[i] = np.random.multivariate_normal(mu_dist0, sigma_dist0)
	else:
		x_vec[i] = np.random.multivariate_normal(mu_dist1, sigma_dist1)


### Try to estimate parameters with EM algorithm ###
## create vectors to hold iteration data ##
mu_dist0_estvec = np.empty((MAXITER+1,d))
sigma_dist0_estvec = np.empty((MAXITER+1,d,d))
mu_dist1_estvec = np.empty((MAXITER+1,d))
sigma_dist1_estvec = np.empty((MAXITER+1,d,d))
tau_estvec = np.empty(MAXITER+1)

loglik = np.empty(MAXITER+1)

## Set initial guesses ##
mu_dist0_estvec[0] = np.array([10,10])
sigma_dist0_estvec[0] = np.array([[10, 0], [0, 10]])
mu_dist1_estvec[0] = np.array([10,10])
sigma_dist1_estvec[0] = np.array([[10, 0], [0, 10]])
tau_estvec[0] = 0.50

loglik[0] = np.inf


## perform EM ##
T = np.empty((N,2))

for i in range(1,MAXITER+1):
	# E Step
	for obs in range(N):
		denom = (1-tau_estvec[i-1])*multivariate_normal(x_vec[obs], mu_dist0_estvec[i-1], sigma_dist0_estvec[i-1]) + tau_estvec[i-1]*multivariate_normal(x_vec[obs], mu_dist1_estvec[i-1], sigma_dist1_estvec[i-1])
		T[obs,0] = (1-tau_estvec[i-1])*multivariate_normal(x_vec[obs], mu_dist0_estvec[i-1], sigma_dist0_estvec[i-1])/denom
		T[obs,1] = tau_estvec[i-1]*multivariate_normal(x_vec[obs], mu_dist1_estvec[i-1], sigma_dist1_estvec[i-1])/denom


	# M Step


## trim storage vectors ##

### Display results ###
