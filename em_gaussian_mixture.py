'''
Gaussian mixture example
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from statutils import multivariate_normal

### set program parameters ###
N = 100  # number of observations
MAXITER = 100  # EM iteration stops once this many iterations are performed

### set true parameters for Gaussian mixture ###
d = 2  # dimensions of Gaussian

# first Gaussian
mu_dist0 = np.array([15, 75])
sigma_dist0 = np.array([[15, 0], [0, 30]])

# second Gaussian
mu_dist1 = np.array([70, 30])
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
sigma_dist0_estvec[0] = np.array([[10, 0], [0, 20]])
mu_dist1_estvec[0] = np.array([10,10])
sigma_dist1_estvec[0] = np.array([[10, 0], [0, 30]])
tau_estvec[0] = 0.50

loglik[0] = np.inf


## perform EM ##

T = np.empty((N,2))

for i in range(1,MAXITER+1):
    # E Step
    for obs in range(N):
        denom = (1-tau_estvec[i-1])*multivariate_normal(x_vec[obs], mu_dist0_estvec[i-1], sigma_dist0_estvec[i-1]) + \
            tau_estvec[i-1]*multivariate_normal(x_vec[obs], mu_dist1_estvec[i-1], sigma_dist1_estvec[i-1])

        T[obs,0] = (1-tau_estvec[i-1])*multivariate_normal(x_vec[obs], mu_dist0_estvec[i-1], sigma_dist0_estvec[i-1])/denom
        T[obs,1] = tau_estvec[i-1]*multivariate_normal(x_vec[obs], mu_dist1_estvec[i-1], sigma_dist1_estvec[i-1])/denom

    # M Step
    tau_estvec[i] = np.sum(T[:,1])/N

    mu_dist0_estvec[i] = T[:,0].dot(x_vec)/np.sum(T[:,0])
    mu_dist1_estvec[i] = T[:,1].dot(x_vec)/np.sum(T[:,1])

    diffpart0 = x_vec-mu_dist0_estvec[i]
    diffpart1 = x_vec-mu_dist1_estvec[i]

    sigma_dist0_estvec[i] = np.zeros(sigma_dist0_estvec[i].shape)
    for k in range(N):    
        sigma_dist0_estvec[i] = sigma_dist0_estvec[i] + \
            T[k,0]*np.outer(diffpart0[k], diffpart0[k].T)
    sigma_dist0_estvec[i] = sigma_dist0_estvec[i]/np.sum(T[:,0])

    sigma_dist1_estvec[i] = np.zeros(sigma_dist1_estvec[i].shape)
    for k in range(N):    
        sigma_dist1_estvec[i] = sigma_dist1_estvec[i] + \
            T[k,1]*np.outer(diffpart1[k], diffpart1[k].T)
    sigma_dist1_estvec[i] = sigma_dist1_estvec[i]/np.sum(T[:,1])

## trim storage vectors ##
numiters = MAXITER  # accounts for index 0

### Display results ###
tau_est = tau_estvec[numiters]
mu_dist0_est = mu_dist0_estvec[numiters,:]
mu_dist1_est = mu_dist1_estvec[numiters,:]
sigma_dist0_est = sigma_dist0_estvec[numiters,:,:]
sigma_dist1_est = sigma_dist1_estvec[numiters,:,:]

plt.scatter(x_vec[:,0], x_vec[:,1])

x = np.arange(0, 100, 0.025)
y = np.arange(0, 100, 0.025)
X, Y = np.meshgrid(x, y)
Z0 = mlab.bivariate_normal(X, Y, sigma_dist0_est[0,0], sigma_dist0_est[1,1], mu_dist0_est[0], mu_dist0_est[1], sigma_dist0_est[1,0])
Z1 = mlab.bivariate_normal(X, Y, sigma_dist1_est[0,0], sigma_dist1_est[1,1], mu_dist1_est[0], mu_dist1_est[1], sigma_dist1_est[1,0])

plt.contour(X,Y,Z0)
plt.contour(X,Y,Z1)

plt.title('Expectation-Maximization Algorithm Estimation of Two Gaussians')

plt.show()
