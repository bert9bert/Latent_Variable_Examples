'''
Example of using a Kalman filter to perform maximum likelihood estimation of the
parameters of an ARMA(1,1) process.
'''

import numpy as np
from scipy import optimize

### Define program parameters ###
MAX_MLE_ITER = 100


### Generate data ###
## data generation burn-in period
burnin = 10

## parameters
T = 100

mu = 10
phi = 0.8
theta = 0.7
sigma2 = 2

## generate data
y_vec = [mu]
epsilon_last = np.random.normal(0, np.sqrt(sigma2), 1)[0]

for i in range(burnin+T):
    epsilon_this = np.random.normal(0, np.sqrt(sigma2), 1)[0]
    y_this = mu + phi*(y_vec[-1]-mu) + epsilon_this + theta*epsilon_last

    y_vec.append(y_this)
    epsilon_last = epsilon_this

y_vec = y_vec[-T:]


### Estimate parameters ###
## Initial parameter guesses ##
mu_guess     = 1
phi_guess    = 0.2
theta_guess  = 0.3
sigma2_guess = 3

mu_estvec     = [mu_guess]
phi_estvec    = [phi_guess]
theta_estvec  = [theta_guess]
sigma2_estvec = [sigma2_guess]


## Define log likelihood function
def loglik(y_vec, state_est_vec, state_esterror_covmat_vec, mu, phi, theta, sigma2):

    F = make_F(phi)
    Q = make_Q(sigma2)
    A = np.array([[mu]])
    x_this = np.array([[1]])
    H = make_H(theta)
    R = [[0]]

    n = 1

    # compute observation level likelihoods
    lik_t = []
    for t in range(len(y_vec)):
        state_est_last = state_est_vec[t]
        state_esterror_covmat_last = state_esterror_covmat_vec[t]

        innovation_residual = y_vec[t] - A.T.dot(x_this) - H.T.dot(state_est_last)
        innovation_covmat = H.T.dot(state_esterror_covmat_last).dot(H) + R

        lik_new = \
            (2*np.pi)**(-n/2) * np.linalg.det(H.T.dot(state_esterror_covmat_last).dot(H) + R)**(-1/2) * \
            np.exp((-1/2) * innovation_residual.T.dot(np.linalg.inv(innovation_covmat)).dot(innovation_residual))
        lik_t.append(lik_new[0][0])

        ### BEGIN DEBUG
        if lik_new<=0:
            print(innovation_residual[0][0], np.linalg.inv(innovation_covmat)[0][0])
        ### END DEBUG

    # compute and return log likelihood
    return sum([np.log(lik_t[t]) for t in range(len(y_vec))])

## Define useful functions for Kalamn filter
def make_F(phi):
    return np.array([[phi, 0], [1, 0]])

def make_Q(sigma2):
    return np.array([[sigma2, 0], [0, 0]])

def make_H(theta):
    return(np.array([[1], [theta]]))

def make_state_esterror_covmat_0(phi, sigma2):
    return np.array([[sigma2/(1-phi**2), phi*sigma2/(1-phi**2)], [phi*sigma2/(1-phi**2), sigma2/(1-phi**2)]])


## Estimate parameters

for i in range(MAX_MLE_ITER):
    ## set up state space for Kalman filter ##
    F = make_F(phi_estvec[-1])
    Q = make_Q(sigma2_estvec[-1])
    A = np.array([[mu_estvec[-1]]])
    x_this = np.array([[1]])
    H = make_H(theta_estvec[-1])
    R = [[0]]

    state_est_0 = np.array([[0], [0]])
    state_esterror_covmat_0 = make_state_esterror_covmat_0(phi_estvec[-1], sigma2_estvec[-1])

    state_est_vec = [state_est_0]
    state_esterror_covmat_vec = [state_esterror_covmat_0]

    ## perform Kalman filter ##
    for j in range(1,T):
        state_est_last = state_est_vec[-1]
        state_esterror_covmat_last = state_esterror_covmat_vec[-1]

        innovation_residual = y_vec[j] - A.T.dot(x_this) - H.T.dot(state_est_last)
        innovation_covmat = H.T.dot(state_esterror_covmat_last).dot(H) + R

        state_est_this = \
            F.dot(state_est_last) + \
            F.dot(state_esterror_covmat_last).dot(H).dot(np.linalg.inv(innovation_covmat)).dot(innovation_residual)

        state_esterror_covmat_this = \
            F.dot(state_esterror_covmat_last - state_esterror_covmat_last.dot(H).dot(np.linalg.inv(innovation_covmat)).dot(H.T).dot(state_esterror_covmat_last)).dot(F.T) + Q

        state_est_vec.append(state_est_this)
        state_esterror_covmat_vec.append(state_esterror_covmat_this)

        state_est_last = state_est_this
        state_esterror_covmat_last = state_esterror_covmat_this

    ## find this iteration's MLE estimates of the parameters ##
    # ...
    loss = lambda x: -loglik(y_vec, state_est_vec, state_esterror_covmat_vec, x[0], x[1], x[2], x[3])
    x0 = (mu_estvec[-1], phi_estvec[-1], theta_estvec[-1], sigma2_estvec[-1])
    bnds = ((None,None), (None,None), (None,None), (0,None))
    res = optimize.minimize(loss, x0, method='SLSQP', bounds=bnds)  # TODO: put in other args

    mu_estvec.append(res.x[0])
    phi_estvec.append(res.x[1])
    theta_estvec.append(res.x[2])
    sigma2_estvec.append(res.x[3])

## Display results
print('(mu, phi, theta, sigma2) = (%f, %f, %f, %f)' % (mu, phi, theta, sigma2))
print('(mu, phi, theta, sigma2)^hat = (%f, %f, %f, %f)' % (mu_estvec[-1], phi_estvec[-1], theta_estvec[-1], sigma2_estvec[-1]))
