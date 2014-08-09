'''
Example of using a Kalman filter to perform maximum likelihood estimation of the
parameters of an ARMA(1,1) process.
'''

import numpy as np
from scipy import optimize

### Define program parameters ###
MAX_MLE_ITER = 250


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
mu_guess     = 9
phi_guess    = phi
theta_guess  = 0.5
sigma2_guess = sigma2

mu_estvec     = [mu_guess]
phi_estvec    = [phi_guess]
theta_estvec  = [theta_guess]
sigma2_estvec = [sigma2_guess]

eval_loglik_vec = []  # vector to store the evaluated log likelihood


## Define log likelihood function
def loglik(y_vec, state_est_vec, state_esterror_covmat_vec, mu, phi, theta, sigma2):

    F = make_F(phi)
    Q = make_Q(sigma2)
    A = np.array([[mu]])
    x_this = np.array([[1]])
    H = make_H(theta)
    R = [[0]]

    n = 1

    # compute vector of measurement residuals with these parameters
    ytilde_vec = [y_vec[t] - A.T.dot(x_this) - H.T.dot(state_est_vec[t]) for t in range(T)]

    # compute vector of measurement residual covariances with these parameters
    S_vec = [H.T.dot(state_esterror_covmat_vec[t]).dot(H) + R for t in range(T)]

    # compute and return log likelihood
    LL = -n*T/2*np.log(2*np.pi) - (1/2) * \
        sum([  np.log(np.linalg.det(S_vec[t])) + ytilde_vec[t].T.dot(np.linalg.inv(S_vec[t])).dot(ytilde_vec[t])[0][0] for t in range(T) ])
    return LL

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

    measresid_vec = []
    measresid_covmat_vec = []
    kalmangain_vec = []


    ## perform Kalman filter ##
    for j in range(T):
        # get the values we need from the last iteration
        xi_thispredict = state_est_vec[-1]  # the state estimate at time t given data up to time t-1
        P_thispredict = state_esterror_covmat_vec[-1]  # the state estimate covariance at time t given data up to time t-1

        # get next predicted state given the last predicted state
        ytilde_this = y_vec[j] - A.T.dot(x_this) - H.T.dot(xi_thispredict)  # measurement residual
        S_this = H.T.dot(P_thispredict).dot(H) + R  # measurement residual covariance

        K_this = P_thispredict.dot(H).dot(np.linalg.inv(S_this))  # optimal Kalman gain

        xi_thisupdate = xi_thispredict + K_this.dot(ytilde_this)  # the state estimate at time t given data up to time t
        P_thisupdate = P_thispredict - K_this.dot(H.T).dot(P_thispredict)  # the state estimate covariance at time t given data up to time t

        xi_nextpredict = F.dot(xi_thisupdate)
        P_nextpredict = F.dot(P_thisupdate).dot(F.T) + Q

        # store the updated estimate and intermediate computations
        state_est_vec.append(xi_nextpredict)
        state_esterror_covmat_vec.append(P_nextpredict)

        measresid_vec.append(ytilde_this)
        measresid_covmat_vec.append(S_this)
        kalmangain_vec.append(K_this)



    ## find this iteration's MLE estimates of the parameters ##
    loss = lambda x: -loglik(y_vec, state_est_vec, state_esterror_covmat_vec, x[0], x[1], x[2], x[3])
    x0 = (mu_estvec[-1], phi_estvec[-1], theta_estvec[-1], sigma2_estvec[-1])
    bnds = ((None,None), (None,None), (None,None), (0,None))
    res = optimize.minimize(loss, x0, method='SLSQP', bounds=bnds)  # TODO: put in other args

    assert res.success

    mu_estvec.append(res.x[0])
    phi_estvec.append(res.x[1])
    theta_estvec.append(res.x[2])
    sigma2_estvec.append(res.x[3])

    eval_loglik_vec.append(-res.fun)

## Display results
print('(mu, phi, theta, sigma2) = (%f, %f, %f, %f)' % (mu, phi, theta, sigma2))
print('(mu, phi, theta, sigma2)_0 = (%f, %f, %f, %f)' % (mu_guess, phi_guess, theta_guess, sigma2_guess))
print('(mu, phi, theta, sigma2)^hat = (%f, %f, %f, %f)' % (mu_estvec[-1], phi_estvec[-1], theta_estvec[-1], sigma2_estvec[-1]))
