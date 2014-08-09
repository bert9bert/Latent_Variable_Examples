'''
Example of using a Kalman filter to perform maximum likelihood estimation of the
parameters of an ARMA(1,1) process.
'''

import numpy as np
from scipy import optimize
from statsmodels.tsa import arima_model
import time

### Define program parameters ###
MAX_MLE_ITER = 1e4


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



### Define function that will perform Kalman filtering using the given
### parameters, and then output the resulting state estimate, 
### state estimate covariance, and (most importantly) the log likelihood
def kalman_arma11(y_vec, mu, phi, theta, sigma2):
    T = len(y_vec)

    ## Define components of the state space system -- based off Hamilton (1994)
    F = np.array([[phi, 0], [1, 0]])
    Q = np.array([[sigma2, 0], [0, 0]])
    A = np.array([[mu]])
    x_this = np.array([[1]])
    H = np.array([[1], [theta]])
    R = [[0]]

    ## unconditional mean and variance of first state to start Kalman filter with
    state_est_0 = np.array([[0], [0]])
    state_esterror_covmat_0 = np.array([[sigma2/(1-phi**2), phi*sigma2/(1-phi**2)], 
        [phi*sigma2/(1-phi**2), sigma2/(1-phi**2)]])

    ## perform Kalman filter ##
    state_est_vec = [state_est_0]
    state_esterror_covmat_vec = [state_esterror_covmat_0]

    ytilde_vec = []
    S_vec = []
    kalmangain_vec = []
    
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

        ytilde_vec.append(ytilde_this)
        S_vec.append(S_this)
        kalmangain_vec.append(K_this)

    ## compute the log likelihood ##
    loglik = -1*T/2*np.log(2*np.pi) - (1/2) * \
        sum([  np.log(np.linalg.det(S_vec[t])) + ytilde_vec[t].T.dot(np.linalg.inv(S_vec[t])).dot(ytilde_vec[t])[0][0] for t in range(T) ])

    ## return results ##
    return {'state_est_vec': state_est_vec, 
        'state_esterror_covmat_vec': state_esterror_covmat_vec, 
        'loglik': loglik}


### Estimate parameters by MLE using Kalman filtering ###
## Initial parameter guesses ##
mu_guess     = 9
phi_guess    = 0.6
theta_guess  = 0.5
sigma2_guess = 2.1

## Perform MLE optimization
loss = lambda x: -kalman_arma11(y_vec, x[0], x[1], x[2], x[3])['loglik']
x0 = (mu_guess, phi_guess, theta_guess, sigma2_guess)
bnds = ((None,None), (None,None), (None,None), (0,None))

time_mymle_start = time.time()

res = optimize.minimize(loss, x0, method='SLSQP', bounds=bnds, options={'maxiter': MAX_MLE_ITER})

time_mymle_end = time.time()
time_mymle = time_mymle_end - time_mymle_start

assert res.success

mu_hat     = res.x[0]
phi_hat    = res.x[1]
theta_hat  = res.x[2]
sigma2_hat = res.x[3]

### Now, for comparison purposes, estimate using the StatsModels package ###
sm = arima_model.ARMA(y_vec, order=(1,1))

time_sm_start = time.time()

fittedsm = sm.fit(disp=0)

time_sm_end = time.time()
time_sm = time_sm_end - time_sm_start

### Display results ###
print('(mu, phi, theta, sigma2) = (%f, %f, %f, %f)' % (mu, phi, theta, sigma2))
print('(mu, phi, theta, sigma2)_0 = (%f, %f, %f, %f)' % (mu_guess, phi_guess, theta_guess, sigma2_guess))
print('(mu, phi, theta, sigma2)^hat = (%f, %f, %f, %f)' % (mu_hat, phi_hat, theta_hat, sigma2_hat))
print('(mu, phi, theta, sigma2)^hatSM = (%f, %f, %f, %f)' % (fittedsm.params[0], fittedsm.params[1], fittedsm.params[2], fittedsm.sigma2))
print('Time My MLE: %f, Time StatsModels: %f' % (time_mymle, time_sm))

