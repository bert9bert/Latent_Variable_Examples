'''
Example of representing the position of a truck moving on a straight road
with noisy zero mean acceleration and noisy GPS measurement as a state space
system, and estimating it with a Kalman filter.
'''


import numpy as np
from scipy import linalg
import pylab

### Parameters ###
N = 200

### Define natural process ###
delta_t = 1
variance_accel = 0.05
variance_meas  = 500

F = np.array([[1, delta_t], [0, 1]])
G = np.array([delta_t^2, delta_t])
Q = np.outer(G, G.T)*variance_accel

H = np.array([1, 0])
R = np.array(variance_meas)

### Initial values ###
# initialize actual starting state
x0 = np.array([0, 1])

# initial starting state estimate
xhat_0_given_0 = np.array([0, 1])

# covariance matrix of starting state estimate
P_0_given_0 = np.array([[0,0], [0,0]])


### Draw state and measurement data ###
# draw acceleration

# draw measurement error
noise_state = np.random.multivariate_normal(np.array([0,0]), Q, N)
noise_state[0] = np.array([0, 0])

noise_meas  = np.random.normal(0, np.sqrt(variance_meas), N)

# create state time series
state_ts = np.empty((N,2))
state_ts[0] = x0

meas_ts = np.empty(N)
meas_ts[0] = noise_meas[0]

for i in range(1,N):
    state_ts[i] = F.dot(state_ts[i-1]) + noise_state[i]
    meas_ts[i] = H.dot(state_ts[i]) + noise_meas[i]


### Estimate state time series with dead reckoning ###
xhat_dr_k = np.empty((N,2))
xhat_dr_k[0] = xhat_0_given_0

for k in range(1,N):
    xhat_dr_k[k] = F.dot(xhat_dr_k[k-1])

### Estimate state time series with Kalman filter ###
xhat_k_given_k = np.empty((N,2))
P_k_given_k = np.empty((N,2,2))

xhat_k_given_k[0] = xhat_0_given_0
P_k_given_k[0] = P_0_given_0

for k in range(1,N):
    ## Predict steps ##
    # predict the next state with the current data and get this prediction's cov mat
    xhat_k_given_klag1 = F.dot(xhat_k_given_k[k-1])
    P_k_given_klag1 = F.dot(P_k_given_k[k-1]).dot(F.T) + Q

    ## Update steps ##
    # get measurement residual estimate and cov mat
    meas_resid_estimate = meas_ts[k] - H.dot(xhat_k_given_klag1)
    meas_resid_covmat = H.dot(P_k_given_klag1).dot(H.T) + R

    # compute the optimal Kalman gain
    meas_resid_covmat_inv = 1/meas_resid_covmat  # since this is actually a scalar
    optimal_kalman_gain = P_k_given_klag1.dot(H.T).dot(meas_resid_covmat_inv)

    # update the state estimate and the cov mat of the state estimate with the computed gain
    xhat_k_given_k[k] = xhat_k_given_klag1 + optimal_kalman_gain*meas_resid_estimate
    P_k_given_k[k] = (1 - optimal_kalman_gain.dot(H))*(P_k_given_klag1)  # works b/c 1-d

### Plot results ###
pylab.plot(range(N), state_ts[:,0], 'r-x', label='Actual Position')
pylab.plot(range(N), meas_ts, 'g-x', label='GPS Estimate')
pylab.plot(range(N), xhat_dr_k[:,0], 'y-x', label='Dead Reckoning Estimate')
pylab.plot(range(N), xhat_k_given_k[:,0], 'b-x', label='Kalman Filter Estimate')

pylab.title('Estimating where my truck is')

pylab.xlabel('Elapsed Time')
pylab.ylabel('Distance from Mile Marker 0')

pylab.legend(loc='best')

pylab.show()
