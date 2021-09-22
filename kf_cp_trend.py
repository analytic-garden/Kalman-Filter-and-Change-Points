#!/home/bill/anaconda3/envs/pCO2/bin/python
# -*- coding: utf-8 -*-
"""
kp_cp_trnd.py
        A python program implementing a Bayesian change point routine
        based on a Local Linear Trend Kalman Filter.
        
        See Bayesian inference on biopolymer models by Liu and Lawrence and
        https://analyticgarden.blogspot.com/2018/01/changes.html for
        change point details.

@author: Bill Thompson
@license: GPL 3
@copyright: 2021_09_03
"""
import sys
import argparse
import time
import random
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.special as sp
import statsmodels.api as sm
import warnings

# from https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_local_linear_trend.html
# see also Dynmaic Linear Models with R chapter 2

# Create a new class with parent sm.tsa.statespace.MLEModel
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    # Define the initial parameter vector; see update() below for a note
    # on the required order of parameter values in the vector
    start_params = [1.0, 1.0]
    
    # We require a single instantiation argument, the
    # observed dataset, called `endog` here.
    def __init__(self, endog):
        k_states = k_posdef = 2
         
        super(LocalLinearTrend, self).__init__(endog, 
                                         k_states = k_states,
                                         k_posdef = k_posdef,
                                         initialization='approximate_diffuse',
                                         loglikelihood_burn=k_states)
        
        # Specify the fixed elements of the state space matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                           [0, 1]])
        self.ssm['selection'] = np.eye(k_states)
        
        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)
                
    # Here we define how to update the state space matrices with the
    # parameters. Note that we must include the **kwargs argument
    def update(self, params, *args, **kwargs):
        # Using the parameters in a specific order in the update method
        # implicitly defines the required order of parameters
        
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        
        self.ssm['obs_cov', 0, 0] = params[0]
        self.ssm[self._state_cov_idx] = params[1:]

def GetArgs():
    """
    GetArgs - return arguments from command line. 
    Use -h to see all arguments and their defaults.

    Returns
    args - Parameter values.
    -------
    TYPE
        A Parser object.
    """
    def ParseArgs(parser):
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write('error: %s\n' % message)
                self.print_help()
                sys.exit(2)

        parser = Parser(description='Bayesian Change Point Kalman filter')
        parser.add_argument('input_file',
                            help="""
                            Input file (required).
                            Must contain columns Year and Temperature_Anomaly
                            """)
        parser.add_argument('-k', '--max_num_segments',
                            required = False,
                            default = 10,
                            type=int,
                            help='Maximum number of segments (default = 10)')
        parser.add_argument('-n', '--num_samples',
                            required=False,
                            type=int,
                            default = 1000,
                            help='Number of sample iterations. Default=1000')
        parser.add_argument('-s', '--rand_seed',
                            required=False,
                            type=int,
                            help='Random mumber generator seed.')
        
        return parser.parse_args()

    parser = argparse.ArgumentParser(description='Bayesain Poisson Change Point Data')
    args = ParseArgs(parser)

    return args

def normalize(p):
    """
    normalize
        normalize a numpy array of probabilities.

    Parameters
    ----------
    p : numpy array
        A numpty array or list of probability values.

    Returns
    -------
    numpy array
        A normaized numpy array. np.sum(array) = 1.
        
    Requires
    --------
    A values in p must be numeric and >= 0.

    """
    if np.sum(p) == 0:
        return np.array([0])
    return np.array(p)/np.sum(p)

def optimize_params(model):
    """
    optimize_params - find optimal variances for Kalman filter.

    Parameters
    ----------
    model : a LocalLinearTrend object

    Returns
    -------
    A scipy.optimize.minimize object
    
    Note
    ----
    Unknown variances are parameterized by their logs to avoind negative values.

    """       
    def neg_loglike(params):
        return -model.loglike(np.exp(params))        
    
    output = minimize(neg_loglike, 
                      # model.start_params, 
                      [ 0.01, 0.01, 0.01], 
                      # [0.006553403, 0.009242106, 0.00597295], # from trend_gibbs.R
                      method='Nelder-Mead')
    
    return output

def calcSegments(Y, kmax):
    """
    calcSegments - Calculate the likelihood of a Kalman filter for each
    possible segement of Y.

    Parameters
    ----------
    Y : numpy array
        A sequence of values, a time series.
    kmax : int
        Number of segments.

    Returns
    -------
    S : numpy array
        S[k, i,j - likelihood of Y[i:j] containing k segments.
    pkdata : numpy array
        P(k|Y) - probability of k segments.
    obs_vars : numpy array
        obs_var[i:j] Saved observation varainace for Kalnam filter of Y[i:j].
    state_vars_1 : numpy array
        state_vars_1[i:j] Saved state varaince 1 (mu)for Kalnam filter of Y[i:j].
    state_vars_2 : numpy array
        state_vars_2[i:j] Saved state varaince 2 (slope) for Kalnam filter of Y[i:j].
        
    Note
    ----
    This code is horrible. I apologize.
    First of all, S is terribly inefficent and wastes a ton of memory.
    It will be fixed later.
    
    Also, maximim likelihood of the parameters is the wrong thing to do. 
    I only use it because I don't know of any way to integrate out 
    the parameters of P(Y[i:j] | epsilon, eta_1, eta_2) to get P(Y[i:j]).
    
    This function returns too many objects. A data class should be made
    and return an object of that class.
    """
    S = np.zeros((kmax, len(Y), len(Y)))
    obs_vars = np.zeros((len(Y), len(Y)))
    state_vars_1 = np.zeros((len(Y), len(Y)))
    state_vars_2 = np.zeros((len(Y), len(Y)))
    
    # calculate P(x[i:j],k=0) for all possible subsequences
    for i in range(len(Y)):
        print('i =', i)
        # for j in range(i, len(Y)):
        for j in range(i+1, len(Y)):  # this loop could be run in parallel
            kf = LocalLinearTrend(Y[i:j+1])
            output = optimize_params(kf)
            ll = - output.fun
            S[0,i,j] = np.exp(ll)
            obs_vars[i, j] = np.exp(output.x[0])
            state_vars_1[i, j] = np.exp(output.x[1])
            state_vars_2[i, j] = np.exp(output.x[2])

    # calculate P[x[:j]|k) for k > 0 for all possible ending positions with k-1 change points
    for k in range(1, kmax):
        for j in range(len(Y)):
            s = 0
            for v in range(j):
                s += S[k-1, 0, v] * S[0, v+1, j]
            S[k, 0, j] = s

    pkdata = np.zeros(kmax)    
    p_prior = np.zeros(kmax)  # prior for k
    p_prior[0] = 0.5
    p_prior[1:kmax] = np.array([0.5/(k+1) for k in range(1, kmax)])
    
    for k in range(kmax):
        pkdata[k] = ((S[k, 0, len(Y)-1]/kmax) / sp.comb(len(Y), k)) * p_prior[k]
         
    pkdata = normalize(pkdata)

    return S, pkdata, obs_vars, state_vars_1, state_vars_2

def backsample(Y, 
               S, pkdata, 
               obs_vars, state_vars_1, state_vars_2,
               kmax,
               samples = 1000,
               tol = 1e-12):
    """
    backsample - sample change points from S and save data.
    See Liu and Lawrence and 
    https://analyticgarden.blogspot.com/2018/01/changes.html for details.

    Parameters
    ----------
    Y : numpy array
        Original timeseries data.
    S : numpy array
        Segment array returned from calcSegments.
    pkdata : numpy array
        P(k | Y) returned from calcSegments.
    obs_vars : numpy array
        Saved observation variance from calcSegments.
    state_vars_1 : numpy array
        Saved state variance from calcSegments.
    state_vars_2 : numpy array
        Saved state variance from calcSegments.
    kmax : int
        Number of segments.
    samples : int, optional
        Number of samples. The default is 1000.
    tol : float, optional
        Tolerance to see if sum of probs is really 1. The default is 1e-12.

    Returns
    -------
    k_samples : numpy array
        Count of samples for each value of k.
    pos_samples : numpy array
        Count of samples for each position in Y.
    obs_var_samples : numpy array
        Values of variance from obs_vars at sampled position.
    state_var_samples_1 : numpy array
        Values of variance from state_vars_1 at sampled position.
    state_var_samples_2 : numpy array
        Values of variance from state_vars_2 at sampled position.
    mu_samples : numpy array
        Values of filter mu at sampled position.

    """
    print('Backsampling...')
    
    length = len(Y)
    k_samples = np.zeros(kmax)
    pos_samples = np.zeros(length)
    obs_var_samples = np.zeros((samples, length))
    state_var_samples_1 = np.zeros((samples, length))
    state_var_samples_2 = np.zeros((samples, length))
    mu_samples = np.zeros((samples, length))
    trend_samples = np.zeros((samples, length))
    
    # Get the full length filter result to save time.
    kf_full = LocalLinearTrend(Y)
    output_full = optimize_params(kf_full)
    filtered_full = kf_full.filter(np.exp(output_full.x))

    for i in range(samples):
        k = int(np.random.choice(kmax, size=1, replace=True, p=pkdata)) # sample number of change points                
        k_samples[k] += 1
        L = length
        if k == 0:  # only one segment - the whole sequence
            pos_samples[L-1] += 1
            obs_var_samples [i, 0:L] = np.exp(output_full.x[0])
            state_var_samples_1 [i, 0:L] = np.exp(output_full.x[1])
            state_var_samples_2 [i, 0:L] = np.exp(output_full.x[2])
            mu_samples[i, 0:L] = filtered_full.filtered_state[0]
            trend_samples[i, 0:L] = filtered_full.filtered_state[1]
        else:  # sample backwards from end of sequence
            for l in range(int(k-1), -1, -1):
                probs =  np.array([S[l, 0, v-1] * S[0, v, L-1] for v in range(1,L)])
                
                normalized = normalize(probs)
                if np.abs(1.0 - np.sum(normalized)) > tol:
                    break
                
                v = int(np.random.choice(len(probs), size=1, replace=True, p=normalized))
                kf = LocalLinearTrend(Y[v:L])
                obs_var_samples[i, v:L] = obs_vars[v, L-1]
                state_var_samples_1[i, v:L] = state_vars_1[v, L-1]
                state_var_samples_2[i, v:L] = state_vars_2[v, L-1]
                filtered = kf.filter(np.exp([obs_vars[v, L-1], 
                                             state_vars_1[v, L-1],
                                             state_vars_2[v, L-1]]))
                mu_samples[i, v:L] = filtered.filtered_state[0]
                trend_samples[i, v:L] = filtered.filtered_state[1]
                # L = v-1
                L = v
                pos_samples[v] += 1
                if L <= 1:
                    break
        # if L > 1:
        if L > 2:
            kf = LocalLinearTrend(Y[0:L])
            obs_var_samples[i, 0:L] = obs_vars[0, L-1]
            state_var_samples_1[i, 0:L] = state_vars_1[0, L-1]
            state_var_samples_2[i, 0:L] = state_vars_2[0, L-1]
            filtered = kf.filter(np.exp([obs_vars[0, L-1], 
                                         state_vars_1[0, L-1],
                                         state_vars_2[0, L-1]]))
            mu_samples[i, 0:L] = filtered.filtered_state[0]
            trend_samples[i, 0:L] = filtered.filtered_state[1]

    return k_samples, pos_samples, \
            obs_var_samples, \
            state_var_samples_1, state_var_samples_2, \
            mu_samples, trend_samples

def plot_samples(X, Y,
                 k_samples, pos_samples, 
                 obs_var_samples, 
                 state_var_samples_1, state_var_samples_2,
                 mu_samples, trend_samples,
                 kmax = 10,
                 samples = 1000):
    """
    plot_samples - make some plots.

    Parameters
    ----------
    X : numpy array
        Times for x-axis.
    Y : numpy array
        Timeseries data.
    k_samples : numpy array
        Number of times each k value is sampled
    pos_samples : numpy array
        Samples of breakpoints for each position.
    obs_var_samples : numpy array
        Values of kalman filter observation variance sampled at each point.
    state_var_samples_1 : numpy array
        Values of kalman filter state variance 1 sampled at each point.
    state_var_samples_2 : numpy array
        Values of kalman filter state variance 1 sampled at each point.
    mu_samples : numpy array
        Values of kalman filter mu values sampled at each point.
    trend_samples : numpy array
        Values of kalman filter trend values sampled at each point.
    kmax : int, optional
        Maximum number of segments. The default is 10.
    samples : int, optional
        Number of backsamples. The default is 1000.

    Returns
    -------
    None.

    """
    size = 2
    
    # plot data
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'o', markersize = size)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly')
    plt.show(block = False)

    # probability of each change point
    fig, ax = plt.subplots()
    ax.stem(list(range(kmax)), k_samples / samples)
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Probability of K Change Points')
    plt.show(block = False)
        
    # probability of change point at each position
    fig, ax = plt.subplots()
    ax.stem(X, normalize(pos_samples))
    plt.xlabel('Year')
    plt.ylabel('Probability of Change Point')
    plt.title('Change Point Probability')
    plt.show(block = False)
    
    std_color = 'darkgrey'
    
    # plot varianaces
    mean_obs_var = np.mean(obs_var_samples, axis = 0)
    std_obs_var = np.std(obs_var_samples, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(X, mean_obs_var, label = r'$\sigma _e^2$')
    ax.fill_between(X, 
                    mean_obs_var - 2 * std_obs_var, 
                    mean_obs_var + 2 * std_obs_var, 
                    color = std_color,
                    label = '+/- 2 std')
    ax.set_ylabel(r'$\sigma _e^2$')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\sigma _e ^2$')
    ax.set_title('Mean Sampled Obs. Variance')
    plt.show(block = False)
    
    mean_state_var_1 = np.mean(state_var_samples_1, axis = 0)
    std_state_var_1 = np.std(state_var_samples_1, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(X, mean_state_var_1, label = r'$\sigma _\eta ^2$')
    ax.fill_between(X, 
                    mean_state_var_1 - 2 * std_state_var_1, 
                    mean_state_var_1 + 2 * std_state_var_1, 
                    color = std_color,
                    label = '+/- 2 std')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\sigma _{\eta_1} ^2$')
    ax.set_title('Mean Sampled State Variance 1')
    plt.show(block = False)
    
    mean_state_var_2 = np.mean(state_var_samples_2, axis = 0)
    std_state_var_2 = np.std(state_var_samples_2, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(X, mean_state_var_2, label = r'$\sigma _\eta_1 ^2$')
    ax.fill_between(X, 
                    mean_state_var_2 - 2 * std_state_var_2, 
                    mean_state_var_2 + 2 * std_state_var_2, 
                    color = std_color,
                    label = '+/- 2 std')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\sigma _{\eta_2} ^2$')
    ax.set_title('Mean Sampled State Variance 2')
    plt.show(block = False)
    
    mean_mu = np.mean(mu_samples, axis = 0)
    std_mu = np.std(mu_samples, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'o', markersize = size, label = 'Data')
    ax.plot(X, mean_mu, '-o', 
            markersize = size,
            label = 'Predicted Temperature Anomaly')
    ax.fill_between(X, 
                    mean_mu - 2 * std_mu, 
                    mean_mu + 2 * std_mu, 
                    color = std_color,
                    label = '+/- 2 std')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Temperature Anomaly $^\degree$ C')
    ax.set_title('Mean Sampled Estimate')
    plt.show(block = False)

    mean_trend = np.mean(trend_samples, axis = 0)
    std_trend = np.std(trend_samples, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(X, mean_trend, '-o', 
            markersize = size,
            label = 'Predicted Temperature Anomaly')
    ax.fill_between(X, 
                    mean_trend - 2 * std_trend, 
                    mean_trend + 2 * std_trend, 
                    color = std_color,
                    label = '+/- 2 std')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Trend')
    ax.set_title('Mean Sampled Trend Estimate')
    plt.show(block = False)
    
def main():
    warnings.filterwarnings("error")
    
    args = GetArgs()
    data_file = args.input_file
    kmax = args.max_num_segments
    samples = args.num_samples
    rand_seed = args.rand_seed
    
    if rand_seed is None:
        rand_seed = int(time.time())
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    # data_file = '/mnt/d/Documents/pCO2/flickering_switch/kf_change_point/data/HadCRUT.4.6.0.0.annual_ns_avg.csv'   
    # data_file = '/mnt/d/Documents/pCO2/flickering_switch/kf_change_point/data/test2.csv'
    # data_file = '/mnt/d/Documents/pCO2/flickering_switch/kf_change_point/data/test3.csv'
    # data_file = '/mnt/d/Documents/pCO2/flickering_switch/kf_change_point/data/posterior_data.csv'   
    
    rand_seed = int(time.time())
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    print('seed = ', rand_seed)

    df = pd.read_csv(data_file)
    
    X = np.array(df['Year'])
    Y = np.array(df['Temperature_Anomaly'])
    
    # sample probability of segments
    S, pkdata, obs_vars, state_vars_1, state_vars_2 = calcSegments(Y, kmax)
    
    # backsample change points
    k_samples, pos_samples, \
    obs_var_samples, \
    state_var_samples_1, state_var_samples_2, \
    mu_samples, trend_samples = \
        backsample(Y, 
                   S, pkdata, 
                   obs_vars, state_vars_1, state_vars_2,
                   kmax, 
                   samples = samples)
    
    # plot them
    plot_samples(X, Y,
                 k_samples, pos_samples, 
                 obs_var_samples, 
                 state_var_samples_1, state_var_samples_2,
                 mu_samples, trend_samples,
                 kmax,
                 samples)
    
    input('Press Enter to Exit...')

if __name__ == '__main__':
    main()