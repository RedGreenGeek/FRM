import numpy as np
from HelperFunctions.HelperFunctions import *
from scipy.stats import norm


def MBA_simple(df_returns, alpha, percentile=0.99, n=250):
    var_array = np.zeros(df_returns.shape[0])
    es_array = np.zeros(df_returns.shape[0])
    
    for i in range(n, df_returns.shape[0]+1):
        cov_matrix = df_returns[i-n:i].cov()
        var_array[i-1] = VaR(cov_matrix, alpha, percentile)
        es_array[i-1] = expected_shortfall(cov_matrix, alpha, percentile)

    return var_array, es_array


def VaR(sigma, alpha, percentile=0.99):

    sigma_p = sigma_p_func(sigma, alpha)
    
    return sigma_p*norm.ppf(percentile)

def expected_shortfall(sigma, alpha, percentile=0.99):
    
    sigma_p = sigma_p_func(sigma, alpha)
    
    return sigma_p * norm.pdf(norm.ppf(percentile))/(1-percentile)
