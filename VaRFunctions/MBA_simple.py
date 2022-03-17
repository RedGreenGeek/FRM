from HelperFunctions.HelperFunctions import *
from scipy.stats import norm
import numpy as np

def MBA_simple(df_returns, alpha, percentile=0.99, n=250):
    var_array = np.zeros(df_returns.shape[0])
    for i in range(n, df_returns.shape[0]):
        var_array[i] = VaR(df_returns[i-n:i].cov(), alpha, percentile)

    return var_array


def VaR(sigma, alpha, percentile=0.99):

    sigma_p = sigma_p_func(sigma, alpha)
    
    return sigma_p*norm.ppf(percentile)