from HelperFunctions.HelperFunctions import *
from scipy.stats import norm
import numpy as np


# add an n
def MBA_simple(df_data, investments, alpha, percentile=0.99, n=200):

    df_data = returns(df_data, investments, alpha)

    pct_investments = [inv+'_pct' for inv in investments]
    
    var_array = np.zeros(df_data.shape[0])
    for i in range(n,df_data.shape[0]):
        var_array[i] = VaR(df_data[pct_investments][i-n:i].cov(), alpha, percentile)
    df_data[(f'MBA_simple_{percentile}_{n}')] = var_array

    return df_data

def VaR(sigma, alpha, percentile=0.99):

    sigma_p = sigma_p_func(sigma, alpha)
    
    return sigma_p*norm.ppf(percentile)