from HelperFunctions.HelperFunctions import *
from VaRFunctions.MBA_simple import *
import numpy as np


def MBA_EWMA(df_data, investments, alpha, percentile=0.99, n=200, lam=0.95, sigma_init=None):

    df_data = returns(df_data, investments, alpha)

    pct_investments = [inv+'_pct' for inv in investments]
    
    var_array = np.zeros(df_data.shape[0])
    for i in range(n,df_data.shape[0]):
        var_array[i] = VaR(ewma(df_data[pct_investments][i-n:i],lam), alpha, percentile)
    df_data[(f'MBA_EWMA_{percentile}_{n}_{lam}')] = var_array

    return df_data


def ewma(r_df, lam, sigma_init=None):

    # convert to numpy
    r_df_ndarry = r_df.values

    # assume initialize with matrix identity if nothing is specified
    if sigma_init is None:
        sigma = np.eye(r_df.shape[1])
    else:
        sigma = sigma_init

    for returns in r_df_ndarry:
        sigma = sigma * lam + (1 - lam) * np.outer(returns, returns)

    return sigma