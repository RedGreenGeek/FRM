from HelperFunctions.HelperFunctions import returns
from VaRFunctions.MBA_simple import VaR
import numpy as np


def MBA_EWMA(df_returns, alpha, percentile=0.99, n=250, lam=0.95, sigma_init=None):
    var_array = np.zeros(df_returns.shape[0])
    for i in range(n, df_returns.shape[0]):
        cov_emwa = ewma(df_returns[i-n:i], lam)
        var_array[i] = VaR(cov_emwa, alpha, percentile)

    return var_array


# def MBA_EWMA(df_data, investments, alpha, percentile=0.99, n=200, lam=0.95, sigma_init=None):


#     for i in range(n,df_data.shape[0]):
#         var_array[i] = VaR(ewma(df_data[pct_investments][i-n:i],lam), alpha, percentile)

#     df_data[(f'MBA_EWMA_{percentile}_{n}_{lam}')] = var_array

#     return df_data


def ewma(df_pct_data, lam, sigma_init=None):

    # convert to numpy
    df_pct_data_nparray = df_pct_data.values

    # assume initialize with matrix identity if nothing is specified
    if sigma_init is None:
        sigma = np.eye(df_pct_data.shape[1])
    else:
        sigma = sigma_init

    for returns in df_pct_data_nparray:
        sigma = sigma * lam + (1 - lam) * np.outer(returns, returns)

    return sigma