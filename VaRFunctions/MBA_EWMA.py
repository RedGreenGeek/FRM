from HelperFunctions.HelperFunctions import returns
from VaRFunctions.MBA_simple import VaR
import numpy as np


def MBA_EWMA(df_data, investments, alpha, percentile=0.99, n=200, lam=0.95, sigma_init=None):

    USD_investments = [inv+'_USD' if '.DE' in inv or '.L' in inv else inv for inv in investments ]

    df_data = returns(df_data, USD_investments, alpha)

    pct_investments = [inv+'_pct' for inv in USD_investments]
    
    var_array = np.zeros(df_data.shape[0])

    for i in range(n,df_data.shape[0]):
        var_array[i] = VaR(ewma(df_data[pct_investments][i-n:i],lam), alpha, percentile)

    df_data[(f'MBA_EWMA_{percentile}_{n}_{lam}')] = var_array

    return df_data


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