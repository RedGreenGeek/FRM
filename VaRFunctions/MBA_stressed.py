import numpy as np
from VaRFunctions.MBA_simple import VaR, expected_shortfall

def MBA_stressed(df_returns,df_subset, alpha, stressed_range, investments, percentile=0.99, n=250):
    var_array = np.zeros(df_subset.shape[0])
    es_array = np.zeros(df_subset.shape[0])
    
    cov_matrix = df_returns.loc[stressed_range[0]:stressed_range[1], investments].cov()
    for i in range(n, df_subset.shape[0]+1):
        var_array[i-1] = VaR(cov_matrix, alpha, percentile)
        es_array[i-1] = expected_shortfall(cov_matrix, alpha, percentile)

    return var_array, es_array

