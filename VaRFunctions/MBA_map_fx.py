from HelperFunctions.pct_change import pct_change
from VaRFunctions.MBA_simple import VaR
import numpy as np

def MBA_map_fx(df_data, investments, alpha, forex, alpha_fx, percentile=0.99, n=200):
    
    investments = np.concatenate((np.array(investments), np.array(forex)))

    alpha = np.concatenate((alpha, alpha_fx))

    df_data = pct_change(df_data, investments)

    pct_investments = [inv+'_pct' for inv in investments]
    
    var_array = np.zeros(df_data.shape[0])

    for i in range(n,df_data.shape[0]):
        var_array[i] = VaR(df_data[pct_investments][i-n:i].cov(), alpha, percentile)

    df_data[(f'MBA_map_fx_{percentile}_{n}')] = var_array

    return df_data

 
        