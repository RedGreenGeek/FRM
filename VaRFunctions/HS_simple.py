from math import ceil
from HelperFunctions.HelperFunctions import returns
import numpy as np

def HS_simple(df_data, investments, alpha, percentile=0.99, n=250):
 
    USD_investments = [inv+'_USD' if '.DE' in inv or '.L' in inv else inv for inv in investments ]

    df_data = returns(df_data, USD_investments, alpha)

    var_array = np.zeros(df_data.shape[0])
    
    df_data['losses'] = -1*df_data['returns']
    for i in range(n,df_data.shape[0]):
        var_array[i] = HS_simple_VaR(df_data['returns'][i-n:i], percentile=percentile, n=n)
        
    df_data[(f'HS_simple_{percentile}_{n}')] = var_array

    return df_data

def HS_simple_VaR(returns_df, percentile=0.99, n=200):
    idx = ceil(n - n*percentile)-1

    return -np.sort(returns_df.iloc[-n:].values)[idx]
    
def HS_simple(loss, percentile=0.99, n=200):
    var_array = np.empty(loss.shape[0])
    for i in range(n, loss.shape[0]): 
        loss_sort = np.sort(loss[i-n:i])
        var_array[i] = loss_sort[int(n*percentile)]
    
    return var_array