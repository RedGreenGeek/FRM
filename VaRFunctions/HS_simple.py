from HelperFunctions.HelperFunctions import returns
import numpy as np

def HS_simple(df_data, investments, alpha, percentile=0.99, n=200):
 
    USD_investments = [inv+'_USD' if '.DE' in inv or '.L' in inv else inv for inv in investments ]

    df_data = returns(df_data, USD_investments, alpha)

    var_array = np.zeros(df_data.shape[0])

    for i in range(n,df_data.shape[0]):
        var_array[i] = HS_simple_VaR(df_data['returns'][i-n:i], percentile=percentile, n=n)
        
    df_data[(f'HS_simple_{percentile}_{n}')] = var_array

    return df_data

def HS_simple_VaR(returns_df, percentile=0.99, n=200):
    percentile = 1-percentile
    return -returns_df.iloc[-n:].quantile(q=percentile)