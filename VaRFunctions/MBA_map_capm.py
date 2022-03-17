from HelperFunctions.pct_change import pct_change
from VaRFunctions import VaR
import numpy as np

from scipy.stats import norm
import numpy as np


def MBA_map_capm(df_data, investments, alpha, forex, alpha_fx, percentile=0.99, n=200):
    
    df_data = pct_change(df_data, investments)

    # Should be made dynamic
    markets = [ '^GDAXI', '^GDAXI', '^GDAXI', '^FTSE','^FTSE','^FTSE', '^DJI', '^DJI']
    df_data = pct_change(df_data, markets)

    df_data = pct_change(df_data, forex)

    pct_investments = [inv+'_pct' for inv in investments]
    pct_markets = [inv+'_pct' for inv in markets]

    c = np.zeros((df_data.shape[0], len(markets)))   
    for i in range(n+1,df_data.shape[0]):
        for j in range(len(markets)):
            x = df_data[pct_markets[j]].values[i-n:i]
            y = df_data[pct_investments[j]].values[i-n:i]
            A = np.vstack([x, np.ones(len(x))]).T
            c[i,j] = np.linalg.lstsq(A, y, rcond=None)[0][0]

    m = ['^GDAXI', '^FTSE', '^DJI']
    for mar in m:
        df_data[(f'{mar}_c')] = np.zeros(df_data.shape[0])

    for j in range(len(markets)):
        df_data[(f'{markets[j]}_c')] += c[:,j]

    for mar in m:
        df_data[(f'{mar}_c_pct')] = df_data[(f'{mar}_c')] * df_data[(f'{mar}_pct')]

    # Should be made dynamic
    alpha_sx = [1000, 1000, 1000]

    alpha_sx = np.concatenate((alpha_sx, alpha_fx))

    c_pct_markets = [mar+'_c_pct' for mar in m]
    c_pct_markets = np.concatenate((c_pct_markets, forex))
    
    var_array = np.zeros(df_data.shape[0])

    for i in range(n,df_data.shape[0]):
        var_array[i] = VaR(df_data[c_pct_markets][i-n:i].cov(), alpha_sx, percentile)

    df_data[(f'MBA_map_capm_{percentile}_{n}')] = var_array

    return df_data
 
        