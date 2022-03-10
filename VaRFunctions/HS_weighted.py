from HelperFunctions.HelperFunctions import *
import numpy as np

def HS_weighted(loss, percentile=0.99, n=200, lam=0.995):
    var_array = np.empty(loss.shape[0])
    data = np.empty((n,3))
    data[:, 1] = exp_weight(n, lam)  

    data = data[(data[:,0]).argsort()]
    data[:, 2] = np.cumsum(data[:,1])

    for i in range(n, loss.shape[0]):
        data[:, 0] = loss[i-n:i]
        data = data[(data[:,0]).argsort()]
        data[:, 2] = np.cumsum(data[:,1])
        idx_exceed = np.argmin(data[:,2] <= (1-percentile))
        idx_lower = np.arange(idx_exceed+1)
        var_array[i] = interpt_2_pts(percentile, data[idx_lower[-2:], 2], data[idx_lower[-2:], 0])
    
    return var_array




# def HS_weighted(df_data, investments, alpha, lam=0.995, percentile=0.99, n=200):
    
#     USD_investments = [inv+'_USD' if '.DE' in inv or '.L' in inv else inv for inv in investments ]

#     df_data = returns(df_data, USD_investments, alpha)

#     var_array = np.zeros(df_data.shape[0])

#     for i in range(n,df_data.shape[0]):
#         var_array[i] = HS_weighted_VaR(df_data['returns'][i-n:i], lam=lam, percentile=percentile, n=n)
        
#     df_data[(f'HS_weighted_{percentile}_{n}_{lam}')] = var_array

#     return df_data

def HS_weighted_VaR(returns, n=200, lam = 0.995, percentile=0.99):
    
    data = np.empty((n,3))
    data[:, 0] = returns[-n:]
    data[:, 1] = exp_weight(n, lam)  

    data = data[(data[:,0]).argsort()]
    data[:, 2] = np.cumsum(data[:,1])

    idx_exceed = np.argmin(data[:,2] <= (1-percentile))
    idx_lower = np.arange(idx_exceed+1)

    return -interpt_2_pts(1-percentile, data[idx_lower[-2:], 2], data[idx_lower[-2:], 0])