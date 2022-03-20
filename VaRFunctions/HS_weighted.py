from HelperFunctions.HelperFunctions import *
import numpy as np

def HS_weighted(loss, percentile=0.99, n=250, lam=0.995):
    var_array = np.empty(loss.shape[0])
    es_array = np.empty(loss.shape[0])
    data = np.empty((n,3))

    for i in range(n, loss.shape[0]+1):
        data[:, 0] = loss[i-n:i].copy()
        data[:, 1] = exp_weight(n, lam)
        data = data[(data[:,0]).argsort()]
        data[:, 2] = np.flip(np.cumsum(np.flip(data[:,1])))
        # idx_exceed = np.argmin(data[:,2] <= (percentile))
        idx_exceed = np.argmin(data[:,2] > (1-percentile))
        idx_interpol = [idx_exceed-1, idx_exceed]
        var_array[i-1] = interpt_2_pts(1-percentile, data[idx_interpol, 2], data[idx_interpol, 0])
        es_array[i-1] = np.average(data[(idx_exceed):, 0], weights=data[(idx_exceed):, 1])
    
    return var_array, es_array

# def es_weighted_calc(input_data, idx_exceed, percentile): 
#     weights = input_data[(idx_exceed-1):, 1].copy()
#     weights[0] -=  (1-percentile) 
#     weights /= (1-percentile) 
    
#     return np.average(input_data[(idx_exceed-1):, 0], weights=weights)




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