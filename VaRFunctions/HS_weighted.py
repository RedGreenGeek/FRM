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


def HS_weighted_VaR(returns, n=200, lam = 0.995, percentile=0.99):
    
    data = np.empty((n,3))
    data[:, 0] = returns[-n:]
    data[:, 1] = exp_weight(n, lam)  

    data = data[(data[:,0]).argsort()]
    data[:, 2] = np.cumsum(data[:,1])

    idx_exceed = np.argmin(data[:,2] <= (1-percentile))
    idx_lower = np.arange(idx_exceed+1)

    return -interpt_2_pts(1-percentile, data[idx_lower[-2:], 2], data[idx_lower[-2:], 0])