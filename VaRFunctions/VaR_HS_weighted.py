import numpy as np
from SupportFunctions.exp_weight import exp_weight
from SupportFunctions.interpt_2_pts import interpt_2_pts


def VaR_HS_weighted(returns, n=500, lam = 0.995, percentile=0.99):
    
    data = np.empty((n,3))
    data[:, 0] = returns[-n:]
    data[:, 1] = exp_weight(n, lam)  

    data = data[(data[:,0]).argsort()]
    data[:, 2] = np.cumsum(data[:,1])

    idx_exceed = np.argmin(data[:,2] <= (1-percentile))
    idx_lower = np.arange(idx_exceed+1)

    return -interpt_2_pts(1-percentile, data[idx_lower[-2:], 2], data[idx_lower[-2:], 0])