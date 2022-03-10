from HelperFunctions.HelperFunctions import returns
import numpy as np

def HS_simple(loss, percentile=0.99, n=200):
    var_array = np.empty(loss.shape[0])
    for i in range(n, loss.shape[0]): 
        loss_sort = np.sort(loss[i-n:i])
        var_array[i] = loss_sort[int(n*percentile)]
    
    return var_array