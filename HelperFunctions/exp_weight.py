import numpy as np

def exp_weight(n: float, lam: float):

    # coeff = (1-lam)/(1-lam**n)
    # #poewr operations are expensive, the addition operations reduces runtime.
    # lam_i_n = np.power(lam,(-np.arange(1,n+1)+n))
    # weight = coeff * lam_i_n
    weight = lam**(n - np.arange(1, n+1)) * (1 - lam)/(1 - lam**n)
    
    return weight