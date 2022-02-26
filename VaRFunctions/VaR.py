from VaRFunctions.sigma_p import sigma_p_func
from scipy.stats import norm

def VaR(sigma, alpha, percentile=0.99):
    sigma_p = sigma_p_func(sigma, alpha)
    return sigma_p*norm.ppf(percentile)