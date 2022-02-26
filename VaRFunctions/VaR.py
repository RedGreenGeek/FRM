from sigma_p import sigma_p_func
from scipy.stats import norm

def VaR(sigma, alpha, X):
    sigma_p = sigma_p_func(sigma, alpha)
    return sigma_p*norm.ppf(X)