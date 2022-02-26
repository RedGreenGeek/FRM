import numpy as np

def sigma_p_func(sigma, alpha):
    sigma_p_var = np.dot(np.dot(np.transpose(alpha), sigma), alpha)
    sigma_p = np.sqrt(sigma_p_var)

    return sigma_p