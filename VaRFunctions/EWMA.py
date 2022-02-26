import numpy as np

def EWMA(r_df
         , lam
         , sigma_init=None):
    n_obs, n_assets = r_df.shape

    # convert to numpy
    r_df_ndarry = r_df.values

    # assume initialize with matrix identity if nothing is specified
    if sigma_init is None:
        sigma = np.eye(n_assets)
    else:
        sigma = sigma_init

    for returns in r_df_ndarry:
        sigma = sigma * lam + (1 - lam) * np.outer(returns, returns)

    return sigma