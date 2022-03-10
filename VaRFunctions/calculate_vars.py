from VaRFunctions.VaRFunctions import *
import matplotlib.pyplot as plt

def calculate_vars_fx_converted(df_data, investments, alpha):#), forex, alpha_fx, percentiles, n=200):
    stock_returns = df_data.loc[:,investments].pct_change().dropna()
    weighted_returns = stock_returns @ alpha 
    loss = -weighted_returns
    
    # # historical approach 
    # HS_simple_var = HS_simple(loss, percentile=0.99, n=200)
    # HS_weighted_var = HS_weighted(loss, percentile=0.99, n=200, lam=0.995)
    # MBA_simple_var = MBA_simple(stock_returns, alpha, percentile=0.99, n=200)
    MBA_EWMA_var = MBA_EWMA(stock_returns.iloc[-250:, :], alpha, percentile=0.99, n=200, lam=0.95, sigma_init=None)
    #pf_var = alpha.values @ stock_returns.cov().values  @ alpha.T
    print('lol')
    


def calculate_vars(df_data, investments, alpha, forex, alpha_fx, percentiles, n=200):

    for p in percentiles:
        df_data = MBA_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_EWMA(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_weighted(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_map_fx(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
        df_data = MBA_map_capm(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
    return df_data
