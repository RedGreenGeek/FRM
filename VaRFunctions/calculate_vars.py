from VaRFunctions.VaRFunctions import *
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_vars_fx_converted(df_data, investments, alpha, market_dict):#), forex, alpha_fx, percentiles, n=200):
    stock_returns = df_data.loc[:,investments].pct_change().dropna()
    weighted_returns = stock_returns @ alpha 
    loss = -weighted_returns
    
    # # historical approach 
    hs_simple_var = HS_simple(loss, percentile=0.99, n=200)
    hs_weighted_var = HS_weighted(loss, percentile=0.99, n=200, lam=0.995)
    mba_simple_var = MBA_simple(stock_returns, alpha, percentile=0.99, n=200)
    mba_ewma_var = MBA_EWMA(stock_returns.iloc[-250:, :], alpha, percentile=0.99, n=200, lam=0.95, sigma_init=None)
    
    
    return pd.DataFrame([hs_simple_var, hs_weighted_var, mba_simple_var, mba_ewma_var])
    

def calculate_vars_risk_factors(df_data
                                , investments
                                , alpha_linked
                                , market_dict
                                , fx_dict
                                , indices_dict
                                , base_currency):
    df_returns = df_data.pct_change().dropna()
    
    mba_map_fx_var = mba_map_fx(df_returns, investments, alpha_linked, market_dict, fx_dict
                                      , indices_dict, base_currency, percentile=0.99, n=2500)
    mba_map_index_var = mba_map_index(df_returns, investments, alpha_linked, market_dict, fx_dict
                                      , indices_dict, base_currency, percentile=0.99, n=2500)
    
    return pd.DataFrame([mba_map_fx_var,mba_map_index_var])

def calculate_vars(df_data, investments, alpha, forex, alpha_fx, percentiles, n=200):

    for p in percentiles:
        df_data = MBA_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_EWMA(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_weighted(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_map_fx(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
        df_data = MBA_map_capm(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
    return df_data
