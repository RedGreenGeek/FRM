from VaRFunctions.VaRFunctions import *
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_vars_fx_converted(df_data, investments, alpha, market_dict, percentiles):#), forex, alpha_fx, percentiles, n=200):
    stock_returns = df_data.loc[:, investments].pct_change().dropna()
    weighted_returns = stock_returns @ alpha 
    loss = -weighted_returns
    
    dict_output = {}
    n_calc = 250
    
    # # historical approach 
    for perc_i in percentiles:
        hs_simple_var, es_hs_simple = HS_simple(loss, percentile=0.99, n=n_calc)
        hs_weighted_var, es_ha_weighted = HS_weighted(loss, percentile=0.99, n=n_calc, lam=0.995)
        mba_simple_var, es_array_mba = MBA_simple(stock_returns, alpha, percentile=0.99, n=n_calc)
        mba_ewma_var, es_array_ewma = MBA_EWMA(stock_returns, alpha, percentile=0.99, n=n_calc, lam=0.95, sigma_init=None)
        
        dict_output[f"var_{perc_i}"] = pd.DataFrame(np.array([hs_simple_var, hs_weighted_var, mba_simple_var, mba_ewma_var]).T)
        dict_output[f"es_{perc_i}"] = pd.DataFrame(np.array([es_hs_simple, es_ha_weighted, es_array_mba, es_array_ewma]).T)
        
    
    
    
    return dict_output
    

def calculate_vars_risk_factors(df_data
                                , investments
                                , alpha_linked
                                , market_dict
                                , fx_dict
                                , indices_dict
                                , base_currency):
    df_returns = df_data.pct_change().dropna()
    
    mba_map_fx_var, es_mba_map_fx = mba_map_fx(df_returns, investments, alpha_linked, market_dict, fx_dict
                                      , indices_dict, base_currency, percentile=0.99, n=250)
    mba_map_index_var, es_mba_map_index = mba_map_index(df_returns, investments, alpha_linked, market_dict, fx_dict
                                      , indices_dict, base_currency, percentile=0.99, n=250)
    
    return pd.DataFrame([mba_map_fx_var,mba_map_index_var])





def calculate_vars(df_data, investments, alpha, forex, alpha_fx, percentiles, n=200):

    for p in percentiles:
        df_data = MBA_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_EWMA(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_weighted(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_map_fx(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
    return df_data
