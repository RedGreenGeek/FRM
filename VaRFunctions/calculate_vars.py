from VaRFunctions.VaRFunctions import *
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_var_es(stock_returns, loss, df_returns, df_data, investments, alpha, alpha_linked, market_dict, fx_dict
                            , indices_dict, base_currency, percentiles, n_calc_days):
    # prepare output dict
    dict_output = {}
    
    # # historical approach 
    for perc_i in percentiles:
        hs_simple_var, es_hs_simple = HS_simple(loss, percentile=perc_i, n=n_calc_days)
        hs_weighted_var, es_ha_weighted = HS_weighted(loss, percentile=perc_i, n=n_calc_days, lam=0.995)
        mba_simple_var, es_array_mba = MBA_simple(stock_returns, alpha, percentile=perc_i, n=n_calc_days)
        mba_ewma_var, es_array_ewma = MBA_EWMA(stock_returns, alpha, percentile=perc_i, n=n_calc_days, lam=0.95, sigma_init=None)
        
        # mapping approaches
        mba_map_fx_var, es_mba_map_fx = mba_map_fx(df_returns, investments, alpha_linked, market_dict, fx_dict
                                      , indices_dict, base_currency, percentile=perc_i, n=n_calc_days)
        mba_map_index_var, es_mba_map_index = mba_map_index(df_returns, investments, alpha_linked, market_dict, fx_dict
                                        , indices_dict, base_currency, percentile=perc_i, n=n_calc_days)
        
        # stressed VaR
        mba_stressed_var, es_stressed = MBA_stressed(df_data, df_returns, alpha, stressed_range=[2493-20,2493+229]
                                                     , investments=investments, percentile=perc_i, n=n_calc_days)
        
        name_column = ['hs', 'pw_hs','mba', 'mba_ewma', 'map_w_fx','map_idx', 'stressed']
        # save results in dict for perc_i
        dict_output[f"var_{perc_i}"] = pd.DataFrame(np.array([hs_simple_var, hs_weighted_var
                                                              , mba_simple_var, mba_ewma_var
                                                              , mba_map_fx_var, mba_map_index_var
                                                              , mba_stressed_var]).T
                                                    , columns=name_column)
        #dict_output[f"violation_{perc_i}"] = dict_output[f"var_{perc_i}"].iloc[-n_calc_days:].values - loss.iloc[-n_calc_days:][:,np.newaxis]
        dict_output[f"violations_{perc_i}"] = (dict_output[f"var_{perc_i}"]
                                              ).apply(lambda x:
                                                (x.iloc[-n_calc_days:]-loss.iloc[-n_calc_days:].values)<0)
        dict_output[f"es_{perc_i}"] = pd.DataFrame(np.array([es_hs_simple, es_ha_weighted
                                                             , es_array_mba, es_array_ewma
                                                             , es_mba_map_fx, es_mba_map_index
                                                             , es_stressed]).T
                                                   , columns=name_column)
    
    
    return dict_output, name_column
    

# def calculate_vars_risk_factors(df_data
#                                 , investments
#                                 , alpha_linked
#                                 , market_dict
#                                 , fx_dict
#                                 , indices_dict
#                                 , base_currency):
#     df_returns = df_data.pct_change().dropna()
    
#     mba_map_fx_var, es_mba_map_fx = mba_map_fx(df_returns, investments, alpha_linked, market_dict, fx_dict
#                                       , indices_dict, base_currency, percentile=0.99, n=250)
#     mba_map_index_var, es_mba_map_index = mba_map_index(df_returns, investments, alpha_linked, market_dict, fx_dict
#                                       , indices_dict, base_currency, percentile=0.99, n=250)
    
#     return pd.DataFrame([mba_map_fx_var,mba_map_index_var])
