import numpy as np
from HelperFunctions.pct_change import pct_change
from scipy.stats import norm

from VaRFunctions.MBA_simple import VaR


def mba_map_fx(df_returns
               , investments
               , alpha_linked
               , market_dict
               , fx_dict
               , indices_dict
               , base_currency
               , percentile=0.99
               , n=200):
    var_array = np.empty(df_returns.shape[0])
    es_array = np.empty(df_returns.shape[0])
    
    
    for i in range(n, df_returns.shape[0]+1):
        df_returns_i = df_returns.iloc[i-n:i,:]
        exposures = []
        alpha = np.array([])
        
        for market_i in market_dict.keys():
            investments_in_market_i = list(set(investments) & set(market_dict[market_i]))
            fx_exposure = alpha_linked[investments_in_market_i].sum(axis=1)
            invest_alpha = alpha_linked[investments_in_market_i].values[0]
            if market_i == base_currency:
                alpha = np.hstack((alpha, invest_alpha))
                exposures.extend(investments_in_market_i)
            else:
                alpha = np.hstack((alpha, invest_alpha,fx_exposure))
                exposures.extend(investments_in_market_i)
                exposures.extend([fx_dict[market_i]])

        cov_matrix_capm = df_returns_i.loc[:, exposures].cov().dropna()
        portfolio_variance_capm = (alpha.T @ cov_matrix_capm  @ alpha)
        var_array[i-1] = np.sqrt(portfolio_variance_capm)*norm.ppf(percentile)
        es_array[i-1] = np.sqrt(portfolio_variance_capm) * norm.pdf(norm.ppf(percentile))/(1-percentile)
    
    return var_array, es_array
    

def mba_map_index(df_returns
                  , investments
                  , alpha_linked
                  , market_dict
                  , fx_dict
                  , indices_dict
                  , base_currency
                  , percentile=0.99
                  , n=200):
    var_array = np.empty(df_returns.shape[0])
    es_array = np.empty(df_returns.shape[0])

    for i in range(n, df_returns.shape[0]+1):
        df_returns_i = df_returns.iloc[i-n:i,:]
        exposures = []
        alpha = []
        
        for market_i in market_dict.keys():
            investments_in_market_i = list(set(investments) & set(market_dict[market_i]))
            index_i = indices_dict[market_i]
            cov_market_i = df_returns_i.loc[:, investments_in_market_i + [index_i]].cov()
            betas_market_i = cov_market_i.loc[:, index_i] / cov_market_i.loc[index_i, index_i]
            beta_exp = betas_market_i[investments_in_market_i] * alpha_linked[investments_in_market_i]
            sum_beta = beta_exp.sum(axis=1)
            fx_exposure = alpha_linked[investments_in_market_i].sum(axis=1)
            if market_i == base_currency:
                alpha.extend([sum_beta])
                exposures.extend([index_i])
            else:    
                alpha.extend([sum_beta, fx_exposure])
                exposures.extend([index_i, fx_dict[market_i]])
        
        alpha = np.array(alpha)
        cov_matrix_capm = df_returns_i.loc[:, exposures].cov().dropna()
        portfolio_variance_capm = (alpha.T @ cov_matrix_capm  @ alpha).loc[0,0]
        var_array[i-1] = np.sqrt(portfolio_variance_capm)*norm.ppf(percentile)
        es_array[i-1] = np.sqrt(portfolio_variance_capm) * norm.pdf(norm.ppf(percentile))/(1-percentile)
    
    return var_array, es_array
        
    
