import numpy as np


def prepare_dfs(df_data, alpha, investments, market_dict, n_calc_days, n_days_back_test):
    
    df_data_used = df_data.iloc[-(n_calc_days+n_days_back_test):,:]
    
    # data fx converted
    df_fx_converted = df_data_used.copy()
    df_fx_converted.loc[:,market_dict['EUR']] *= df_fx_converted.loc[:,'EURUSD=X'].values[:, np.newaxis]
    df_fx_converted.loc[:,market_dict['GBP']] *= df_fx_converted.loc[:,'GBPUSD=X'].values[:, np.newaxis]
    stock_returns = df_fx_converted.loc[:, investments].pct_change().dropna()
    weighted_returns = stock_returns @ alpha 
    loss = -weighted_returns
    
    # returns for mappings:
    df_map_returns = df_data_used.pct_change().dropna()
    
    return stock_returns, loss, df_map_returns