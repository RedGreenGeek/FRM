from VaRFunctions.VaRFunctions import *

def calculate_vars(df_data, investments, alpha, forex, alpha_fx, percentiles, n=200):

    for p in percentiles:
        df_data = MBA_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_EWMA(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = HS_weighted(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_map_fx(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
        df_data = MBA_map_capm(df_data, investments, alpha, forex, alpha_fx, percentile=p, n=n)
    return df_data

