from VaRFunctions.VaRFunctions import *

def calculate_vars(df_data, investments, alpha, percentiles, n=200):

    for p in percentiles:
        df_data = MBA_simple(df_data, investments, alpha, percentile=p, n=n)
        df_data = MBA_EWMA(df_data, investments, alpha, percentile=p, n=n)
    
    
    return df_data

