
def HS_simple(returns_df, n=500,  percentile=0.99):
    percentile = 1-percentile
    return -returns_df['returns'].iloc[-n:].quantile(q=percentile)