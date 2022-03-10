import pandas as pd


def pct_change(df_data, investments):

    for inv in investments:
        if (inv+'_pct') not in df_data:
            df_data[(inv+'_pct')] = df_data[inv].pct_change()
    return df_data
