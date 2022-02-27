import pandas as pd


def pct_change(df_data):

    investments = df_data.iloc[:,3:].columns

    for inv in investments:
        df_data[(inv+'pct')] = df_data[inv].pct_change()
    return df_data
