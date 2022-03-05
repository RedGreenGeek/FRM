import pandas as pd


def convert_fx(df_stock, investments, exchange):

    for i in range(len(investments)):
        df_stock[(investments[i]+'_'+exchange[3:6])] = df_stock[investments[i]].values * df_stock[exchange].values
    return df_stock

