from functools import reduce

import pandas as pd

from HelperFunctions.convert_fx import *


def import_dfs(file_names, market):
    dfs = [pd.read_csv(file_i) for file_i in file_names]
    stock_market = {market[i]: dfs[i].columns[3:].to_list() for i in range(len(file_names))}
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Year','Month','Day'],
                                            how='outer'), dfs)
    return df_merged, stock_market

def file_reader(file_names, base_currency):
    
    for i in range(len(file_names)):
        locals()[f'x{i}'] = pd.read_csv(file_names[i])
        if base_currency in locals()[f'x{i}'].columns[-1]:
            exchange = locals()[f'x{i}'].columns[-1]
            investments = locals()[f'x{i}'].columns[3:-2]
            locals()[f'x{i}'] = convert_fx(locals()[f'x{i}'], investments, exchange)

    df_stock = locals()[f'x{0}']
    for i in range(1, len(file_names)):
        df_stock = pd.concat([df_stock, locals()[f'x{i}']], axis=1)

    df_stock = df_stock.loc[:,~df_stock.columns.duplicated()]
    return df_stock
