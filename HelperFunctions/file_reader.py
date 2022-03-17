from functools import reduce

import pandas as pd
import re

from pytest import mark

from HelperFunctions.convert_fx import *


def import_dfs(file_names, markets, base_currency):
    dfs = [pd.read_csv(file_i) for file_i in file_names]
    stock_market = {markets[i]: dfs[i].columns[3:].to_list() for i in range(len(file_names))}
    fx_conv_dict = {}
    market_indices = {}
    
    for market_i in markets: 
        # remove the index 
        reg_ex = re.compile("^\^")
        market_index_i = list(filter(reg_ex.match, stock_market[market_i]))[0]
        stock_market[market_i].remove(market_index_i)
        market_indices[market_i] = market_index_i
        
        if market_i != base_currency:
            # remove the currency
            reg_ex = re.compile(f".*{market_i}")
            fx_conv_i = list(filter(reg_ex.match, stock_market[market_i]))[0]
            stock_market[market_i].remove(fx_conv_i)
            fx_conv_dict[market_i] = fx_conv_i

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Year','Month','Day'],
                                            how='outer'), dfs)
    
    return df_merged, stock_market, fx_conv_dict, market_indices

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
