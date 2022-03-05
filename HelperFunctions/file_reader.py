from HelperFunctions.convert_fx import *
import pandas as pd

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