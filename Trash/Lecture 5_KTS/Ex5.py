

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from FRM.HelperFunctions.returns import returns
from FRM.HelperFunctions.pct_change import pct_change
from FRM.HelperFunctions.sigma_p import sigma_p_func
from scipy.stats import norm



df_data = pd.read_csv('./FRM/Trash/Lecture 5_KTS/StockData.txt')

headers = df_data.columns

investments = df_data.iloc[:,3:8].columns
for i in range(0,5):
    print(investments[i])
    df_data[(investments[i]+' EUR')] = df_data[investments[i]].values * df_data['USDEUR'].values


df_stocks = pd.concat([df_data.iloc[:,0:3], df_data.iloc[:,11:16]], axis=1)

alpha = np.array([2000, 2000, 2000, 2000, 2000])

df_stocks = returns(df_stocks, alpha)

sigma_simple = df_stocks.iloc[:,8:13].cov()

# add an n
def VaR(sigma, alpha, percentile=0.99):
    sigma_p = sigma_p_func(sigma, alpha)
    return sigma_p*norm.ppf(percentile)

VaR(sigma_simple, alpha)

# 2

df_stocks = pd.concat([df_data.iloc[:,0:8], df_data.iloc[:,10]], axis=1)

df_stocks = pct_change(df_stocks)

alpha = np.array([2000, 2000, 2000, 2000, 2000, 10000])

sigma_simple = df_stocks.iloc[:,9:15].cov()

VaR(sigma_simple, alpha)

# 3

df_sigma_stocks = df_data.iloc[:,0:9]

df_sigma_stocks = pct_change(df_sigma_stocks)

stock_sigma = df_sigma_stocks.iloc[:,9:15].cov()

