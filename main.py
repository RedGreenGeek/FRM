from VaRFunctions.VaRFunctions import *
from HelperFunctions.HelperFunctions import *
from PlotFunctions.plt_vars import *
import matplotlib.pyplot as plt

from VaRFunctions.calculate_vars import *

#### settings for calculations ####
# VaR percentiles
percentiles = [0.95, 0.99]

file_names = ['./ExampleData/GEData.txt', './ExampleData/UKData.txt', './ExampleData/USData.txt']
market_cur = ['EUR', 'GBP', 'USD']
base_currency = "USD"
# df_data = file_reader(file_names, base_currency)
df_data, market_dict = import_dfs(file_names,market_cur)
fx_cols = ['EURUSD=X', 'GBPUSD=X']
market_dict['GBP'].remove('GBPUSD=X')
market_dict['EUR'].remove('EURUSD=X')
df_fx_converted = df_data.copy()
df_fx_converted.loc[:,market_dict['EUR']] *= df_data.loc[:,'EURUSD=X'].values[:, np.newaxis]
df_fx_converted.loc[:,market_dict['GBP']] *= df_data.loc[:,'GBPUSD=X'].values[:, np.newaxis]


investments = ['BMW.DE', 'DWNI.DE', 'RWE.DE', 'GSK.L', 'AZN.L', 'BATS.L', 'IBM', 'MS' ]

alpha = np.array([1000]*len(investments))

#df_fx_conv_port = df_fx_converted.loc[:,investments]

#alpha_fx = np.array([10e])

calculate_vars_fx_converted(df_fx_converted, investments, alpha)
alpha_fx = np.array([3000, 3000])
df_data = calculate_vars(df_data, investments, alpha, fx_cols, alpha_fx, percentiles)

labels = df_data.columns[[-18,-17,-16,-15,-6,-5,-4,-3,-2,-1]].values
plt_vars(df_data, labels)
print(df_data.columns[-8:])