#%%
from VaRFunctions.calculate_vars import *
from HelperFunctions.HelperFunctions import *
from PlotFunctions.plt_vars import *


#### settings for calculations ####
# VaR percentiles
percentiles = [0.95, 0.99]

file_names = ['./ExampleData/GEData.txt', './ExampleData/UKData.txt', './ExampleData/USData.txt']
base_currency = "USD"
df_data = file_reader(file_names, base_currency)

investments = ['BMW.DE', 'DWNI.DE', 'RWE.DE', 'GSK.L', 'AZN.L', 'BATS.L', 'IBM', 'MS' ]

alpha = np.ones(len(investments), dtype = int)*1000

forex = ['EURUSD=X','GBPUSD=X']
alpha_fx = np.array([3000, 3000])
df_data = calculate_vars(df_data, investments, alpha, forex, alpha_fx, percentiles)
#%%
var99s = df_data.columns[df_data.columns.str.contains('0.99')]

var95s = df_data.columns[df_data.columns.str.contains('0.95')]
# labels = df_data.columns[[-18,-17,-16,-15,-6,-5,-4,-3,-2,-1]].values
plt_vars(df_data, var95s)
plt_vars(df_data, var99s)

plt_table(df_data, var95s)
plt_table(df_data, var99s)


print(df_data[var99s].tail(1))
print(df_data[var95s].tail(1))


# %%
