from VaRFunctions.VaRFunctions import *
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

labels = df_data.columns[[-18,-17,-16,-15,-6,-5,-4,-3,-2,-1]].values
plt_vars(df_data, labels)
print(df_data.columns[-8:])