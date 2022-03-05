from VaRFunctions.VaRFunctions import *
from HelperFunctions.HelperFunctions import *

file_names = ['./ExampleData/GEData.txt', './ExampleData/UKData.txt', './ExampleData/USData.txt']
base_currency = "USD"
df_data = file_reader(file_names, base_currency)

investments = ['BMW.DE_USD', 'DWNI.DE_USD', 'RWE.DE_USD', 'GSK.L_USD', 'AZN.L_USD', 'BATS.L_USD', 'IBM', 'MS']
alpha = np.ones(len(investments), dtype = int)*1000
percentiles = [0.95, 0.99]

df_data = calculate_vars(df_data, investments, alpha, percentiles)

print(df_data.columns[-4:])