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
df_data, market_dict, fx_dict, indices_dict = import_dfs(file_names,market_cur,base_currency)
inv_market_dict = {}
for k, v in market_dict.items():
    for x in v:
        inv_market_dict[x] = k

# chosen stocks
investments = ['BMW.DE', 'DWNI.DE', 'RWE.DE', 'GSK.L', 'AZN.L', 'BATS.L', 'IBM', 'MS' ]

# array of initial investmens
alpha = np.array([1000]*len(investments))
# link the investments to the name for use in mapping approaches
alpha_linked = pd.DataFrame(np.reshape(alpha,(1,len(alpha))), columns=investments)

n_calc_days = 250
n_days_back_test = 250
var_es_dict = calculate_var_es(df_data, investments, alpha, alpha_linked, market_dict, fx_dict
                            , indices_dict, base_currency, percentiles, n_calc_days, n_days_back_test)

