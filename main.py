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
df_fx_converted = df_data.copy()
df_fx_converted.loc[:,market_dict['EUR']] *= df_data.loc[:,'EURUSD=X'].values[:, np.newaxis]
df_fx_converted.loc[:,market_dict['GBP']] *= df_data.loc[:,'GBPUSD=X'].values[:, np.newaxis]


investments = ['BMW.DE', 'DWNI.DE', 'RWE.DE', 'GSK.L', 'AZN.L', 'BATS.L', 'IBM', 'MS' ]

alpha = np.array([1000]*len(investments))

alpha_linked = pd.DataFrame(np.reshape(alpha,(1,len(alpha))), columns=investments)

#df_fx_conv_port = df_fx_converted.loc[:,investments]

#alpha_fx = np.array([10e])

# var_fx_conv_df = calculate_vars_fx_converted(df_fx_converted
#                                              , investments
#                                              , alpha
#                                              , market_dict
#                                              , [0.99])
var_loss = calculate_vars_risk_factors(df_data, investments, alpha_linked, market_dict, fx_dict
                            , indices_dict, base_currency)
print('lol')
