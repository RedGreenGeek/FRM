
# Count number of VaR exceedings
if -df_data[('varsimple_' + n)][i] > df_data['returns'][i + 1]:
    k += 1


df_Var = VaR(df_data, investments, alpha)

df_data = MBA_EWMA(df_data, investments, alpha)