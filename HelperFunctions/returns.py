import pandas as pd
import numpy as np
from HelperFunctions.pct_change import pct_change


def returns(df_data, alpha):

    investments = df_data.iloc[:,3:].columns
    df_data = pct_change(df_data)

    df_data['returns'] = np.dot((df_data.iloc[:,(3+investments.size):].values), alpha)
    return df_data
