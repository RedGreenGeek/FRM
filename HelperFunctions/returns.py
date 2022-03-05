import pandas as pd
import numpy as np
from HelperFunctions.pct_change import pct_change


def returns(df_data, investments, alpha):

    df_data = pct_change(df_data, investments)
    pct_investments = [inv+'_pct' for inv in investments]
    if 'returns' not in df_data.columns:
        df_data['returns'] = np.dot((df_data[pct_investments].values), alpha)
    return df_data
