from operator import index
import pandas as pd 
import numpy as np
import math
from scipy.stats import chi2


def backtesting(var_es_dict, name_models, percentiles, n_days_back_test):
    # coverage
    list_str = list(map(str,percentiles))
    tests = pd.DataFrame(index=pd.MultiIndex.from_product([name_models, list_str]),
                         columns = ['violations', 'len_period',
                                    'p_value_1', 'chi2_statistic_1',
                                    'p_value_2', 'chi2_statistic_2',
                                    'p_value_3', 'chi2_statistic_3'])
    
    for name_i in name_models:
        for perc_i in list_str:
            tests.loc[(name_i, perc_i),:] = coverage(var_es_dict[f"violations_{perc_i}"][name_i].values, n_days_back_test, float(perc_i))
    
    return tests      
            
def coverage(violations, n_days_back_test, percentile):
    n1 = np.sum(violations)
    n0 = n_days_back_test - n1
    pi_expected = 1 - percentile
    pi_observed = n1/n_days_back_test
    expected_violations = pi_expected*n_days_back_test
    LR_uc = (pi_expected**(n1) * (1 - pi_expected)**(n0))/(pi_observed**(n1) * (1 - pi_observed)**(n0))
    chi2_statistic_1 = -2*math.log(LR_uc, math.exp(1))
    p_value_1 = (1 - chi2.cdf(chi2_statistic_1, 1))
    
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0
        
    for i in range(1, len(violations)):
            
        if ((violations[i-1] == 0) and (violations[i] == 0)):
            n_00 += 1
            
        elif ((violations[i-1] == 0) and (violations[i] == 1)):
            n_01 += 1

        elif ((violations[i-1] == 1) and (violations[i] == 0)):
            n_10 += 1
                
        elif ((violations[i-1] == 1) and (violations[i] == 1)):
            n_11 += 1
                
        else:
        
            print("Something is wrong!")

    pi_01 = n_01 / (n_00 + n_01)
    if (n_10 + n_11) != 0:
        pi_11 = n_11 / (n_10 + n_11)
    else:
        pi_11 = np.nan 
    LR_ind = (pi_observed**(n1) * (1 - pi_observed)**(n0)) / (pi_01**n_01 * (1 - pi_01)**n_00 * pi_11**n_11 * (1 - pi_11)**n_10)
    chi2_statistic_2 = -2*math.log(LR_ind, math.exp(1))
    p_value_2 = (1 - chi2.cdf(chi2_statistic_2, 1))
    
    chi2_statistic_3 = chi2_statistic_1 + chi2_statistic_2
    p_value_3 = (1 - chi2.cdf(chi2_statistic_3, 2))
    
    
    return n1, n_days_back_test, p_value_1, chi2_statistic_1, p_value_2, chi2_statistic_2, p_value_3, chi2_statistic_3