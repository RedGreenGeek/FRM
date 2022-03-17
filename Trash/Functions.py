#### IMPORTING LIBRARIES ######

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from scipy.stats import chi2
import datetime as dt
import seaborn as sns

###### Preparing for load bar ########

# Some of the functions might take long to execute. 
# Therefore, a bar has been incorporated.

###### DEFINING FUNCTIONS ######

### Functions for converting to dates ###

def descriptive_statistics(df, trading_days = 247):

    length = df.shape[0]
    Results = {}
    Years = length/trading_days

    for column in df.columns:

        if column not in ['Day', 'Month', 'Year']:

            AR = round(((df[column].values[length-1]/df[column].values[0])**(1/Years) - 1)*100, 2)
            SD = round(np.std(df[column].pct_change()[1:]) * np.sqrt(trading_days)*100, 2)
            TR = round((df[column].iloc[length-1]/df[column].iloc[0]-1)*100, 2)
            SR = round(AR/SD,2)

            Results[column] = {
                'Annualized return': AR ,
                'Standard deviation': SD,
                'Total return': TR,
                'Sharp ratio':  SR
            }

    return(Results)

def correlation_matrix(df):

    SIGMA = df.corr()
    sns.heatmap(SIGMA)
    plt.savefig('covariance_matrix.svg', dpi=2400)


def convert_to_dates(df):

    length = df.shape[0]
    Dates = []

    for i in range(0,length):

        date_as_string = str(df['Day'].iloc[i]) + "/" + str(df['Month'].iloc[i]) + "/" + str(df['Year'].iloc[i])
        Dates.append(dt.datetime.strptime(date_as_string,'%d/%m/%Y').date())

    return (pd.DataFrame(Dates, columns = ['Date']))


### Conversion of base currency ###

def FX_conversion(stock, FX_rate):

    # Find number of rows
    length = stock.shape[0] 

    # Results
    converted_rates = np.zeros(length)

    # Run conversion new
    for i in range(0,length):
        converted_rates[i] = stock.iloc[i] * FX_rate.iloc[i]

    return (converted_rates)
    
# Converting bunch of stocks

def FX_conversions(stocks_FX, pairing):

    converted_stocks = pd.DataFrame()
    converted_stocks['Year'] = stocks_FX['Year']
    converted_stocks['Month'] = stocks_FX['Month']
    converted_stocks['Day'] = stocks_FX['Day']

    for stock in pairing:

        # check if base

        if pairing[stock] == 'Original':

            col_name = 'USD'
            conversion = stocks_FX[stock]

        else:

            col_name = pairing[stock][3:6]
            conversion = FX_conversion(stocks_FX[stock], stocks_FX[pairing[stock]])

        # Save result
        converted_stocks[stock + "." + col_name] = conversion

    return (converted_stocks)


def exp_weights(lamb, i, n):

    # Cannot take lambda = 1

    return(((lamb**(n-i))*(1-lamb)/(1-lamb**n)))

def compute_return(levels):
    
    returns = [0]
    
    for i in range(0, (len(levels) - 1)):
        
        returns.append((levels[i+1] - levels[i])/levels[i])
        
    return(returns)

def compute_returns(df):

    assets = df.columns
    n_days = df.shape[0]

    df_returns = pd.DataFrame()
    
    for asset in assets:
        df_returns[asset] = compute_return(list(df[asset]))
        
    return(df_returns.iloc[1:,:])

# This function used EWMA to estimate the asset-wise variance
# I wonder why they do not use the entire covariance matrix?

def EWMA_SIGMA_NO_CORRELATION(df, lambdA, n_days):

    # It returns a vector of variances, if k_assets > 1.
    # Else it returns a single number representing the portfolio variance
    
    # Limits
    max_days = df.shape[0]
    k_assets = df.shape[1]
    
    # Select last n days
    if max_days <= n_days:
        n_days = max_days
    
    df = df.tail(n_days).copy()

    if k_assets == 1:

        # Initializing SIGMA as variance of portfolio
        SIGMA = pd.DataFrame.var(df).values[0]

        # Estimating covariance matrix with EWMA
        for i in range(0, n_days):
            
            # Handle SIGMA as a real number
            r = df.iloc[i].values[0]
            SIGMA = lambdA * SIGMA + (1 - lambdA) * (r**2)
    
    else:

        # Initializing SIGMA as variance of columns for asset-wise approach
        SIGMA = pd.DataFrame.var(df).values

        # Estimating covariance matrix with EWMA
        for i in range(0, n_days):
            
            # Handle SIGMA as an array
            r = df.iloc[i].values
            SIGMA = lambdA * SIGMA + (1 - lambdA) * np.power(r, 2)

    return SIGMA

def EWMA_SIGMA(df, lambdA, n_days):
    
    # Limits
    max_days = df.shape[0]
    k_assets = df.shape[1]
    
    # Select last n days
    if max_days <= n_days:
        n_days = max_days
    
    df = df.tail(n_days).copy()
    
    # Initializing SIGMA
    SIGMA = np.identity(k_assets)
    
    # Estimating covariance matrix with EWMA
    for i in range(0, n_days):
        
        r = np.matrix(df.iloc[i].values)
        
        SIGMA = lambdA * SIGMA + (1 - lambdA) * np.dot(np.transpose(r), r)
    
    return SIGMA

def Model_Based_Approach(df, allocation, 
                         n_days = 250, 
                         EWMA = False, 
                         lambdA = 0.995, 
                         verbose = False, 
                         quantiles = [95, 99]):
    
    # Convert quantiles to decimals
    quantiles = [quantile/100 for quantile in quantiles]
    
    # Sanity check allocation
    if len(allocation) != df.shape[1]:
        print("Each asset must be present in the allocation.")
        print("Received ", df.shape[1], " assets and ", len(allocation), " assets.")
    
    # Select last n days
    df = df.tail(n_days)
    
    # Calculate covariance matrix (simple or EWMA)
    if(EWMA):
        SIGMA = EWMA_SIGMA(df, lambdA, n_days)
    else:
        SIGMA = np.matrix(df.cov().values)
    
    # Calculate portfolio variance
    var_P = np.dot(np.dot(np.matrix(allocation), SIGMA),  np.transpose(np.matrix(allocation)))
    sd_P = math.sqrt(var_P)

    # Return 1-day VaR and 1-day ES
    VaRs = []
    ESs = []
    metrics = {}
    
    for quantile in quantiles:
        VaRs.append(sd_P * norm.ppf(quantile))
        metrics['VaR-' + str(int(quantile*100))] = sd_P * norm.ppf(quantile)   
        ESs.append(sd_P * norm.pdf(norm.ppf(quantile))/(1-quantile))
        metrics['ES-' + str(int(quantile*100))] = sd_P * norm.pdf(norm.ppf(quantile))/(1-quantile)

    # Printing results for convenience
    if verbose:
        print("--------- Analysis by the Model-based Approach --------- \n")
        print("    * Days used: ", n_days)
        print("    * EWMA: ", EWMA)
        print("    * lambda: ", lambdA if EWMA else "Not relevant as EWMA is not used", "\n")
        
        for key in metrics:
            
            print(key + ": " + str(metrics[key]))
    
    # Returning parameters
    return metrics
  

def Historical_Simulation(df, allocation, 
                         n_scenarios = 250, 
                         weight_scheme = False, 
                         lambdA = 0.94, 
                         verbose = False, 
                         quantiles = [95, 99]):
    
    # In order to get n scenarios, we must use n + 1 days

    # Definition of scenario i
    # scenario_i = (v_i / v_(i-1) - 1)
    
    # Select last n days
    df = df.tail(n_scenarios).copy()
    
    # If exponential decay, then compute weights. Otherwise just weight them equally.
    if weight_scheme:
        weights = [exp_weights(lambdA, i, n_scenarios) for i in range(1, n_scenarios + 1)]

    else:
        weights = [1/n_scenarios for i in range(0,n_scenarios)]
        
    # Calculate loss on in all scenarios
    loss = []
    
    for i in range(0, n_scenarios):
        loss.append(np.dot(df.iloc[i].values, allocation))
        
    df['Loss'] = loss
    df['Weights'] = weights

    # Sort these by the size of the loss
    df_sorted = df.sort_values(by = ['Loss'])
    
    # Scaling quantiles
    quantiles = [quantile/100 for quantile in quantiles]
    
    # Accumulate the weights to find the right percentiles
    accumulated_weights = [sum(df_sorted['Weights'].iloc[0:i]) for i in range(1, n_scenarios + 1)]
    
    df_sorted['ACC_Weights'] = accumulated_weights

    # Saving the result
    results = {}

    # In order to get VaR, we find the correct index
    for quantile in quantiles: 
        
        i = 0
        
        for i in range(0,n_scenarios-1):
            
            if df_sorted['ACC_Weights'].iloc[i] > (1-quantile):
                
                if i > 0:  
                    results['VaR-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[i-1], 2)
                    results['ES-' + str(int(quantile * 100))] = round(-np.mean(df_sorted['Loss'].iloc[0:i]), 2)
                    break
                    
                else:  
                    results['VaR-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[0], 2)
                    results['ES-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[0], 2)
                    break

    if verbose:
        print(results)

    return results


# Volatility scaled losses on
def Historical_Simulation_Volatility_Adjusted(df, allocation, 
                         n_scenarios = 252, 
                         n_volatility = 252,
                         weight_scheme = False, 
                         portfolio_wise = False,
                         lambdA = 0.94, 
                         EWMA_factor = 0.995,
                         verbose = False, 
                         quantiles = [95, 99]):

    # Total investment
    total_investment = np.sum(allocation)

    # Specify important dimension
    n_observations = df.shape[0]
    n_assets = df.shape[1]

    # If n_scenario + n_volatility is larger than the total dataset, then the number of scenarios is scaled accordingly.
    if (n_scenarios + n_volatility) >= n_observations:
        n_scenarios = n_observations - n_volatility
        print("We are not able to run the simulation, since the number of scenarios and number")
        print("of days to estimate volatility exceeds the total number of observations.\n")
        print("The adjusted number of scenarios is: " + str(n_scenarios))
    
    # In order to get n scenarios, we must use n + 1 days

    # Definition of scenario i
    # scenario_i = (v_i / v_(i-1) - 1)
    
    # Select last n_scenarios + n_volatilities s.t. volatility can be computed on the first scenario
    df_most_recent = df.tail(n_scenarios + n_volatility).copy()
    df_results = df.tail(n_scenarios).copy()

    # If exponential decay, then compute weights. Otherwise just weight them equally.
    if weight_scheme:
        weights = [exp_weights(lambdA, i, n_scenarios) for i in range(1, n_scenarios + 1)]

    else:
        weights = [1/n_scenarios for i in range(0,n_scenarios)]
        
    # Calculate loss on in all scenarios
    adjusted_loss = []

    if not portfolio_wise:

        print("The program is calculating VaR and ES on asset-by-asset approach!\n")

        # Calculate the estimate for the volatility on the day after the last scenario
        # That is use the scenarios up to and including the last one.
        sigma_n_1 = EWMA_SIGMA_NO_CORRELATION(df_most_recent.tail(n_volatility), EWMA_factor, n_volatility)

        for i in range(0, n_scenarios):
            
            # Compute historical volatilities to adjust the losses
            sigma_i = EWMA_SIGMA_NO_CORRELATION(df_most_recent.iloc[i:(n_volatility+i),:], EWMA_factor, n_volatility)
            adjusted_loss.append(np.dot(df.iloc[i].values * (sigma_n_1 / sigma_i), allocation))
            

    if portfolio_wise:

        # Give message
        print("The program is calculating VaR and ES on portfolio-level!")

        # Create a vector of dimension 8 x 1 of relative allocation. It is pct. allocated on each asset.
        relative_allocation = np.transpose(np.matrix(allocation / np.sum(allocation)))

        # Make matrix out of portfolio returns (dimension 2933 x 8).
        pf_returns_matrix = np.matrix(df_most_recent)

        # Calculate relative portfolio returns (dimension 2933 x 1)
        pf_returns = np.dot(pf_returns_matrix, relative_allocation)

        # Convert back to data frame
        pf_returns = pd.DataFrame(pf_returns)
        print(pf_returns)

        # Then calculate E[volatility on day n + 1] = sigma_n_1. 
        sigma_n_1 = EWMA_SIGMA_NO_CORRELATION(pf_returns.tail(n_volatility), EWMA_factor, n_volatility)

        for i in range(0, n_scenarios):

            # Compute historical volatilities to adjust the losses
            sigma_i = EWMA_SIGMA_NO_CORRELATION(pf_returns.iloc[i:(n_volatility+i)], EWMA_factor, n_volatility)
            adjusted_loss.append(pf_returns.iloc[n_volatility + i].values[0] * (sigma_n_1 / sigma_i) * total_investment)

    df_results['Loss'] = adjusted_loss
    df_results['Weights'] = weights

    # Sort these by the size of the loss
    df_sorted = df_results.sort_values(by = ['Loss'])
    
    # Scaling quantiles
    quantiles = [quantile/100 for quantile in quantiles]
    
    # Accumulate the weights to find the right percentiles
    accumulated_weights = [sum(df_sorted['Weights'].iloc[0:i]) for i in range(1, n_scenarios + 1)]
    
    df_sorted['ACC_Weights'] = accumulated_weights

    # Saving the result
    results = {}

    # In order to get VaR, we find the correct index
    for quantile in quantiles: 
        
        i = 0
        
        for i in range(0,n_scenarios-1):
            
            if df_sorted['ACC_Weights'].iloc[i] > (1-quantile):
                
                if i > 0:  
                    results['VaR-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[i-1], 2)
                    results['ES-' + str(int(quantile * 100))] = round(-np.mean(df_sorted['Loss'].iloc[0:i]), 2)
                    break
                    
                else:  
                    results['VaR-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[0], 2)
                    results['ES-' + str(int(quantile * 100))] = round(-df_sorted['Loss'].iloc[0], 2)
                    break

    if verbose:
        print(results)

    return results

def Backtesting(df, allocation, n_days = 252, n_volatility = 252, VaR_level = 99, paradigme = "MBA-simple", save_figure = True, figure_title = "Figure1",x_axis = []):
    
    length = df.shape[0]
    VaR = []
    PL = []
    
    if paradigme == "MBA-simple":
        
        for i in range(n_days, length):
            
            VaR_estimates = Model_Based_Approach(df[(i-n_days):i], allocation, 
                                 n_days = n_days, 
                                 EWMA = False, 
                                 lambdA = 0.994, 
                                 verbose = False, 
                                 quantiles = [VaR_level])

            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "MBA-EWMA":
        
        for i in range(n_days, length):
            
            VaR_estimates = Model_Based_Approach(df[(i-n_days):i], allocation, 
                                 n_days = n_days, 
                                 EWMA = True, 
                                 lambdA = 0.94, 
                                 verbose = False, 
                                 quantiles = [VaR_level])

            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "HS-simple":
        
        for i in range(n_days, length):
            
            VaR_estimates = Historical_Simulation(df[(i-n_days):i], allocation, 
                                         n_scenarios = n_days, 
                                         weight_scheme = False, 
                                         lambdA = 0.94, 
                                         verbose = False, 
                                         quantiles = [VaR_level])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "HS-EWMA":
        
        for i in range(n_days, length):
            
            VaR_estimates = Historical_Simulation(df[(i-n_days):i], allocation, 
                                         n_scenarios = n_days, 
                                         weight_scheme = True, 
                                         lambdA = 0.94, 
                                         verbose = False, 
                                         quantiles = [VaR_level])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "HS-volatility-adjusted-asset-wise":
        
        for i in range(n_days+n_volatility, length):
            
            VaR_estimates = Historical_Simulation_Volatility_Adjusted(df[(i-(n_days+n_volatility)):i], allocation, 
                         n_scenarios = n_days, 
                         n_volatility = n_volatility,
                         weight_scheme = False, 
                         portfolio_wise = False,
                         lambdA = 0.94, 
                         EWMA_factor = 0.995, 
                         verbose = False, 
                         quantiles = [95, 99])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])

    if paradigme == "HS-volatility-adjusted-portfolio":
        
        for i in range(n_days, length):
            
            VaR_estimates = Historical_Simulation_Volatility_Adjusted(df[(i-(n_days+n_volatility)):i], allocation, 
                         n_scenarios = n_days, 
                         n_volatility = n_volatility,
                         weight_scheme = False, 
                         portfolio_wise = True,
                         lambdA = 0.94, 
                         EWMA_factor = 0.995,
                         verbose = False, 
                         quantiles = [95, 99])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    # Then we wish to keep track of violations
    
    violations = []

    for i in range(n_days, length):
        
        PL.append(np.dot(df.iloc[i].values, allocation))
        
    for i in range(0,len(PL)-1):
        
        if VaR[i] > PL[i+1]:
            
            violations.append(1)
            
        else:
            
            violations.append(0)
            
    # Implementation of statistical tests
    
    n_0 = len(violations)
    n_1 = sum(violations)

    pi_expected = 1 - VaR_level/100
    pi_observed = n_1/n_0
    expected_violations = pi_expected*n_0
    
    LR_uc = (pi_expected**(n_1) * (1 - pi_expected)**(n_0))/(pi_observed**(n_1) * (1 - pi_observed)**(n_0))
    chi2_statistic_1 = -2*math.log(LR_uc, math.exp(1))
    
    # Running first test
    
    significance_level = 0.05
    p_value_1 = (1 - chi2.cdf(chi2_statistic_1, 1))
    
    # Running second test
    
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
    pi_11 = n_11 / (n_10 + n_11)
    
    LR_ind = (pi_observed**(n_1) * (1 - pi_observed)**(n_0)) / (pi_01**n_01 * (1 - pi_01)**n_00 * pi_11**n_11 * (1 - pi_11)**n_10)
    chi2_statistic_2 = -2*math.log(LR_ind, math.exp(1))
    p_value_2 = (1 - chi2.cdf(chi2_statistic_2, 1))
    
    # Running third test
    
    chi2_statistic_3 = chi2_statistic_1 + chi2_statistic_2
    p_value_3 = (1 - chi2.cdf(chi2_statistic_3, 2))
    
    print("--------- Statistical Analysis --------- \n \n")
    print("    * Expected number of violations: ", round(expected_violations))
    print("    * Actual number of violations: ", n_1)
    print("    * n_00: ", n_00)
    print("    * n_01: ", n_01)
    print("    * n_10: ", n_10)
    print("    * n_11: ", n_11)
    print("    * Likelihood-ratio test 1 (test statistic): ", chi2_statistic_1)
    print("    * Likelihood-ratio test 1: ", p_value_1)
    print("    * Likelihood-ratio test 2 (test statistic): ", chi2_statistic_2)
    print("    * Likelihood-ratio test 2: ", p_value_2)
    print("    * Likelihood-ratio test 3 (test statistic): ", chi2_statistic_3)
    print("    * Likelihood-ratio test 3: ", p_value_3)

    # Convert x-axis into right length
    x_axis = x_axis[n_days:]

    if len(x_axis) == 0:

        plt.plot(VaR)
        plt.plot(PL)

    else:

        plt.plot(x_axis, VaR)
        plt.plot(x_axis, PL)

    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend(['VaR-' + str(VaR_level), 'Realised returns']) 
    #plt.title("Figure 1: Backtesting \n")

    if save_figure:
        plt.savefig(figure_title + '.svg', dpi=2400)

    plt.show()

    return {'Observered violations': n_1, 'Expected violations': expected_violations}

    