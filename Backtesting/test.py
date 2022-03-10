

def back_testing(df, allocation, n_days, VaR_level = 99, paradigme = "MBA-simple"):
    
    length = df.shape[0]
    VaR = []
    PL = []
    
    if paradigme == "MBA-simple":
        
        for i in range(n_days, length):
            
            VaR_estimates = Model_Based_Approach(df[(i-n_days):i], allocation, 
                                 n_days = n_days, 
                                 EWMA = False, 
                                 lambdA = 0.94, 
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
                                         weighted = False, 
                                         lambdA = 0.94, 
                                         verbose = False, 
                                         quantiles = [VaR_level])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "HS-EWMA":
        
        for i in range(n_days, length):
            
            VaR_estimates = Historical_Simulation(df[(i-n_days):i], allocation, 
                                         n_scenarios = n_days, 
                                         weighted = True, 
                                         lambdA = 0.94, 
                                         verbose = False, 
                                         quantiles = [VaR_level])
            
            VaR.append(-VaR_estimates['VaR-' + str(VaR_level)])
            
    if paradigme == "HS-volatility-adjusted":
        
        for i in range(n_days, length):
            
            VaR_estimates = Historical_Simulation(df[(i-n_days):i], allocation, 
                                         n_scenarios = n_days, 
                                         weighted = False, 
                                         lambdA = 0.94, 
                                         verbose = False, 
                                         quantiles = [VaR_level])
            
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
    
    print(violations)
    
    for i in range(1, len(violations)):
        
        print(violations[i-1], violations[i])
        
        if (violations[i-1] == 0) and (violations[i] == 0):
            n_00 += 1
        
        elif (violations[i-1] == 0) and (violations[i] == 1):
            n_01 += 1

        elif (violations[i-1] == 1) and (violations[i] == 0):
            n_10 += 1
            
        elif (violations[i-1] == 1) and (violations[i] == 1):
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
    print("    * Likelihood-ratio test 1: ", p_value_1)
    print("    * Likelihood-ratio test 2: ", p_value_2)
    print("    * Likelihood-ratio test 3: ", p_value_3)
   
    plt.plot(VaR)
    plt.plot(PL)
    plt.title("Figure 1: Backtesting \n")
    plt.show()
    
    return {'Observered violations': n_1, 'Expected violations': expected_violations}
