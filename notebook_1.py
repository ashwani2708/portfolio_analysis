import pandas as pd
import statsmodels.api as sm

def compound(r):
    return np.expm1(np.log1p(r).sum())


def regress(dependent_variable, explanatory_variables, alpha=True):
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1

    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm

import numpy as np
brka_d = pd.read_csv("data/brka_d_ret.csv", parse_dates=True, index_col=0)
brka_d.head()

brka_m = brka_d.resample('M').apply(compound).to_period('M')
brka_m.to_csv("data/brka_m.csv")

fff = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",header=0, index_col=0, na_values=-99.99)/100
fff.index = pd.to_datetime(fff.index, format="%Y%m").to_period('M')
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
mkt_excess = fff.loc["1990":"2012-05",['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()

"""" FAMA FRENCH MODEL Explaining Berkshire hathaway return wrt Value,size and market returns """

exp_var["Value"] = fff.loc["1990":"2012-05",['HML']]
exp_var["Size"] = fff.loc["1990":"2012-05",['SMB']]
exp_var.head()
lm = sm.OLS(brka_excess, exp_var).fit()
lm.summary()

result = regress(brka_excess, mkt_excess)

result.params
result.tvalues
result.pvalues

regress(brka_excess, exp_var, alpha=False).summary()