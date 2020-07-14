import numpy as np
import pandas as pd

from nb.notebook_1 import regress
from scipy.optimize import minimize

def annualize_vol(r, periods_per_year):

    return r.std()*(periods_per_year**0.5)
def tracking_error(r_a, r_b):

    return np.sqrt(((r_a - r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):

    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dependent_variable, explanatory_variables):

    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }
    solution = minimize(portfolio_tracking_error, init_guess,
                        args=(dependent_variable, explanatory_variables,), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def get_ind_file(filetype, ew=False):

    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        raise ValueError(f"filetype must be one of:{','.join(known_types)}")
    if filetype is "returns":
        name = "ew_rets" if ew else "vw_rets"
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1

    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

ind = get_ind_file('returns',ew=False)["2000":]

mgr_r = 0.3*ind["Beer"] + .5*ind["Smoke"] + 0.2*np.random.normal(scale=0.15/(12**.5), size=ind.shape[0])
weights = style_analysis(mgr_r, ind)*100
weights.sort_values(ascending=False).head(6).plot.bar()
coeffs = regress(mgr_r, ind).params*100
coeffs.sort_values().head()
coeffs.sort_values(ascending=False).head(6).plot.bar()
brka_m = pd.read_csv("data/brka_m.csv", index_col=0, parse_dates=True).to_period('M')


mgr_r_b = brka_m["2000":]["BRKA"]
weights_b = style_analysis(mgr_r_b, ind)
weights_b.sort_values(ascending=False).head(6).round(4)*100

brk2009 = brka_m["2009":]["BRKA"]
ind2009 = ind["2009":]
style_analysis(brk2009, ind2009).sort_values(ascending=False).head(6).round(4)*100

n = ind2009.shape[1]
init_guess = np.repeat(1/n, n)
bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
# construct the constraints
weights_sum_to_1 = {'type': 'eq','fun': lambda weights: np.sum(weights) - 1}
solution = minimize(portfolio_tracking_error, init_guess,args=(brk2009, ind2009,), method='SLSQP',options={'disp': False},constraints=(weights_sum_to_1,),bounds=bounds)
weights = pd.Series(solution.x, index=ind2009.columns)