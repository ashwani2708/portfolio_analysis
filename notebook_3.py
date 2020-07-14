import numpy as np
import pandas as pd
import nb.notebook_2 as pf_2
from nb.notebook_2 import get_ind_file
ind_cw = pf_2.get_ind_file("returns", ew=False)
ind_ew = pf_2.get_ind_file("returns", ew=True)

def get_ind_file(filetype, weighting="vw", n_inds=30):

    if filetype is "returns":
        name = f"{weighting}_rets"
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")

    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind
def annualize_vol(r, periods_per_year):

    return r.std()*(periods_per_year**0.5)
def annualize_rets(r, periods_per_year):

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def sharpe_ratio(r, riskfree_rate, periods_per_year):

    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def get_ind_market_caps(n_inds=30, weights=False):

    ind_nfirms = get_ind_file("nfirms")
    ind_size = get_ind_file("size")
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap

sr = pd.DataFrame({"CW": sharpe_ratio(ind_cw["1945":], 0.03, 12), "EW": sharpe_ratio(ind_ew["1945":], 0.03, 12)})

sr.plot.bar(figsize=(12, 6))


ax = ind_cw.rolling('1825D').apply(sharpe_ratio, raw=True, kwargs={"riskfree_rate":0.03, "periods_per_year":12}).mean(axis=1)["1945":].plot(figsize=(12,5), label="CW", legend=True)
ind_ew.rolling('1825D').apply(sharpe_ratio, raw=True, kwargs={"riskfree_rate":0.03, "periods_per_year":12}).mean(axis=1)["1945":].plot(ax=ax, label="EW", legend=True)
ax.set_title("Average Trailing 5 year Sharpe Ratio 1945-2018")

ind49_rets = pf_2.get_ind_returns(weighting="vw", n_inds=49)["1974":]
ind49_mcap = get_ind_market_caps(49, weights=True)["1974":]

def weight_ew(r):

    n = len(r.columns)
    return pd.Series(1/n, index=r.columns)

def backtest_ws(r, estimation_window=30, weighting=weight_ew):

    n_periods = r.shape[0]
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    # windows is a list of tuples which gives us the (integer) location of the start and stop (non inclusive)
    # for each estimation window
    weights = [weighting(r.iloc[win[0]:win[1]]) for win in windows]
    # List -> DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    # return weights
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns

ewr = backtest_ws(ind49_rets, weighting=weight_ew)
ewi = (1+ewr).cumprod()
ewi.plot(figsize=(12,6), title="49 Industries - Equally Weighted");


def weight_ew(r, **kwargs):
    n = len(r.columns)
    return pd.Series(1/n, index=r.columns)

def weight_cw(r, cap_weights, **kwargs):

    return cap_weights.loc[r.index[0]]

def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):

    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert list of weights to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    # return weights
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns