# encoding=utf-8
"""
    Created on 16:42 2017/3/16 
    @author: Jindong Wang
"""
import numpy as np
from sklearn import metrics

def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

scoring_rmsle = metrics.make_scorer(rmsle, greater_is_better=False)