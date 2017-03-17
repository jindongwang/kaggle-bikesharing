# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import metrics
import model
from scoring import rmsle, scoring_rmsle
import stacking
from feature_eng import process_rawdata,gene_feature


def global_vars():
    global scoring_rmsle
    global root_model
    global root_result
    scoring_rmsle = metrics.make_scorer(rmsle, greater_is_better=False)
    root_model = '../result/model/'
    root_result = '../result/preds/'


def gene_submit(data_pred, count, outfile):
    data_pred['count'] = count
    data_pred.to_csv(outfile, index=None)


def avg_results(result_list):
    final_res = []
    for j in range(len(result_list[0])):
        p_res = []
        for i in range(len(result_list)):
            if result_list[i][j] > 0:
                p_res.append(result_list[i][j])
        final_res.append(np.mean(np.asarray(p_res)))
    return np.asarray(final_res, dtype=float)


def train_model(X_train, y_train, X_valid, y_valid, X_test, data_pred):
    # rf
    # model_rf, val_loss_rf, y_val_pred_rf = model.apply_rf(X_train, y_train, X_valid, y_valid)
    model_rf = joblib.load(root_model + 'model_rf.pkl')
    # gbrt
    # model_gbrt, val_loss_gbrt, y_val_pred_gbrt = model.apply_gbrt(X_train, y_train, X_valid, y_valid)
    model_gbrt = joblib.load(root_model + 'model_gbrt.pkl')
    # xgb
    # model_xgb, val_loss_xgb, y_val_pred_xgb = model.apply_xgb(X_train, y_train, X_valid, y_valid)
    model_xgb = joblib.load(root_model + 'model_xgb.pkl')

    # svr
    # model_svr, val_loss_svr, y_val_pred_svr = model.apply_svr(X_train, y_train, X_valid, y_valid)
    model_svr = joblib.load(root_model + 'model_svr.pkl')

    # knn
    # model_knn, val_loss_knn, y_val_pred_knn = model.apply_knn(X_train, y_train, X_valid, y_valid)
    model_knn = joblib.load(root_model + 'model_knn.pkl')

    models = []
    val_preds = []
    models.append(model_rf)
    models.append(model_gbrt)
    models.append(model_xgb)
    # models.append(model_svr)
    # models.append(model_knn)
    val_preds.append(model_rf.predict(X_valid))
    val_preds.append(model_gbrt.predict(X_valid))
    val_preds.append(model_xgb.predict(X_valid))
    # val_preds.append(model_svr.predict(X_valid))
    # val_preds.append(model_knn.predict(X_valid))

    return models, val_preds


def load_data():
    train_data = np.loadtxt('../result/processed/train.csv', delimiter=',')
    X_test = np.loadtxt('../result/processed/test.csv', delimiter=',')
    X_train = train_data[:-1000, :-1]
    y_train = train_data[:-1000, -1]
    X_valid = train_data[-1000:, :-1]
    y_valid = train_data[-1000:, -1]
    data_test = pd.read_csv('../data/test.csv')
    data_pred = pd.DataFrame(data=data_test['datetime'].values, columns=['datetime'])
    return X_train, y_train, X_valid, y_valid, X_test, data_pred

def build_stack_model(y_preds, y_valid):
    stack_xtrain, stack_ytrain, stack_xval, stack_yval = stacking.merge_results(y_preds, y_valid)
    # stack_model = stacking.stack_ridge(stack_xtrain, stack_ytrain, stack_xval, stack_yval)
    # stack_model = stacking.stack_lasso(stack_xtrain, stack_ytrain, stack_xval, stack_yval)
    # stack_model = stacking.stack_svr(stack_xtrain, stack_ytrain, stack_xval, stack_yval)
    stack_model = stacking.stack_knn(stack_xtrain, stack_ytrain, stack_xval, stack_yval)

if __name__ == '__main__':
    # process_rawdata()
    X_train, y_train, X_valid, y_valid, X_test, data_pred = load_data()
    global_vars()
    models, y_preds = train_model(X_train, y_train, X_valid, y_valid, X_test, data_pred)
    # build_stack_model(y_preds,y_valid)
    res_list = []
    for i in models:
        res_list.append(i.predict(X_test))
    stack_model = joblib.load(root_model + 'stack_model_knn.pkl')
    y_pred = np.asarray(stack_model.predict(np.asarray(res_list).T),dtype=int)
    gene_submit(data_pred, y_pred, '../submit/submit_stack_allfeatures_knn.csv')
