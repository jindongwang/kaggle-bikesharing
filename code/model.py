# encoding=utf-8
"""
    Created on 16:41 2017/3/16 
    @author: Jindong Wang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scoring import rmsle, scoring_rmsle

root_model = '../result/model/'
root_result = '../result/preds/'
root_val_pred = '../result/val_preds/'


# scoring_rmsle = metrics.make_scorer(rmsle, greater_is_better=False)


def apply_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    param_grid = {
        'n_estimators': [16, 17, 18, 20, 25, 30, 35, 50, 100],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10]
    }
    model_xgb = GridSearchCV(xgb.XGBRegressor(), param_grid=param_grid, n_jobs=-1, verbose=1, scoring=scoring_rmsle,
                             cv=4)
    model_xgb.fit(X_train, y_train)
    y_val_pred = model_xgb.best_estimator_.predict(X_valid)
    val_loss = rmsle(y_valid, y_val_pred)
    print('RMSLE for validation: %f' % val_loss)
    print('Best parameter:%s' % model_xgb.best_params_)
    joblib.dump(model_xgb.best_estimator_, root_model + 'model_xgb.pkl')
    np.savetxt(root_val_pred + 'y_val_pred_xgb.csv', y_val_pred, fmt='%f', delimiter=',')
    return model_xgb.best_estimator_, val_loss, y_val_pred


def apply_gbrt(X_train, y_train, X_valid, y_valid):
    from sklearn.ensemble import GradientBoostingRegressor
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
        'n_estimators': [40,45,50,55]
    }
    gbrt = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid, verbose=1, scoring=scoring_rmsle, n_jobs=-1,
                        cv=4)
    gbrt.fit(X_train, y_train)
    y_val_pred = gbrt.best_estimator_.predict(X_valid)
    val_loss = rmsle(y_valid, y_val_pred)
    print('RMSLE for validation: %f' % val_loss)
    print('Best parameter:%s' % gbrt.best_params_)
    joblib.dump(gbrt.best_estimator_, root_model + 'model_gbrt.pkl')
    np.savetxt(root_val_pred + 'y_val_pred_gbrt.csv', y_val_pred, fmt='%f', delimiter=',')
    return gbrt.best_estimator_, val_loss, y_val_pred


def apply_rf(X_train, y_train, X_valid, y_valid):
    from sklearn.ensemble import RandomForestRegressor
    param_grid = {
        'max_depth': [2,12],
        'n_estimators': [40,45,50,55,60]
    }
    rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=4, n_jobs=-1, scoring=scoring_rmsle, verbose=1)
    rf.fit(X_train, y_train)
    y_val_pred = rf.best_estimator_.predict(X_valid)
    val_loss = rmsle(y_valid, y_val_pred)
    print('RMSLE for validation: %f' % val_loss)
    print('Best parameter:%s' % rf.best_params_)
    joblib.dump(rf.best_estimator_, root_model + 'model_rf.pkl')
    np.savetxt(root_val_pred + 'y_val_pred_rf.csv', y_val_pred, fmt='%f', delimiter=',')
    return rf.best_estimator_, val_loss, y_val_pred


def apply_svr(X_train, y_train, X_valid, y_valid):
    from sklearn.svm import SVR
    param_grid = {
        'C': [13, 14, 15, 16, 18, 20, 25, 30, 50, 100],
        'gamma': [0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 10]
    }
    model_svr = GridSearchCV(SVR(), cv=4, n_jobs=-1, param_grid=param_grid, scoring=scoring_rmsle, verbose=1)
    model_svr.fit(X_train, y_train)
    y_val_pred = model_svr.best_estimator_.predict(X_valid)
    val_loss = rmsle(y_valid, y_val_pred)
    print('RMSLE for validation: %f' % val_loss)
    print('Best parameter:%s' % model_svr.best_params_)
    joblib.dump(model_svr.best_estimator_, root_model + 'model_svr.pkl')
    np.savetxt(root_val_pred + 'y_val_pred_svr.csv', y_val_pred, fmt='%f', delimiter=',')
    return model_svr.best_estimator_, val_loss, y_val_pred


def apply_knn(X_train, y_train, X_valid, y_valid):
    from sklearn.neighbors import KNeighborsRegressor
    param_grid = {
        'n_neighbors': [1, 3, 5, 8, 10, 15, 18, 20, 22]
    }
    model_knn = GridSearchCV(KNeighborsRegressor(), cv=4, n_jobs=-1, param_grid=param_grid, scoring=scoring_rmsle,
                             verbose=1)
    model_knn.fit(X_train, y_train)
    y_val_pred = model_knn.best_estimator_.predict(X_valid)
    val_loss = rmsle(y_valid, y_val_pred)
    print('RMSLE for validation: %f' % val_loss)
    print('Best parameter:%s' % model_knn.best_params_)
    joblib.dump(model_knn.best_estimator_, root_model + 'model_knn.pkl')
    np.savetxt(root_val_pred + 'y_val_pred_knn.csv', y_val_pred, fmt='%f', delimiter=',')
    return model_knn.best_estimator_, val_loss, y_val_pred
