# encoding=utf-8
"""
    Created on 17:29 2017/3/16 
    @author: Jindong Wang
"""
import numpy as np
from scoring import rmsle, scoring_rmsle
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def merge_results(res_list, X_valid):
    res_list_arr = np.asarray(res_list).T
    X = np.hstack((res_list_arr, X_valid.reshape((len(X_valid), 1))))

    n_train = int(np.ceil(0.9 * len(X)))

    X_train = X[:n_train, :-1]
    y_train = X[:n_train, -1]
    X_val = X[n_train:, :-1]
    y_val = X[n_train:, -1]
    return X_train, y_train, X_val, y_val


def stack_model(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    loss = rmsle(y_val, y_val_pred)
    print('Stacking loss is:%f' % loss)
    print('Best parameter:%s' % clf.coef_)
    joblib.dump(clf, '../result/model/stack_model_lin.pkl')
    return clf


def stack_ridge(X_train, y_train, X_val, y_val):
    from sklearn import linear_model
    param_grid = {
        'alpha': [0.9, 1, 1.1, 1.2, 1.5, 2]
    }
    clf = GridSearchCV(linear_model.Ridge(), param_grid=param_grid, n_jobs=-1, verbose=1, cv=4, scoring=scoring_rmsle)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    loss = rmsle(y_val, y_val_pred)
    print('Stacking loss is:%f' % loss)
    print('Best parameter:%s' % clf.best_estimator_)
    joblib.dump(clf, '../result/model/stack_model_ridge.pkl')
    return clf


def stack_lasso(X_train, y_train, X_val, y_val):
    from sklearn import linear_model
    param_grid = {
        'alpha': [0.005, 0.1, 0.3, 0.5, 0.8, 1]
    }
    clf = GridSearchCV(linear_model.Lasso(), param_grid=param_grid, n_jobs=-1, verbose=1, cv=4, scoring=scoring_rmsle)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    loss = rmsle(y_val, y_val_pred)
    print('Stacking loss is:%f' % loss)
    print('Best parameter:%s' % clf.best_estimator_)
    joblib.dump(clf, '../result/model/stack_model_lasso.pkl')
    return clf


def stack_svr(X_train, y_train, X_val, y_val):
    from sklearn.svm import SVR
    param_grid = {
        'C': [0.005, 0.1, 0.3, 0.5, 0.8, 1, 5, 10, 50, 100],
        'gamma': [0.005, 0.1, 0.3, 0.5, 0.8, 1, 5, 10, 50, 100]
    }
    clf = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1, verbose=1, cv=4, scoring=scoring_rmsle)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    loss = rmsle(y_val, y_val_pred)
    print('Stacking loss is:%f' % loss)
    print('Best parameter:%s' % clf.best_estimator_)
    joblib.dump(clf, '../result/model/stack_model_svr.pkl')
    return clf


def stack_knn(X_train, y_train, X_val, y_val):
    from sklearn.neighbors import KNeighborsRegressor
    param_grid = {
        'n_neighbors': [20, 25, 30, 35, 50]
    }
    clf = GridSearchCV(KNeighborsRegressor(), param_grid=param_grid, n_jobs=-1, verbose=1, cv=4, scoring=scoring_rmsle)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    loss = rmsle(y_val, y_val_pred)
    print('Stacking loss is:%f' % loss)
    print('Best parameter:%s' % clf.best_estimator_)
    joblib.dump(clf, '../result/model/stack_model_knn.pkl')
    return clf
