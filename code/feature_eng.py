# encoding=utf-8
"""
    Created on 16:47 2017/3/17 
    @author: Jindong Wang
"""
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import FeatureUnion
import pandas as pd


def gene_feature(data_pd):
    # numeric columns
    col_binary = ['holiday', 'workingday']
    index_binary = np.asarray([(col in col_binary) for col in data_pd.columns], dtype=bool)
    # cate columns
    col_cate = ['season', 'weather']
    index_cate = np.asarray([(col in col_cate) for col in data_pd.columns], dtype=bool)
    # numeric columns
    col_num = ['temp', 'atemp', 'humidity', 'windspeed']
    index_num = np.asarray([(col in col_num) for col in data_pd.columns], dtype=bool)
    # normal value
    col_normal = ['month', 'day', 'hour']
    normal_num = np.asarray([(col in col_normal) for col in data_pd.columns], dtype=bool)

    feature_trans_list = [
        ('binary_value', Pipeline(steps=[
            ('select', preprocessing.FunctionTransformer(lambda x: x[:, index_binary])),
            ('transform', preprocessing.OneHotEncoder())
        ])),
        ('cate_value', Pipeline(steps=[
            ('select', preprocessing.FunctionTransformer(lambda x: x[:, index_cate])),
            ('transform', preprocessing.OneHotEncoder())
        ])),
        ('numeric_value', Pipeline(steps=[
            ('select', preprocessing.FunctionTransformer(lambda x: x[:, index_num])),
            ('transform', preprocessing.StandardScaler(with_mean=0))
        ])),
        ('normal_value', Pipeline(steps=[
            ('select', preprocessing.FunctionTransformer(lambda x: x[:, normal_num]))
        ]))
    ]
    feature_union = FeatureUnion(feature_trans_list)
    feature_set = feature_union.fit_transform(data_pd).toarray()
    return feature_set


def process_rawdata():
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    data_train['month'] = data_train['datetime'].apply(lambda x: int(x.split('-')[1]))
    data_train['day'] = data_train['datetime'].apply(lambda x: int(x.split('-')[2].split(' ')[0]))
    data_train['hour'] = data_train['datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))

    data_test['month'] = data_test['datetime'].apply(lambda x: int(x.split('-')[1]))
    data_test['day'] = data_test['datetime'].apply(lambda x: int(x.split('-')[2].split(' ')[0]))
    data_test['hour'] = data_test['datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))

    y = data_train['count'].values
    X = gene_feature(data_train.drop(['datetime', 'casual', 'registered', 'count'], axis=1))
    label_col = np.asarray(data_train['count'].values)
    label_col = label_col.reshape((len(label_col), 1))
    X = np.hstack((X, label_col))
    np.savetxt('../result/train.csv', X, fmt='%.4f', delimiter=',')

    data_pred = pd.DataFrame(data=data_test['datetime'].values, columns=['datetime'])
    X_test = gene_feature(data_test.drop('datetime', axis=1))
    np.savetxt('../result/test.csv', X_test, fmt='%.4f', delimiter=',')
