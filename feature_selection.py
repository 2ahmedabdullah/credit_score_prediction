#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:19:54 2021

@author: abdul
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from catboost import Pool, CatBoostClassifier
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint
from utils import *
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


'''
Assumption1: x001 is an ID and will be dropped from features
Assumption2: all features with high cardinality>100 are continuous
Assumption3: target variable ranges from 300-900
Assumption4: features having cardinality<=100 are categorical features
Assumption5: %count < 5% in categorical feature will be considered as Rare categories and will be summarized accordingly
Assumption6: datapoints above 99percentile will be considered as an OUTLIER in Numeric features
'''


data_path = './data/'

if __name__ == '__main__':
    data =pd.read_csv(data_path+'data.csv')

    nan_mean = data.isna().mean()
    nan_mean = nan_mean[nan_mean != 0].sort_values()
    nan_mean

    #Removing NA cols
    def drop_cols_with_na(df,na_limit):
        df=df.loc[:, df.isnull().mean() < na_limit]
        return df

    data1=drop_cols_with_na(data, 0.3)


    #DROP col with LOW VARIANCE
    describe=data1.describe()
    describe=describe.transpose()
    std_dev=describe['std']

    data1.var().sort_values(ascending=True)


    drop_feature1 = drop_num_variables_with_zero_variance(data1)

    data1.drop(drop_feature1, axis=1, inplace=True)


    y = data1['y']
    X = data1.drop('y', axis=1)


    #correlation
    corr_matrix = X.corr()
    corr_matrix.style.background_gradient(cmap='coolwarm')
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_feature2 = [column for column in upper.columns if any(upper[column] > 0.85)]


    # Drop features which are highly correlated 
    X.drop(drop_feature2, axis=1, inplace=True)
    X.isnull().sum()
    X['y']=y

    data3=drop_rows_with_na(X, len(X.columns))

    data3.isnull().sum().sum()

    y_n = data3['y']
    X_n = data3.drop('y', axis=1)


    heads = list(X_n)


    # random forest for feature importance on a regression problem
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X_n, y_n)
    # get importance
    importance = model.feature_importances_

    # summarize feature importance
    for i,v in enumerate(importance):
    	print('Feature: %s, Score: %.5f' % (heads[i],v))
        
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    imp = pd.Series(importance, index=heads)

    yum = imp<0.0001

    drop_feature3 = list(yum[yum].index)

    X_n.drop(drop_feature3, axis=1, inplace=True)

    final_features = list(X_n)

    np.savetxt(data_path+"my_header_list1.csv", final_features, delimiter=",", fmt='%s')

