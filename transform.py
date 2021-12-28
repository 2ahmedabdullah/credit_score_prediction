from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from utils import *



data_path = './data/'


if __name__ == '__main__':

    features =pd.read_csv(data_path+'my_header_list1.csv',names=['cols'])
    features =features.transpose()
    selected_features = list(features.iloc[0])

    data =pd.read_csv(data_path+'data.csv')

    x=data[selected_features]
    y=data['y']

    cat = dict()
    for var in selected_features:    
        mode = len(x[var].unique())    
        min_ = min(x[var])
        max_ = max(x[var])
        cat[var]=[mode,min_,max_]


    discrete_vars = [var for var in selected_features if len(x[var].unique()) < 101]
    discrete_vars.remove('x287')

        
    for j in range(0,len(discrete_vars)):
        feat = x[discrete_vars[j]]
        b = feat.value_counts()/len(x)*100
        filt = b<5
        rare_cats = list(filt[filt].index)
        val = [-1]*len(rare_cats)
        dictionary = dict(zip(rare_cats, val))
        feat1 = feat.replace(dictionary) 
        x[discrete_vars[j]]=feat1

    # cast all variables as categorical
    x[discrete_vars] = x[discrete_vars].astype('O')

    num_vars = [var for var in x.columns if var not in discrete_vars and var!='x287']


    out1 = outliers(x, num_vars)
    flat_list = [item for sublist in out1 for item in sublist]
    outliers= list(set(flat_list))

    x['y'] = y
    x_new= x.drop(x.index[outliers])

    y_new = x_new['y']
    x_new.drop(['y'],axis=1, inplace=True)


    X_train,X_test,y_train,y_test = train_test_split(x_new,y_new,test_size=0.1,random_state=10)
        
    num_feat_xtrain = X_train[num_vars]
    num_feat_xtest = X_test[num_vars]

    cat_feat_xtrain = X_train[discrete_vars]
    cat_feat_xtest = X_test[discrete_vars]
        

    pt = PowerTransformer(method='yeo-johnson')
    yeo_train = pt.fit_transform(num_feat_xtrain)
    yeo_test = pt.transform(num_feat_xtest)

    yeo_train = pd.DataFrame(yeo_train)
    yeo_test = pd.DataFrame(yeo_test)


    x_train_filled=filling_mean_values(yeo_train)
    x_test_filled=filling_mean_values(yeo_test)

    x_train_cat_filled=filling_mode_values(cat_feat_xtrain)
    x_test_cat_filled=filling_mode_values(cat_feat_xtest)


    x_train_filled = np.asarray(x_train_filled).astype(np.float32)
    x_test_filled = np.asarray(x_test_filled).astype(np.float32)


    x_train_cat_filled = np.asarray(x_train_cat_filled).astype(np.float32)
    x_test_cat_filled = np.asarray(x_test_cat_filled).astype(np.float32)

    x_train = np.concatenate((x_train_filled,x_train_cat_filled), axis=1)
    x_test = np.concatenate((x_test_filled,x_test_cat_filled), axis=1)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)


    x_train.to_csv(data_path+'xtrain.csv', index=False)
    y_train.to_csv(data_path+'ytrain.csv', index=False)

    x_test.to_csv(data_path+'xtest.csv', index=False)
    y_test.to_csv(data_path+'ytest.csv', index=False)

