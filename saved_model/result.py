# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_auc_score,r2_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor

from sklearn import datasets
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
# import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle

## XGBoost
data_dir = './'
dataset = pd.read_csv(data_dir + 'o2o_train.csv')
dataset.label.replace(-1,0,inplace=True)
dataset.drop_duplicates(inplace=True)
X = dataset.drop(['label','Unnamed: 0'],axis=1)
y = dataset.label
x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.1,  random_state = 33)

dataset2 = pd.read_csv(data_dir + 'o2o_test.csv')
dataset2.drop_duplicates(inplace=True)
x_validate = dataset2.drop(['Unnamed: 0'],axis=1)

lgb_test = x_validate

x_validate = xgb.DMatrix(x_validate)
xgbl = xgb.Booster(model_file='xgbmodel')
xgb_pred = xgbl.predict(x_validate)
xgb_pred = MinMaxScaler(copy=True,
                        feature_range=(0,1)).fit_transform(xgb_pred.reshape(-1,1))
xgb_pred = xgb_pred.reshape(1,-1)[0]

gbm = lgb.Booster(model_file='GBMmodel')
lgb_pred = gbm.predict(lgb_test, num_iteration=gbm.best_iteration)
lgb_pred = MinMaxScaler(copy=True,
                        feature_range=(0,1)).fit_transform(lgb_pred.reshape(-1,1))
lgb_pred = lgb_pred.reshape(1,-1)[0]

## tree methods
input2=np.loadtxt(data_dir+"o2o_test.csv",delimiter=',',skiprows = 1)

rf = pickle.load(open('rf.pkl', 'rb'))
pred_of_test=rf.predict(input2[:,1:8])

#predict only rf
pred_of_test[pred_of_test<0]=0
pred_of_test[pred_of_test>=1]=1

prob = 0.4*xgb_pred + 0.45*lgb_pred + 0.15*pred_of_test
pred_result = pd.read_csv(data_dir+'ccf_offline_stage1_test_revised.csv')
pred_result.columns = ['user_id','merchant_id','coupon_id',
                'discount_rate','distance','date_received']
pred_result = pred_result.drop(['merchant_id','discount_rate','distance'],axis=1)
pred_result['prob'] = prob
pred_result.to_csv(data_dir+'result.csv',index=False)
