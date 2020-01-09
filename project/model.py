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
# Deep Forest
from GCForest import *

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

params={'booster':'gbtree',
        'objective': 'rank:pairwise',
        'eval_metric':'auc',
        'gamma':0.1,
        'min_child_weight':1.1,
        'max_depth':5,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'colsample_bylevel':0.7,
        'eta': 0.01,
        'tree_method':'exact',
        'seed':0,
        'nthread':12
        }
train_data = xgb.DMatrix(x_train, label = y_train)
test_data = xgb.DMatrix(x_test, label = y_test)
x_validate = xgb.DMatrix(x_validate)
watchlist = [(train_data,'train'),(test_data,'test')]
model = xgb.train(params,train_data,num_boost_round=3500)
model.save_model(data_dir + 'xgbmodel')

train_preds = []
test_preds = []
train_preds = model.predict(xgb.DMatrix(x_train))
test_preds = model.predict(xgb.DMatrix(x_test))
#xgb_f = open("auc.txt","w+")
print('train auc: %f'%metrics.roc_auc_score(y_train,train_preds))
print('test auc: %f'%metrics.roc_auc_score(y_test,test_preds))

dataset2_preds = dataset2[['Unnamed: 0']]
dataset2_preds['label'] = model.predict(x_validate)
dataset2_preds.label = MinMaxScaler(copy=True,
                                    feature_range=(0,1)).fit_transform(dataset2_preds.label.values.reshape(-1,1))
print(dataset2_preds.describe())

## LightGBM
#define parameters like XGBoost
params_gbm = {
    'boosting': 'gbdt',
    'application': 'binary',
    'metric':'auc',
    'num_class': 1,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'lambda_l1': 0,
    'lambda_l2': 0.5,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
    }
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

## train model
gbm = lgb.train(params_gbm, lgb_train, num_boost_round=3000,
                valid_sets=lgb_eval, early_stopping_rounds=5)
# save model to file
gbm.save_model(data_dir + 'GBMmodel')
lgb_pred = gbm.predict(lgb_test, num_iteration=gbm.best_iteration)
lgb_pred = MinMaxScaler(copy=True,
                        feature_range=(0,1)).fit_transform(lgb_pred.reshape(-1,1))
lgb_pred = lgb_pred.reshape(1,-1)[0]

## tree methods
input0=np.loadtxt(data_dir+"o2o_train.csv",delimiter=',',skiprows = 1)
input1=input0[0:input0.shape[0]//5*4]
## validation set
input11=input0[input0.shape[0]//5*4:input0.shape[0]]
# test set
input2=np.loadtxt(data_dir+"o2o_test.csv",delimiter=',',skiprows = 1)

#reg tree
reg = tree.DecisionTreeRegressor(max_depth=10,max_features='log2',random_state=0)
reg.fit(input1[:,1:8],input1[:,9].ravel())
true=input11[:,9]
true[true<0.5]=0
true[true>=0.5]=1
pred=reg.predict(input11[:,1:8])
pred[pred<0.5]=0
pred[pred>=0.5]=1
#score=accuracy_score(true,pred)
reg_pred=reg.predict(input2[:,1:8])
# save model
with open('regtree.pkl', 'wb') as f:
    pickle.dump(reg, f)
del reg

#random forest
rf=RandomForestRegressor(max_depth=8,n_estimators=100,max_features="log2",n_jobs=-1,random_state=0)
rf.fit(input1[:,1:8],input1[:,9])
true=input11[:,9]
true[true<0.5]=0
true[true>=0.5]=1
pred=rf.predict(input11[:,1:8])
pred[pred<0.5]=0
pred[pred>=0.5]=1
#score=accuracy_score(true,pred)
pred_of_test=rf.predict(input2[:,1:8])

#predict only rf and limit its value into [0,1]
pred_of_test[pred_of_test<0]=0
pred_of_test[pred_of_test>=1]=1
# save randomforest model
with open('rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
del rf

# deepforest
gc=gcForest(shape_1X=8,window=8,n_jobs=-1)
gc.fit(input0[:,1:8],input0[:,9].ravel())
pred_of_test_gc=gc.predict_proba(input2[:,1:8])
pred_of_test_gc=pred_of_test_gc[:,1]
pred_of_test_gc[pred_of_test_gc<0]=0
pred_of_test_gc[pred_of_test_gc>1]=1
# save deep forest model
with open('deepforest.pkl', 'wb') as f:
    pickle.dump(gc, f)
del gc

##blend two models and save results
prob = 0.4*dataset2_preds.label + 0.2*pred_of_test + 0.4*lgb_pred
pred_result = pd.read_csv(data_dir+'ccf_offline_stage1_test_revised.csv')
pred_result.columns = ['user_id','merchant_id','coupon_id',
                'discount_rate','distance','date_received']
pred_result = pred_result.drop(['merchant_id','discount_rate','distance'],axis=1)
pred_result['prob'] = prob
pred_result.to_csv(data_dir+'result.csv',index=False)
