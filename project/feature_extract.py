# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:53:44 2019
Last modified on 2020/1/3

automatical feature engineering with FeatureTools
@author: Jiafei Lyu
@Department: Engineering Physics
@Theme: ML final project feature engineering

feature engineering by hand
@author: Zhenyao Yan
@Department: School of materials

do feature engineering to split datasets again

we would utilize extract_feature.py to split the task into several parts
1. merchant related
2. user related
3. coupon related
4. user merchant combined related
5. other features

while here, we would utilize featuretools to do feature engineering automatically first
"""

import pandas as pd
import numpy as np
from datetime import date
import featuretools as ft
import os

## manual feature engineering
# info of user taking coupons
# this function refers to feature engineering in Github
# URL: https://github.com/wepe/O2O-Coupon-Usage-Forecast
def other_feature(data):
    dataset=data.copy(deep=True)
    
    # how many coupons a user takes
    t = dataset[['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    
    # how many same coupons a user takes
    t1 = dataset[['user_id','coupon_id']]
    t1['user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    # how many coupons a user takes on someday
    t2 = dataset[['user_id','date_received']]
    t2['this_day_user_receive_all_coupon_count'] = 1
    t2 = t2.groupby(['user_id','date_received']).agg('sum').reset_index()

    # how may identical coupons a user takes in the same day
    t3 = dataset[['user_id','coupon_id','date_received']]
    t3['this_day_user_receive_same_coupon_count'] = 1
    t3 = t3.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()
 
    other_feature = pd.merge(t,t1,on='user_id',how='left')
    other_feature = pd.merge(dataset,other_feature,on=['user_id','coupon_id'],how='left')
    other_feature = pd.merge(other_feature,t2,on=['user_id','date_received'],how='left')
    other_feature = pd.merge(other_feature,t3,on=['user_id','coupon_id','date_received'],how='left')
    
    return other_feature

# calculate discount rate
def calc_discount_rate(s):
    s =str(s)
    if s == '0':
        return 1
    elif ':' in s:
        s = s.split(':')
        return 1.0-float(s[1])/float(s[0])
    else :
        return float(s)

# split discount rate and returns full price you have to reach
def get_discount_man(s):
    s =str(s)
    if ":" in s:
        s = s.split(':')
        return int(s[0])
    else:
        return 0
        
# get how many discount you can get if you reach certain price
def get_discount_jian(s):
    s =str(s)
    if ":" in s:
        s = s.split(':')
        return int(s[1])
    else:
        return 0

# test whether the user get discounted or not
def is_man_jian(s):
    s =str(s)
    if s =='0':
        return -1
    elif ":" in s:
        return 1
    else:
        return 0

# get if the day the user use a coupon is weekday
def getWeekday(s):
    if s == '0':
        return s
    else:
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8])).weekday() + 1
 
# get the Month when a user use a coupon   
def getMonthday(s):
    if s == '0':
        return s
    else:
        return int(s[6:8])

# utilize the above functions and get features by hand
def prepare(predataset):
    dataset=predataset.copy(deep=True)
    dataset['discount_man'] =dataset.discount_rate.apply(get_discount_man)
    dataset['discount_jian'] = dataset.discount_rate.apply(get_discount_jian)
    dataset['is_man_jian'] = dataset.discount_rate.apply(is_man_jian)
    dataset['discount_rate'] = dataset.discount_rate.apply(calc_discount_rate)
    
    dataset['distance']=dataset.distance.replace('0',-1) 
    
    dataset['getWeekday']=dataset['date_received'].astype(str).apply(getWeekday)
    dataset['getMonthday']=dataset['date_received'].astype(str).apply(getMonthday)
    # set it to 1 if weekend, else 0
    dataset['ifWeekend']=dataset['getWeekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    
    return dataset

# calculate discount rate which corresponds with FeatureTools
def cal_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s)==1:
        return round(float(s[0].strip()),2)
    else:
        return round(1.0-float(s[1].strip())/float(s[0].strip()),2)

# get label for the training set
def get_label(s):
    s = str(s).strip()
    s = s.split(':')
    if len(s[0]) < 8 or len(s[1]) < 8:
        return 0
    # if less than 15 days, give it label 1, else label -1
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-
          date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1

def main():
    # define data direction, one can change the direction depending on his PC
    data_dir = './'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # the file is large, it consumes time to read the csv
    offline_train = pd.read_csv(data_dir+'ccf_offline_stage1_train.csv',
                                header=None,low_memory=False).drop([0])
    offline_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance',
                             'date_received','date']
    online_train = pd.read_csv(data_dir+'ccf_online_stage1_train.csv',
                               header=None,low_memory=False).drop([0])
    online_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate',
                            'date_received','date']
    offline_test = pd.read_csv(data_dir+'ccf_offline_stage1_test_revised.csv',
                               header=None,low_memory=False).drop([0])
    offline_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

    # check the data
    print(offline_train.head())

    
    offline_train['discount_rate'] = offline_train.discount_rate.apply(cal_discount_rate)
    offline_test['discount_rate'] = offline_test.discount_rate.apply(cal_discount_rate)
    # online_train['discount_rate'] = online_train.discount_rate.apply(cal_discount_rate)
    # fill nan with 0
    offline_train.fillna('0',inplace=True)
    offline_test.fillna('0',inplace=True)
    # combine trainset and test set to avoid repeating the same procedure
    comb = offline_train.drop(['date'],axis = 1).append(offline_test,ignore_index = True)
    # construct indexes for datasets in order to use FeatureTools
    comb['unid'] = comb['user_id'] + comb['coupon_id']
    comb['umid'] = comb['user_id'] + comb['merchant_id']
    comb['mcid'] = comb['merchant_id'] + comb['coupon_id']
    #comb.set_index('unid',inplace=True)
    #comb.index = pd.DatetimeIndex(comb.index)
    # fill in null value
    #print(comb.isnull())
    #comb['coupon_id'].fillna(comb['coupon_id'].mean(),inplace = True)
    #comb['discount_rate'].fillna(comb['discount_rate'].mean(),inplace = True)
    print(comb.shape)

    # perform automatic feature engineering with featuretools
    # construct user table
    user = comb.drop(['coupon_id','discount_rate','date_received','merchant_id','distance','mcid'],axis=1).drop_duplicates(subset=['user_id'],keep='first')
    user['umid'] = pd.to_numeric(user['umid'])
    # construct merchant table
    merchant = comb.drop(['coupon_id','discount_rate','date_received','unid'],axis=1).drop_duplicates(subset=['merchant_id'],keep='first')
    merchant['mcid'] = pd.to_numeric(merchant['mcid'])
    merchant['umid'] = pd.to_numeric(merchant['umid'])
    # construct coupon table
    coupon = comb.drop(['user_id','distance','unid','umid'],axis=1).drop_duplicates(subset=['coupon_id'],keep='first')
    coupon['coupon_id'] = pd.to_numeric(coupon['coupon_id'])
    coupon['mcid'] = pd.to_numeric(coupon['mcid'])
    #use = comb.drop(['user_id','merchant_id','distance','discount_rate'],axis=1).drop_duplicates(subset=['coupon_id'],keep='first')
    #use['coupon_id'] = pd.to_numeric(use['coupon_id'])
    # Make an entityset and add the entity
    es = ft.EntitySet(id = 'offline')
    es.entity_from_dataframe(entity_id = 'user',dataframe = user,
                           make_index=False,index='umid')
    es.entity_from_dataframe(entity_id = 'merchant',dataframe = merchant,
                           make_index=False, index='mcid')
    es.entity_from_dataframe(entity_id = 'coupon',dataframe = coupon,
                         make_index=False, index='coupon_id')
    #es.entity_from_dataframe(entity_id = 'use',dataframe = use,
    #                     make_index = False, index = 'coupon_id')
    # link these 2 entities with user_id
    es.add_relationship(ft.Relationship(es['user']['umid'],es['merchant']['umid']))
    es.add_relationship(ft.Relationship(es['merchant']['mcid'],es['coupon']['mcid']))
    #es.add_relationship(ft.Relationship(es['coupon']['coupon_id'],es['use']['coupon_id']))
    es.normalize_entity(base_entity_id = 'coupon',new_entity_id = 'discount',
                    index = 'merchant_id',additional_variables = ['discount_rate','date_received'])
    # start to generate features with FeatureTools
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'coupon',
                                        max_depth=2)
    # one can see feature_matrix columns, head and shape
    print(feature_matrix.columns)
    print(feature_matrix.head())
    print(feature_matrix.shape)

    # remove features that have less than 2 unique values
    #from featuretools.selection import remove_low_information_features
    #feature_matrix = remove_low_information_features(feature_matrix)

    # get features by hand
    dataset=prepare(comb)
    print(dataset)

    data_feature=other_feature(dataset)
    print(data_feature.shape)
    data_feature['merchant'] = pd.to_numeric(data_feature['merchant_id'])
    
    feature_matrix['merchant_id'] = pd.to_numeric(feature_matrix['merchant_id']).astype('i8')
    feature_matrix = feature_matrix.reindex(range(data_feature.shape[0])).ffill()
    feature_matrix = feature_matrix.drop(['mcid','merchant_id'],axis=1)
    print(feature_matrix.shape)
    feature_matrix = pd.concat([feature_matrix, data_feature],axis=1,join='outer')
    feature_metrix = feature_matrix.reindex(range(data_feature.shape[0]),method='ffill')
    print(feature_matrix.shape)
    print(feature_matrix.head())

    #feature_matrix = pd,DataFrame(feature_matrix)
    print(feature_matrix.isnull().any())
    print(feature_matrix.head())
    #feature_matrix[feature_matrix.isnull()]=0
    #feature_matrix = feature_matrix.values
    #feature_matrix = np.array(feature_matrix).reshape(1,-1)
    print(feature_matrix.shape)
    # do PCA to reduce feature dimension
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale
    # reduce the dimension to 8
    pca = PCA(n_components = 8)
    print(feature_matrix.values)
    feature_matrix = feature_matrix.apply(pd.to_numeric,axis=0)
    print(feature_matrix.dtypes)
    fea_mat = scale(feature_matrix.values)
    pca.fit(fea_mat)
    feature_matrix = pca.fit_transform(fea_mat)
    # print variance ratio of 8 principle components
    print(pca.explained_variance_ratio_)

    # split dataset
    feature_matrix = pd.DataFrame(feature_matrix)
    from sklearn.model_selection import train_test_split
    train = feature_matrix[:len(offline_train.user_id)]
    test = feature_matrix[len(offline_train.user_id):]

    # attach labels for the training set
    offline_train['label'] = (offline_train.date.astype('str') + ':' + \
                            offline_train.date_received.astype('str')).apply(str)
    train['label'] = None
    train.label = offline_train.label.apply(get_label)
    train.replace(np.nan,0,inplace=True)
    train.replace(np.inf,0,inplace=True)
    train.label = train.label.astype('i8')
    # see train set and test set
    print(train.head())
    print(test.head())
    #train = train.replace('null',np.nan)
    train.to_csv(data_dir+'o2o_train.csv')
    test.to_csv(data_dir+'o2o_test.csv')

    X_train, X_validate, y_train, y_validate = train_test_split(train,train.label,test_size=0.3, 
                                                            random_state = 20, shuffle=True)

if __name__ == "__main__":
    main()
