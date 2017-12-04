# -*- coding:utf-8 -*-  

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

login = pd.read_csv('../data/newdata/login.csv')
login_test = pd.read_csv('../data/newdata/login_test.csv')
trade = pd.read_csv('../data/newdata/trade.csv')
trade_test = pd.read_csv('../data/newdata/trade_test.csv')
login_trade = pd.read_csv('../data/newdata/login_trade.csv')
login_trade_test = pd.read_csv('../data/newdata/login_trade_test.csv')

#过滤掉没有登录信息的交易
login_trade = login_trade.dropna(axis=0, how='any').reset_index(drop=True)
login_trade_test = login_trade_test.dropna(axis=0, how='any').reset_index(drop=True)

# 时间trade_time和login_time

#时间字符串转换为datetime格式
login['login_time'] = pd.to_datetime(login['login_time'])
login_test['login_time'] = pd.to_datetime(login_test['login_time'])
login_trade['login_time'] = pd.to_datetime(login_trade['login_time'])
login_trade_test['login_time'] = pd.to_datetime(login_trade_test['login_time'])
login_trade['trade_time'] = pd.to_datetime(login_trade['trade_time'])
login_trade_test['trade_time'] = pd.to_datetime(login_trade_test['trade_time'])

feature = login_trade
feature_test = login_trade_test

#交易时的hour值
feature['hour'] = feature['trade_time'].map(lambda x : x.hour)
#交易时间与登陆时间的差值
feature['delta_time'] = (feature['trade_time']-feature['login_time']).map(lambda x : x.total_seconds())

#每次交易时间的差
feature['trade_time_sub'] = feature[['id','trade_time']].sort_values(by='trade_time').groupby('id').diff()['trade_time']
feature['trade_time_sub_day'] = feature['trade_time_sub'].map(lambda x : x.total_seconds())
feature['trade_time_sub_day'].fillna(0, inplace=True)
del feature['trade_time_sub']

#测试集
#交易时的hour值
feature_test['hour'] = feature_test['trade_time'].map(lambda x : x.hour)
#交易时间与登陆时间的差值
feature_test['delta_time'] = (feature_test['trade_time']-feature_test['login_time']).map(lambda x : x.total_seconds())
#每次交易时间的差
feature_test['trade_time_sub'] = feature_test[['id','trade_time']].sort_values(by='trade_time').groupby('id').diff()['trade_time']
feature_test['trade_time_sub_day'] = feature_test['trade_time_sub'].map(lambda x : x.total_seconds())
feature_test['trade_time_sub_day'].fillna(0, inplace=True)
del feature_test['trade_time_sub']

## city,device,ip,log_from,type

### city,device,ip是否多次变化（针对用户id）

#得到DataFrame中的无重复的id
def getUserIDFromDataFrame(dataFrame):
    return pd.DataFrame({'id':dataFrame['id'].unique()})

#得到某列的分组不同的个数
def getCountsByColumnName(loginData,IDdata,columnName):
    col_data = login[columnName].groupby(login['id']).nunique().reset_index()
    col_data.rename(columns={columnName : columnName + '_count'}, inplace=True)
    IDdata = pd.merge(IDdata, col_data, on='id', how='left')
    return IDdata

#每个id登录成功记录对所有登录记录所占的比例
def getResultCount(loginData, IDdata):
    col_data = loginData[['id', 'result']].groupby('id').count().reset_index()
    loginData['result_1'] = loginData['result']==1
    col_data2 = loginData[['id', 'result_1']].groupby('id').sum().reset_index()
    col_data['result_rate'] = col_data2['result_1']/col_data['result']
    loginData.drop('result_1',axis=1,inplace=True)
    col_data.drop('result',axis=1,inplace=True)
    del col_data2
    IDdata = pd.merge(IDdata, col_data, on='id', how='left')
    return IDdata

IDdata = getUserIDFromDataFrame(login)
IDdata = getCountsByColumnName(login, IDdata, 'city')
IDdata = getCountsByColumnName(login, IDdata, 'device')
IDdata = getCountsByColumnName(login, IDdata, 'ip')

IDdata = getResultCount(login, IDdata)

feature = pd.merge(feature, IDdata, on='id', how='left')

IDdata = getUserIDFromDataFrame(login_test)
IDdata = getCountsByColumnName(login_test, IDdata, 'city')
IDdata = getCountsByColumnName(login_test, IDdata, 'device')
IDdata = getCountsByColumnName(login_test, IDdata, 'ip')

IDdata = getResultCount(login_test, IDdata)

feature_test = pd.merge(feature_test, IDdata, on='id', how='left')

### 交易表中的city，device，IP，log_from，type是否为登录表中用户最常用的类型

#选取指定列具有最大值的行的函数
def top(df, n, column):
    return df.sort_index(by=column)[-n:]

def get_commontype(loginData, feature, columnName):
    col_data = loginData[['id',columnName,'log_id']].groupby(['id',columnName]).count()
    col_data.reset_index(inplace=True)
    col_data.rename(columns = {columnName : 'commontype_'+columnName}, inplace=True)
    col_data = col_data.groupby('id', as_index=False).apply(top, n=1, column='log_id').reset_index(drop=True)
    col_data.drop('log_id', axis=1, inplace=True)
    feature = pd.merge(feature, col_data, on='id', how='left')
    feature['is_common_'+columnName] = (feature[columnName]==feature['commontype_'+columnName]).astype(int)
    feature.drop('commontype_'+columnName, axis=1, inplace=True)
    return feature

#交易表中的city，device，IP，log_from，type是否为登录表中用户最常用的类型
feature = get_commontype(login, feature, 'city')
feature = get_commontype(login, feature, 'device')
feature = get_commontype(login, feature, 'ip')
feature = get_commontype(login, feature, 'log_from')
feature = get_commontype(login, feature, 'type')

#交易表中的city，device，IP，log_from，type是否为登录表中用户最常用的类型
feature_test = get_commontype(login_test, feature_test, 'city')
feature_test = get_commontype(login_test, feature_test, 'device')
feature_test = get_commontype(login_test, feature_test, 'ip')
feature_test = get_commontype(login_test, feature_test, 'log_from')
feature_test = get_commontype(login_test, feature_test, 'type')

 ### 对log_from,type进行one-hot处理

# 对log_from,type进行one-hot处理
feature = pd.get_dummies(feature, columns=['log_from','type'], prefix=['log_from','type'])
feature_test = pd.get_dummies(feature_test, columns=['log_from','type'], prefix=['log_from','type'])
feature.drop('log_from_18.0', axis=1, inplace=True)
#feature_test中不含log_from_18.0，且feature中只有7个log_from_18.0，故去掉

### 将is_scan,is_sec的bool值改为0/1

# 将is_scan,is_sec的bool值改为0/1
feature[['is_scan', 'is_sec']] = feature[['is_scan', 'is_sec']].astype(int)
feature_test[['is_scan', 'is_sec']] = feature_test[['is_scan', 'is_sec']].astype(int)

### 删除无用的特征

# 删除无用的特征
feature.drop(['log_id', 'login_time', 'result', 'timelong', 'timestamp'], axis=1, inplace=True)
feature_test.drop(['log_id', 'login_time', 'result', 'timelong', 'timestamp'], axis=1, inplace=True)

### 保存特征文件

feature.to_csv('../feature/feature001.csv',index=None)
feature_test.to_csv('../feature/feature_test001.csv',index=None)


