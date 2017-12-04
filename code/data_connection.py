# -*- coding:utf-8 -*-  

#数据表的关联

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

login = pd.read_csv('../data/t_login.csv')
login_test = pd.read_csv('../data/t_login_test.csv')
trade = pd.read_csv('../data/t_trade.csv')
trade_test = pd.read_csv('../data/t_trade_test.csv')

#区分login和trade的time
login.rename(columns={'time':'login_time'},inplace=True)
login_test.rename(columns={'time':'login_time'},inplace=True)
trade.rename(columns={'time':'trade_time'},inplace=True)
trade_test.rename(columns={'time':'trade_time'},inplace=True)

#时间字符串转换为datetime格式
login['login_time'] = pd.to_datetime(login['login_time'])
login_test['login_time'] = pd.to_datetime(login_test['login_time'])
trade['trade_time'] = pd.to_datetime(trade['trade_time'])
trade_test['trade_time'] = pd.to_datetime(trade_test['trade_time'])

#过滤掉timelong小于等于0的登录记录
login = login[login['timelong'] > 0].reset_index(drop=True)
login_test = login_test[login_test['timelong'] > 0].reset_index(drop=True)

#存储简单处理过的数据
login.to_csv('../data/newdata/login.csv', index=False)
login_test.to_csv('../data/newdata/login_test.csv', index=False)
trade.to_csv('../data/newdata/trade.csv', index=False)
trade_test.to_csv('../data/newdata/trade_test.csv', index=False)

#登录表和交易表的训练集连接
trade_temp=pd.DataFrame(columns=login.columns)
for index in trade.index:
    temp=login[(trade.loc[index,'id']==login['id'])&(trade.loc[index,'trade_time']>login['login_time'])&(login['result']==1)]
    temp=temp.sort_index(by='id')[-1:]
    #def top(df, n, column):
    #    return df.sort_index(by=column)[-n:]
    #temp=temp.groupby('id',as_index=False).apply(top,n=1,column='login_time')
    temp.reset_index(drop=True, inplace=True)
    trade_temp = trade_temp.append(temp, ignore_index=True)
    trade_temp.loc[index,'rowkey'] = trade.loc[index, 'rowkey']

trade_temp.drop('id', axis=1, inplace=True)
login_trade = pd.merge(trade, trade_temp, on='rowkey', how='left')
login_trade.to_csv('../data/newdata/login_trade.csv', index=None)
del temp, trade_temp

#登录表和交易表的测试集连接
trade_test_temp=pd.DataFrame(columns=login_test.columns)
for index in trade_test.index:
    temp=login_test[(trade_test.loc[index,'id']==login_test['id'])&(trade_test.loc[index,'trade_time']>login_test['login_time'])&(login_test['result']==1)]
    temp=temp.sort_index(by='id')[-1:]
    #def top(df, n, column):
    #    return df.sort_index(by=column)[-n:]
    #temp=temp.groupby('id',as_index=False).apply(top,n=1,column='login_test_time')
    temp.reset_index(drop=True, inplace=True)
    trade_test_temp = trade_test_temp.append(temp, ignore_index=True)
    trade_test_temp.loc[index,'rowkey'] = trade_test.loc[index, 'rowkey']

trade_test_temp.drop('id', axis=1, inplace=True)
login_trade_test = pd.merge(trade_test, trade_test_temp, on='rowkey', how='left')
login_trade_test.to_csv('../data/newdata/login_trade_test.csv', index=None)
del temp, trade_test_temp

