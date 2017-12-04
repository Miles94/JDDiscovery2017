

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

random_seed = 1201

#读入数据
feature_csv = pd.read_csv('../feature/feature001.csv')
feature_test_csv = pd.read_csv('../feature/feature_test001.csv')

#选取训练集和测试集
train_raw = feature_csv.drop(['rowkey', 'trade_time', 'id','city','device','ip'], axis=1)
test = feature_test_csv.drop(['rowkey', 'trade_time', 'id','city','device','ip'], axis=1)
rowkeytest = feature_test_csv['rowkey']

print "neg:{0},pos:{1}".format(len(train_raw[train_raw['is_risk']==0]),len(train_raw[train_raw['is_risk']==1]))

#划分训练集与验证集
train,val = train_test_split(train_raw, test_size = 0.2,random_state=1)
y = train['is_risk']
X = train.drop(['is_risk'],axis=1)
val_y = val['is_risk']
val_X = val.drop(['is_risk'],axis=1)

#自定义评价函数
from sklearn.metrics import confusion_matrix
def customedscore(preds, dtrain):
    label = dtrain.get_label()
    pred = [int(i>=0.5) for i in preds]
    confusion_matrixs = confusion_matrix(label, pred)
    recall =float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][0]+confusion_matrixs[1][1])
    precision = float(confusion_matrixs[1][1]) / float(confusion_matrixs[0][1]+confusion_matrixs[1][1])
    F = (1+0.01)*precision* recall/(0.01*precision+recall)
    return 'FSCORE',float(F)

#xgboost start here
dtest = xgb.DMatrix(test)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds':100,
    'scale_pos_weight': 1,
#    'eval_metric': 'auc',
    'gamma':0.1,#0.2 is ok
    'max_depth':8,
    'lambda':550,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':2.5, 
    'eta': 0.007,
    'seed':random_seed
    }

watchlist  = [(dtrain,'train'),(dval,'val')]#The early stopping is based on last set in the evallist
model = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist, feval=customedscore)
model.save_model('../model/xgb.model')
print "best best_ntree_limit",model.best_ntree_limit   #did not save the best,why?

#predict test set (from the best iteration)
test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["rowkey","score"])
test_result.rowkey = rowkeytest
test_result.score = test_y
test_result.to_csv("../xgb/xgb.csv",index=None,encoding='utf-8')  #remember to edit xgb.csv , add ""

test_result = pd.read_csv("../xgb/xgb.csv")
test_result.score = test_result.score.map(lambda i: int(i>=0.5))

trade_test = pd.read_csv('../data/t_trade_test.csv')
trade = pd.read_csv('../data/t_trade.csv')

xgb001 = pd.merge(trade_test[['rowkey']], test_result, on='rowkey', how='left')
xgb001 = xgb001.fillna(0)
xgb001['score']=xgb001['score'].astype('int')
xgb001.to_csv("../xgb/xgb001.csv",header=False,index=False)
