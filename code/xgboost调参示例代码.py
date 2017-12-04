#
#xgboost�������
#��һ����ȷ��ѧϰ���ʺ�tree_based �������ŵĹ�������Ŀ 

from xgboost import XGBClassifier
 xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
 #�ڶ����� max_depth �� min_weight ��������
 #grid_search�ο� :
 #http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html 
 #http://blog.csdn.net/abcjennifer/article/details/23884761 
 
 #��������scoring=��roc_auc��ֻ֧�ֶ����࣬�������Ҫ�޸�scoring(Ĭ��֧�ֶ����)
 
 param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
#param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
from sklearn import svm, grid_search, datasets
from sklearn import grid_search
gsearch1 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=140, max_depth=5,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test1,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_

#��������gamma��������

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(
estimator = XGBClassifier( 
learning_rate =0.1, 
n_estimators=140, 
max_depth=4, 
min_child_weight=6, 
gamma=0, 
subsample=0.8, 
colsample_bytree=0.8, 
objective= 'binary:logistic', 
nthread=4, 
scale_pos_weight=1,
seed=27), 
param_grid = param_test3, 
scoring='roc_auc',
n_jobs=4,
iid=False, 
cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

#���Ĳ�������subsample �� colsample_bytree ����
#ȡ0.6,0.7,0.8,0.9��Ϊ��ʼֵ

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=3,
min_child_weight=4,
gamma=0.1,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test4,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

 #������������ѧϰ����
 
 xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)

 #�����������򻯲�������
 param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=4,
min_child_weight=6,
gamma=0.1,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test6,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
 
#pythonʾ��
import xgboost as xgb
import pandas as pd
#��ȡ����
from sklearn import cross_validation
from sklearn.datasets import load_iris
iris = load_iris()
#�з����ݼ�
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
#���ò���
m_class = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
#ѵ��
m_class.fit(X_train, y_train)
test_21 = m_class.predict(X_test)
print "Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21)
#Ԥ�����
#test_2 = m_class.predict_proba(X_test)
#�鿴AUC���۱�׼
from sklearn import metrics
print "Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21)
##�����������ܼ���
##print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)
#�鿴��Ҫ�̶�
feat_imp = pd.Series(m_class.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
import matplotlib.pyplot as plt
plt.show()
#�ع�
#m_regress = xgb.XGBRegressor(n_estimators=1000,seed=0)
#m_regress.fit(X_train, y_train)
#test_1 = m_regress.predict(X_test)