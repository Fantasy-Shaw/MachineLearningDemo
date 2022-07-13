"""
选择iris中的两类数据对应的样本进行分析。
k-折交叉验证（1<k<n-1)可直接根据sklearn.model_selection.cross_val_predict()得到精度、F1值等度量。
留一法稍微复杂一点，这里采用loop实现。
"""
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import LeaveOneOut

'''
1-st. iris data set importing and visualization using seaborn
'''

sns.set(style="white", color_codes=True)
iris = sns.load_dataset("iris")
X = iris.values[50:150, 0:4]
y = iris.values[50:150, 4]
'''
2-nd logistic regression using sklearn
'''

# log-regression lib model
log_model = LogisticRegression()
m = np.shape(X)[0]

# 10-folds CV
y_pred = cross_val_predict(log_model, X, y, cv=10)
print(metrics.accuracy_score(y, y_pred))

# LOOCV

loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X):
    log_model.fit(X[train], y[train])  # fitting
    y_p = log_model.predict(X[test])
    if y_p == y[test]:
        accuracy += 1
print(accuracy / np.shape(X)[0])
