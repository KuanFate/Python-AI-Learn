import os.path

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score
from sklearn.preprocessing import LabelEncoder

currentWorkDir = os.path.dirname(__file__)
dataFile = os.path.join(currentWorkDir, './data/iris.data')
names = ['x1', 'x2', 'x3', 'x4', 'y']
df = pd.read_csv(dataFile, header=None, names=names, sep=",")


def parse_record(row):
    result = []
    r = zip(names, row)
    for name, value in r:
        if name == 'y':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            elif value == 'Iris-virginica':
                result.append(3)
            else:
                result.append(0)
    return result


# data cleansing
df = df.apply(lambda row: pd.Series(parse_record(row), index=names), axis=1)
df['y'] = df['y'].astype(np.int32)
df.info()
print(df['y'].value_counts())
flag = False

# get x y
x = df[names[0:-1]]
print(x.shape)
y = df[names[-1]]
print(y.shape)
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
print("训练数据x的格式：{}，以及类型：{}".format(x_train.shape, type(x_train)))

# 特征工程 feature engineering
# pass

# build model
knn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='kd_tree')

# train model
knn.fit(x_train, y_train)

# evaluate model
train_predict = knn.predict(x_train)
test_predict = knn.predict(x_test)
print("knn 测试集准确率: {}".format(knn.score(x_test, y_test)))
print("knn 训练集准确率: {}".format(knn.score(x_train, y_train)))

print(accuracy_score(y_true=y_train, y_pred=train_predict))

# save model
import joblib

joblib.dump(knn, "./knn.m")
