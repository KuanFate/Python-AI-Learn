import numpy as np

trainData = []
targetData = [18, 90]
k = 5


def knn(trainData, tartgetData, k):
    listDistance = []
    for train in trainData:
        distance = np.sum((np.array(train[:-1]) - np.array(targetData)) ** 2) ** 0.5
        listDistance.append([distance, train[-1]])
    listDistance.sort()

    weights = []
    for i in listDistance[:k]:
        # +0.001防止距离为0
        weight = 1 / (i[0] + 0.001)
        weights.append(weight)
    weightsSum = sum(weights)
    result = 0
    for i in range(0, k):
        weights[i] = weights[i] / weightsSum
        num = weights[i] * listDistance[i][1]
        result += num
    predict = -1 if result < 0 else 1
    print(predict)
