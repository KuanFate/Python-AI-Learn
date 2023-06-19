import sys
import matplotlib.pylab as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Prevent Chinese garbled characters
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
'''
这段代码是通过NumPy库中的array函数创建一个一维数组x1，并将该数组reshape为一个二维数组，其中第一个维度自动计算得到，而第二个维度为1。数组x1的元素包括10、15、20、30、50、60、60和70。
需要注意的是，数组x1的reshape操作相当于将原数组沿着行方向重新排列成了一个n行1列的二维数组，其中n为x1数组的长度，也就是8。这里使用了-1作为reshape函数的参数，表示程序将自动根据数组x1的长度来确定第一个维度的大小。因此，这段代码等价于x1 = np.array([[10],[15],[20],[30],[50],[60],[60],[70]])。
这种数组的形式在矩阵计算中比较常见，因为它可以表示一个n行1列的列向量。'''
X1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))
Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

# 添加一个截距项对应的X值
flag = True

# 不加截距项
if flag:
    X = np.column_stack((np.ones_like(X1), X1))
else:
    X = X1
# 为了方便求解，将Numpy的ndarray的数据类型转换为矩阵的形式
X = np.mat(X)
Y = np.mat(Y)

theta = (X.T * X).I * X.T * Y

predict_y = X * theta
print(predict_y.shape)

mse = mean_squared_error(y_true=np.asarray(Y), y_pred=np.asarray(predict_y))
print("MSE", mse)

r2 = r2_score(y_true=np.asarray(Y), y_pred=np.asarray(predict_y))
print("r^2", r2)
print("theta",theta,theta.shape)
if flag:
    # 1代表x0
    x_test = [[1,55], [1,60]]
else:
    x_test = [[55],[60]]
y_test_hat = x_test * theta
print("价格: ", y_test_hat)

'''第一条折线图用蓝色圆形代表真实值，第二条折线图用红色虚线圆形代表预测值。其中，X1代表横坐标值，Y代表第一条折线图的纵坐标值，predict_y代表第二条折线图的纵坐标值。
legend()函数用来添加图例，即用于标识不同图像代表的内容的文本。所以，'真实值'和'预测值'是每条线对应的文本标签，'lower right'代表将图例放置在图像的右下角位置。最后，show()函数用于显示图像。'''
plt.plot(X1, Y, 'bo', label=u'真实值')
plt.plot(X1, predict_y, 'r--o', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
