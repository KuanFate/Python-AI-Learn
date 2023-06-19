import numpy as np


# one dimension picture
def f1(x):
    return 0.5 * (x - 0.25) ** 2


# derived function
def h1(x):
    return 0.5 * 2 * (x - 0.25)


# gradient descent
GD_X = []
GD_Y = []
x = 0  # initial
alpha = 1  # initial step size
f_change = f1(x)
f_current = f_change
GD_X.append(x)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 50:  # condition of stopping iteration
    iter_num += 1
    x = x - alpha * h1(x)
    tmp = f1(x)
    f_change = np.abs(f_current - tmp) # The derivative approaches 0
    f_current = tmp
    GD_X.append(x)
    GD_Y.append(f_current)
print("final result:(%.5f, %.5f)" % (x, f_current))
print("numbers of interation:%d" % iter_num)
print(GD_X)
# build data
X = np.arange(-8,)

