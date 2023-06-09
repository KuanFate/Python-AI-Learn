import numpy as np

a = np.array([2, 3, 4, 5, 6]) # (5,) 1
print(a.shape,a.ndim)
b = np.array([[2, 3, 4, 5, 6]]) # (1,5) 2
print(b.shape,b.ndim)
c = np.array([[2, 3, 4, 5, 6],[2, 3, 4, 5, 6]]) # (2, 5) 2
print(c.shape,c.ndim)