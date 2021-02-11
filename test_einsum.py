import numpy as np 


x = np.ones((1,50))
y = np.array([10,20])[None]

print(x.shape, y.shape)

s = x+y.T
print(s.shape)
print(s)