import numpy as np

a = np.ones((4,1))
b = np.array([8])
print(a.shape, b.shape)
print(a@b)

c = np.array([2,2,3,2])
print(c)
print(c[:, None].shape)

L = np.array([1,2,3,4,5,6])

i = ((L<3) & (L>1))
print(i)
print(np.any([0,1,0]))