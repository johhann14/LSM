import numpy as np

A = np.ones(5)
B = np.zeros(5)

L = [2,4]
print(A)
print(B)
A[L] = B[L]

print(A)

B = np.ones(5) * 2
print(A)
print(B)