import numpy as np

a = np.array([[1,1,1],[1,0,1],[0,1,0]])
print(a)

nonzero = a.nonzero()
print(nonzero)

nonzero_1 = np.array(nonzero[0])
print(nonzero_1)

nonzero_2 = np.array(nonzero[1])
print(nonzero_2)

