import numpy as n
import  timeit as t

# a = n.arange(1,12+1)
# print(a,a.shape)
#
# b = a[n.newaxis,:]
# print(b,b.shape)
#
# c = a[:,n.newaxis]
# # c = c.astype(np.float64)
# print(c,c.shape,c.dtype)
#
# d = n.ones(c.shape)
# d = d.astype(n.int32)
# print(d,d.shape,d.dtype)
#
# e = n.hstack((d,c))
# print("e",e,e.shape,e.dtype)
#
# print("e",e[1])
#
# C = n.array([[1,2,3,4],[5,6,7,8]])
#
# print('%s %s'%("1행만 출력",C[0:1,:]))
# # c = c[n.newaxis,:][0]
# # print(c,c.shape)
#
# # print(a == b)
# # print(a is b)

A = n.array([[1,2,3],[1,2,3]])
# B = n.array([[1,2,3],[1,2,3]])

print("A",A.shape)
# print("B",B.shape)

# C = n.dot(A,B)
# print(C)

B = n.array([[1,2,3],[1,2,3],[1,2,3]])
print("B",B.shape)

C = n.dot(A,B)

print(C)

n.matrix_rank(1,2,3)