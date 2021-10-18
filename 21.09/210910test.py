import numpy as np
import matplotlib.pyplot as plt

A = np.array([[7,2],[-7,5]])
b = np.array([-5,12])

x = np.linalg.solve(A, b)

x1, y1 = x
print(x1, y1)

fig = plt.figure("test")
ax = fig.add_subplot(1,1,1)

# a1 = A[:,0]
# b1 = A[:,1]
# c1 = b

print(A[:,0])
print(A[:,1])

print(zip(A[:,0],A[:,1]))

# for c1,c2,c3 in zip(A[:,0],A[:,1],b):
#     x = np.linspace(-7,7,100)
#     y = (c3 - c1 * x) / c2
#     print("c1 :",c1)
#     print("c2 :", c2)
#     print("c3 :", c3)
#     ax.plot(x, y, color = "black")

# A = np.array([[7,2],[-7,5]]) # ([[c1][c2]])
# b = np.array([-5,12]) # ([c3])

x = np.linspace(-7, 7, 100)
y = (-5 - (7) * x) / 2
Y1 = (12 - (-7) * x) / 5
ax.plot(x, y, color = "black")
ax.plot(x, Y1, color = "black")

ax.plot(x1, y1, 'ro') # X,Y 교차점 빨간 점 표시

ax.axis([-7, 7, -7, 7]) # X,Y 선 길이 지정
ax.set_xticks(range(-7, 7)) # X 좌표값 -7 ~ 7표시 1단위로
ax.set_yticks(range(-7, 7)) # Y 좌표값 -7 ~ 7표시 1단위로

ax.grid() # 격자 표시
ax.set_axisbelow(True)
ax.set_aspect('equal', adjustable='box')

ax.spines['left'].set_position('zero') # Y좌표값 중심이동
ax.spines['bottom'].set_position('zero') # X좌표값 중심이동

ax.spines['right'].set_color('none') # 상단 검은줄 제거
ax.spines['top'].set_color('none') # 우측 검은줄 제거

plt.show() # 디스플레이
