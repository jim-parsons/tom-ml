import numpy as np

# ---------------exp和sqrt--------------------
B = np.arange(3)
print(B)  # [0 1 2]
print(np.exp(B))  # [ 1.   2.71828183  7.3890561 ]
print(np.sqrt(B))  # [ 0.   1.          1.41421356]

print('========1=========')
# ---------------------floor向下取整---------------------
a = np.floor(10 * np.random.random((3, 4)))
# ----------------ravel把矩阵拉长-----------------------
print(a.ravel())
print('========2=========')
# ---------shape定义矩阵的行和宽，与resize用法相同--------
a.shape = (6, 2)
# ----------------矩阵转置------------------------------
print(a.T)
print(a.resize((2, 6)))
print(a)
print('========3=========')

# ----------矩阵拼接hstack水平拼接，vstack竖直拼接--------------
a = np.floor(10 * np.random.random((2, 2)))
b = np.floor(10 * np.random.random((2, 2)))

print(np.hstack((a, b)))
print(np.vstack((a, b)))

print('========4=========')
# ----------矩阵切分hsplit水平切分，vsplit竖直切分--------------
a = np.floor(10 * np.random.random((2, 12)))
print(np.hsplit(a, 3))  # 把矩阵a水平均匀切分3等分
print(np.hsplit(a, (3, 4)))  # 在矩阵a的第3列和第4列后边切分
print('=========5========')
a = np.floor(10 * np.random.random((12, 2)))
print(np.vsplit(a, 3))  # 把矩阵a数值均匀切分3等分

# -------------等号赋值a和b其实是一回事，对a进行的任何操作，b也会跟着改变---------
a = np.arange(12)
b = a  # =号赋值后，a和b属于同一块区域，对a进行操作，b也发生变化

# --------------view共享数据，对a进行除改数据之外任何操作，对c都没影响（浅复制）-------------
c = a.view()

# --------------------------深复制，a和d完全独立-------------------------------
d = a.copy()







