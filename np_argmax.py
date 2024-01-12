import numpy as np

a = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]

print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))
print(np.argmax(a, axis=2))

# 这是一个2x2x3的矩阵，2 2 3依次等于从最外到最内的数组成员个数
# axis指派的是，在(2, 2, 3)中的第axis位，把：对应的“立体”边长所指那一条方向上的各项，作为一个计算“最大值在其中位次”的基本单位