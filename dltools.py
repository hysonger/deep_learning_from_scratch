# 鱼书练习代码的通用函数库

import cupy as np

def step(x):
    """step function compatible with numpy"""
    y = x > 0
    return y.astype(int) # astype：转换矩阵的元素数据类型（这个方法是矩阵的不是np的）

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    #return np.max() # max是一个数组中的最大值，maximum是比较两数组的更大值，不一样
    return np.maximum(x, 0)

def softmax_worse(x):
    # 他们这个softmax是比较差的（容易溢出）
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def numerical_diff(f, x): # 不好用，别用，仅演示算式
    h = 0.0001
    return (f(x + h) - f(x - h)) / h / 2

def gradient_descent(f, x, step_num, learning_rate):
    """梯度下降法示例，仅演示"""
    for i in range(step_num):
        x -= learning_rate * numerical_diff(f, x)

    return x

# ——————以上无实用价值，仅示例————————

def softmax(x):
    """指数运算很容易溢出，故需要把x做一下减法，计算可知统一减去常数C不影响softmax函数结果"""
    if x.ndim >= 2:
        x = x.T - np.max(x.T, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    else:
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

def mean_squared(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy(y, t):
    """交叉熵损失函数(该版本t需为one-hot表示)"""
    if y.ndim == 1 and t.ndim == 1:
        y = y[np.newaxis, :] # 在“最外层”扩充一个维度
        t = t[np.newaxis, :]


    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

def cross_entropy_no_onehot(y, t):
    if y.ndim == 1:
        y = y[np.newaxis, :]
        #t = t[np.newaxis, :]

    batch_size = y.shape[0]

    # 这里t是一维，每项分别对应一条样本值（y中一条），用高级索引，取y的第arange()条记录的第t个值
    # （相当于one-hot中只有这项系数为1，其他全不要了）
    return -np.sum(np.log(y[np.arange(batch_size), t]) + 1e-7)

def im2col(input_data, FH, FW, stride = 1, padding = 0):
    N, C, H, W = input_data.shape
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant') # 填充(padding)
    col = np.zeros((N, C, OH, OW, FH, FW)) # col的初始形状

    for block_y in range(OH):
        for block_x in range(OW):
            # 从img中取一格filter，赋给col
            col[:, :, block_y, block_x, :, :] = img[:, :, block_y * stride:block_y * stride + FH, block_x * stride:block_x * stride + FW]

    # 对col维度顺序重排，再压缩成二维矩阵，达到“压平”效果
    # 0维为N * OH * OW（总的“小方块”个数），1维为C * FH * FW（每个小方块的大小，也即滤波器方块大小）
    return col.transpose(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, -1)
    """
    # 这是官方代码，作为对比
    N, C, H, W = input_data.shape
    OH = (H + 2*padding - FH)//stride + 1
    OW = (W + 2*padding - FW)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, FH, FW, OH, OW))

    for y in range(FH):
        y_max = y + stride*OH
        for x in range(FW):
            x_max = x + stride*OW
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*OH*OW, -1)
    return col
    """

def col2im(col, input_shape, FH, FW, stride=1, padding=0):
    N, C, H, W = input_shape
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1
    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 1, 2, 4, 5)

    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(OH):
        for x in range(OW):
            # 注意，逆运算这里是+=！若stride<FH或FW，则显然img上每取一格filter必互相有重叠，重叠部分应进行相加
            img[:, :, y * stride:y * stride + FH, x * stride:x * stride + FW] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding] # 去掉周边的填充部分(center crop)
    """
    # 这是官方代码，作为对比
    N, C, H, W = input_shape
    OH = (H + 2*padding - FH)//stride + 1
    OW = (W + 2*padding - FW)//stride + 1
    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*padding + stride - 1, W + 2*padding + stride - 1))
    for y in range(FH):
        y_max = y + stride*OH
        for x in range(FW):
            x_max = x + stride*OW
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]
    """
    
    
def accuracy(net, x, t, batch_size=100):
    """方便地计算accuracy"""
    accuracy = 0

    for i in range(0, x.shape[0], batch_size):
        y_argmax = softmax(net.predict(x[i:i + batch_size], False)).argmax(axis = 1)
        t_argmax = t[i:i + batch_size].argmax(axis = 1)
        accuracy += np.sum(y_argmax == t_argmax)

    return accuracy / len(x)

def accuracy_once(net, x, t, batch_size=100):
    """方便地计算accuracy（仅随机取算一个batch的版本）"""
    choice = np.random.choice(len(x), batch_size, replace=False)

    y_argmax = softmax(net.predict(x[choice], False)).argmax(axis = 1)
    t_argmax = t[choice].argmax(axis = 1)

    return np.sum(y_argmax == t_argmax) / batch_size

if __name__ == "__main__":
    print(step(np.array([6, -7])))