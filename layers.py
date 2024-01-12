import cupy as np
import dltools, optimizer



class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        raise Exception("I should be covered and never be executed")

    def backward(self, dout): # dout是后一层反向传回的梯度(dL/dy)
        raise Exception("I should be covered and never be executed")

    def update_params(self): # 更新层参数。不需要更新参数的层不用覆盖该方法
        pass

# 一点注释：
# BatchNormalization 和 Dropout 正向传播时需要区分是否是训练状态
# Affine BatchNorm Convolution 三者有需要不断根据梯度更新的参数
# Affine 反向传播时需要额外传入权值衰减计算函数（有权值衰减时）
# 在输入x为非二维矩阵时，Affine和BatchNorm需要额外做处理，不同在于，Affine输出仍为二维，而BatchNorm输出也需要变回原始形状

class Net:
    def __init__(self):
        #self.y = None
        self.weight_decay_lambda = 0 # 设为0即不使用权值衰减

        # 继承该类的类都需要自行声明这两个变量
        #self.layers = []
        #self.layer_loss = SoftmaxWithLoss()

    def predict(self, x, train_flag = False): # 仅推理
        for i, l in enumerate(self.layers):
            if isinstance(l, (BatchNormalization, Dropout)):
                x = self.layers[i].forward(x, train_flag)
            else:
                x = self.layers[i].forward(x)

        return x

    def loss(self, x, t, train_flag = False): # 推理，并调用损失层计算损失
        # 计算加到损失函数上的权值衰减值（即：所有层权值的平方和）
        weight_decay_totalnum = 0
        for i, l in enumerate(self.layers):
            if isinstance(l, Affine):
                weight_decay_totalnum += 0.5 * self.weight_decay_lambda * np.sum(self.layers[i].W ** 2)

        return self.layer_loss.forward(self.predict(x, train_flag), t) + weight_decay_totalnum

    def gradient(self):
        #loss = self.loss(x, t, True) # 正向计算一遍

        dout = self.layer_loss.backward(114514)

        for l in reversed(self.layers):
            if isinstance(l, Affine):
                # 权值衰减时，反向传播额外做一次减法（注意这里不是求和！）
                dout = l.backward(dout, lambda W: self.weight_decay_lambda * W) 
            else:
                dout = l.backward(dout)

        #return loss

    def learning(self):
        #loss = self.gradient(x, t)

        for i, l in enumerate(self.layers):
            self.layers[i].update_params()
        
        #return loss

class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0) # ReLU层记录的是“开关”状态，有大于0流过就置为开
        return x * self.mask

    def backward(self, dout):
        dx = dout * self.mask
        return dx

class Sigmoid(Layer):
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = dltools.sigmoid(x)
        return self.y
    
    def backward(self, dout):
        return dout * (self.y * (1.0 - self.y))

class Affine(Layer):
    def __init__(self, shape_x, shape_y, optimizer, **optimizer_params):
        # 平平无奇的初始值设置（模型过于复杂时，这种初始权值过大，会导致学习难以进行，不信可以配合大学习率试试，试试就逝世）
        #self.W = 0.01 * np.random.randn(shape_x, shape_y)

        # 简化Xavier初始值（只考虑上一层），适合配Sigmoid
        #self.W = np.random.randn(shape_x, shape_y) / np.sqrt(shape_x) 

        # 简化He初始值，适合配ReLU
        self.W = np.sqrt(2.0 / shape_x) * np.random.randn(shape_x, shape_y)

        self.b = np.zeros(shape_y) # b用全0初始化就可以了

        self.x = None

        self.op_W = optimizer(**optimizer_params)
        self.op_b = optimizer(**optimizer_params)

    def forward(self, x):
        # 这部分是为应对高维矩阵（如卷积的4维）而加入
        self.x_original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(self.x_original_shape[0], -1) # (N, -1)


        self.x = x # 输入的x需要记录下来以备反向传播

        # 注意是x・W，x前W后；无论输入是四维还是二维，最终输出都是二维
        return np.dot(self.x, self.W) + self.b 
    
    def backward(self, dout, weight_decay_fx):
        """注意，Affine层的backward需要额外提供一个weight_decay_fx用于权值衰减时梯度反向传播的计算"""
        self.dW = np.dot(self.x.T, dout) + weight_decay_fx(self.W)
        dx = np.dot(dout, self.W.T) # x不是层的参数，不需要保留梯度以用于反向传播学习
        self.db = np.sum(dout, axis=0)

        return dx.reshape(*self.x_original_shape) # 还原为输入端矩阵形状

    def update_params(self): 
        self.W = self.op_W.optimize(self.W, self.dW)
        self.b = self.op_b.optimize(self.b, self.db)


class SoftmaxWithLoss(Layer):
    def __init__(self):
        pass

    def forward(self, x, t):
        self.y = dltools.softmax(x)
        self.t = t
        return dltools.cross_entropy(self.y, self.t) # 返回的是损失(loss)
    
    def backward(self, dout):
        # 因为传入的是一整批(batch)数据，故误差反向传播也需要除以batch_size，得到“平均损失”
        return (self.y - self.t) / self.t.shape[0] 


class BatchNormalization(Layer):
    def __init__(self, optimizer, **optimizer_params):
        self.op_gamma = optimizer(**optimizer_params)
        self.op_beta = optimizer(**optimizer_params)
        self.op_mean = optimizer(**optimizer_params)
        self.op_var = optimizer(**optimizer_params)

        # 这四个参数是本层需要学习的参数
        self.gamma = None
        self.beta = None

        # running两个为不断随新输入数据更新的全体数据平均值，在正向传播时即完成更新
        self.running_mean = None
        self.running_var = None

    def forward(self, x, train_flag):
        # 这里是为应对高维矩阵（如卷积的4维）而加入
        self.x_original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(self.x_original_shape[0], -1) # (N, -1)


        if self.gamma is None:
            self.gamma = np.ones(x.shape[-1])

        if self.beta is None:
            self.beta = np.zeros(x.shape[-1])

        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[-1])

        if self.running_var is None:
            self.running_var = np.zeros(x.shape[-1])

        if train_flag: # 注意！！！本层的正向传播要区分训练和非训练状态，非训练状态下，不应修改层参数
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            # 用于反向传播
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # 不断根据输入batch的平均值和方差，更新整体平均值和方差，该法名为：指数加权移动平均
            self.running_mean = 0.9 * self.running_mean + 0.1 * mu 
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        
        out = self.gamma * xn + self.beta

        # 把形状还原（如col的四维）
        return out.reshape(*self.x_original_shape)

    def backward(self, dout):
        # 这里是为应对高维矩阵（如卷积的4维）而加入
        if dout.ndim > 2:
            dout = dout.reshape(dout.shape[0], -1)


        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.xc.shape[0]) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.xc.shape[0]
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        # 把形状还原（如col的四维）
        return dx.reshape(*self.x_original_shape)

    def update_params(self): # 这里只更新这俩参数
        self.gamma = self.op_gamma.optimize(self.gamma, self.dgamma)
        self.beta = self.op_gamma.optimize(self.beta, self.dbeta)

class Dropout(Layer):
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

# 涉及到卷积部分，用类似书上的双大写字母法进行变量表述。
# 如FH即filter_H，OH即output_H，PH即pool_H
# 正、反向传播配有矩阵形状变化，帮助理解

class Convolution(Layer):
    def __init__(self, filter_N, C, filter_H, filter_W, optimizer, stride = 1, padding = 0, **optimizer_params):
        super().__init__()

        self.stride = stride
        self.padding = padding

        # He初始值，注意这里计算用到的n = C * FH * FW
        self.filter = np.sqrt(2.0 / (C * filter_H * filter_W)) * np.random.randn(filter_N, C, filter_H, filter_W)
        #self.filter = np.random.randn(filter_N, C, filter_H, filter_W) * 0.01

        self.b = np.zeros(filter_N)

        self.op_filter = optimizer(**optimizer_params)
        self.op_b = optimizer(**optimizer_params)
        
    def forward(self, x):
        FN, C, FH, FW = self.filter.shape
        N, C, H, W = x.shape
        
        OH = 1 + (H + 2 * self.padding - FH) // self.stride
        OW = 1 + (W + 2 * self.padding - FW) // self.stride

        self.x_shape = x.shape

        # (N, C, H, W) => (N * OH * OW, C * FH * FW)
        self.col = dltools.im2col(x, FH, FW, self.stride, self.padding)

        # (FN, C, FH, FW) => (C * FH * FW, FN)
        self.col_filter = self.filter.reshape(FN, -1).T

        # (N * OH * OW, C * FH * FW) dot (C * FH * FW, FN) = (N * OH * OW, FN) 
        out = np.dot(self.col, self.col_filter) + self.b

        # (N * OH * OW, FN) => (N, FN, OH, OW)
        return out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        FN, C, FH, FW = self.filter.shape

        # (N, FN, OH, OW) => (N * OH * OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        # (C * FH * FW, N * OH * OW) dot (N * OH * OW, FN)
        self.dfilter = np.dot(self.col.T, dout).reshape(C, FH, FW, FN).transpose(3, 0, 1, 2)

        # (N * OH * OW, FN) dot (FN, C * FH * FW)
        dcol = np.dot(dout, self.col_filter.T)

        # (N * OH * OW, C * FH * FW) => (N, C, H, W)
        return dltools.col2im(dcol, self.x_shape, FH, FW, self.stride, self.padding)

    def update_params(self):
        self.filter = self.op_filter.optimize(self.filter, self.dfilter)
        self.b = self.op_b.optimize(self.b, self.db)

class Pooling(Layer):
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = 1 + (H - self.pool_h) // self.stride
        out_w = 1 + (W - self.pool_w) // self.stride

        col = dltools.im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = dltools.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
        
        return dx

