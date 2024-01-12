import cupy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        #self.params = {} # 按算法需要自行添加辅助变量dict

    def optimize(self, param, grad): # return updated param
        pass

            
class SGD(Optimizer):
    def __init__(self, learning_rate = 0.01):
        super().__init__(learning_rate)

    def optimize(self, param, grad):
        param -= self.learning_rate * grad
        return param

class Momentum(Optimizer):
    def __init__(self, learning_rate = 0.01, momentum = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def optimize(self, param, grad):
        if self.v is None: # 必须用is不能用==否则报ValueError
            self.v = np.zeros_like(param)

        self.v = self.momentum * self.v - self.learning_rate * grad
        param += self.v

        return param

class AdaGrad(Optimizer):
    def __init__(self, learning_rate = 0.01):
        super().__init__(learning_rate)
        self.h = None

    def optimize(self, param, grad):
        if self.h is None:
            self.h = np.zeros_like(param)

        self.h += grad ** 2

        param -= self.learning_rate * grad / np.sqrt(self.h + 1e-7)
        return param

        
class Adam(Optimizer):

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def optimize(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        self.m += (1 - self.beta1) * (grad - self.m)
        self.v += (1 - self.beta2) * (grad ** 2 - self.v)
            
        param -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)

        return param
        