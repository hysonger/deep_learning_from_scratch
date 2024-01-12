import cupy as np
import dltools, layers, optimizer, dataset.mnist, pickle, datetime

# 神功大成！相比书上版本，还加入了Affine的L2权值衰减和BatchNorm层
class ConvNet(layers.Net):
    def __init__(self):
        super().__init__()

        self.weight_decay_lambda = 1e-6 # 权值衰减系数不宜取得太大

        # 第一层Conv大概像下面这样
        # C, H, W = 1, 28, 28
        # FN, C, FH, FW = 30, 1, 3, 3
        # stride, padding = 1, 0

        #OH = (H - FH + 2 * padding) // stride + 1
        #OW = (W - FW + 2 * padding) // stride + 1

        #P_OH = OH // 2
        #P_OW = OW // 2

        self.layers = [
            layers.Convolution(16, 1, 3, 3, optimizer.Adam, stride=1, padding=1), 
            layers.ReLU(),
            layers.Convolution(16, 16, 3, 3, optimizer.Adam, stride=1, padding=1), # 注意，卷积层会造成通道数变化，每一层的C应等于上一层的FN
            layers.ReLU(),
            layers.Pooling(2, 2, stride=2),
            layers.Convolution(32, 16, 3, 3, optimizer.Adam, stride=1, padding=1), 
            layers.ReLU(),
            layers.Convolution(32, 32, 3, 3, optimizer.Adam, stride=1, padding=2),
            layers.ReLU(),
            layers.Pooling(2, 2, stride=2),
            layers.Convolution(64, 32, 3, 3, optimizer.Adam, stride=1, padding=1), 
            layers.ReLU(),
            layers.Convolution(64, 64, 3, 3, optimizer.Adam, stride=1, padding=1),
            layers.ReLU(),
            layers.Pooling(2, 2, stride=2),
            layers.Affine(64 * 4 * 4, 50, optimizer.Adam), # 从最后一级池化层流出的数据二维化后：FN * P_OH * P_OW
            layers.BatchNormalization(optimizer.Adam),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Affine(50, 10, optimizer.Adam),
            layers.Dropout(0.5)
        ]
        self.layer_loss = layers.SoftmaxWithLoss()

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = dataset.mnist.load_mnist(one_hot_label=True, flatten=False)

    net = ConvNet()

    epoch = 20
    batch_size = 100
    train_size = x_train.shape[0]
    iter_num = epoch * (train_size // batch_size)

    start_time = datetime.datetime.now()
    
    for i in range(iter_num):
        # 随机取样
        batch_choice = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_choice]
        t_batch = t_train[batch_choice]

        loss = net.loss(x_batch, t_batch, True)
        net.gradient()
        net.learning()

        # 基准：第一个epoch后模型就应达到97%准确率，否则模型实现存在问题
        #print(loss, dltools.accuracy_once(net, x_train, t_train), dltools.accuracy_once(net, x_test, t_test))
        print(loss)

        if (i + 1) % (train_size // batch_size) == 0: # 每epoch计算一次测试集准确率
            print("### (%d / %d) Accuracy:" % (i + 1, iter_num), dltools.accuracy(net, x_test, t_test, 200))

    # 训练时间大概接近半个小时（RTX 2060 6G Laptop）
    end_time = datetime.datetime.now()
    print("Total time:", end_time - start_time)

    
    # 这个文件会有200MB+，我自己跑出来99.48%的成绩
    with open("./convnet.pkl", "wb") as f:
        pickle.dump(net, f)

    #with open("./convnet.pkl", "rb") as f:
    #    net = pickle.load(f)

    #print("Prediction accuracy:", dltools.accuracy(net, x_test, t_test))