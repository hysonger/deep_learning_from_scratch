import cupy as np
import dltools, layers, optimizer, dataset.mnist, pickle, datetime

class SimpleConvNet(layers.Net):
    def __init__(self):
        super().__init__()


        self.layers = [
            layers.Convolution(30, 1, 5, 5, optimizer.Momentum, stride=1, padding=0), # OH = OW = 24, P_OH = P_OW = 12
            layers.ReLU(),
            layers.Pooling(2, 2, stride=2),
            layers.Affine(30 * 12 * 12, 100, optimizer.Momentum), # 从最后一级池化层流出的数据二维化后：FN * P_OH * P_OW
            layers.ReLU(),
            layers.Affine(100, 10, optimizer.Momentum),
        ]
        self.layer_loss = layers.SoftmaxWithLoss()

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = dataset.mnist.load_mnist(one_hot_label=True, flatten=False)

    net = SimpleConvNet()

    epoch = 20
    batch_size = 100
    train_size = x_train.shape[0]
    iter_num = epoch * (train_size // batch_size)

    start_time = datetime.datetime.now()
    
    for i in range(iter_num):
        batch_choice = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_choice]
        t_batch = t_train[batch_choice]

        loss = net.loss(x_batch, t_batch, True)
        net.gradient()
        net.learning()

        #print(loss, dltools.accuracy_once(net, x_train, t_train), dltools.accuracy_once(net, x_test, t_test))
        print(loss)

        if (i + 1) % (train_size // batch_size) == 0: # 每epoch计算一次测试集准确率
            print("### (%d / %d) Accuracy:" % (i + 1, iter_num), dltools.accuracy(net, x_test, t_test, 200))

    end_time = datetime.datetime.now()
    print("Total time:", end_time - start_time)

    
    with open("./simpleconvnet.pkl", "wb") as f:
        pickle.dump(net, f)

    #with open("./convnet.pkl", "rb") as f:
    #    net = pickle.load(f)

    #print("Prediction accuracy:", dltools.accuracy(net, x_test, t_test))