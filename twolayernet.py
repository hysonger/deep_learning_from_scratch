import dltools, layers, optimizer, dataset.mnist
#import common.layers, two_layer_net
import cupy as np

class TwoLayerNet(layers.Net):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layers = [
            layers.Affine(input_size, hidden_size, optimizer.Momentum, learning_rate=0.1, momentum=0.9),
            #layers.BatchNormalization(optimizer.Momentum(0.1), optimizer.Momentum(0.1)),
            layers.ReLU(),
            layers.Affine(hidden_size, output_size, optimizer.Momentum, learning_rate=0.1, momentum=0.9)
        ]

        self.layer_loss = layers.SoftmaxWithLoss()
        

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = dataset.mnist.load_mnist(one_hot_label=True)

    net = TwoLayerNet(784, 50, 10)

    iter_num = 10000
    batch_size = 100
    train_size = x_train.shape[0]

    for i in range(iter_num):
        # 从全体训练集中随机选取一个batch的数据
        batch_choice = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_choice]
        t_batch = t_train[batch_choice]

        loss = net.loss(x_batch, t_batch, True)
        net.gradient()
        net.learning()

        print(loss)
        if (i + 1) % (train_size // batch_size) == 0: # 每epoch计算一次测试集准确率
            print("### (%d / %d) Accuracy:" % (i + 1, iter_num), dltools.accuracy(net, x_test, t_test, 1000))
    
    with open("./twolayernet.pkl", "wb") as f:
        pickle.dump(net, f)

    #with open("./twolayernet.pkl", "rb") as f:
    #    net = pickle.load(f)


    print("Test loss is:", dltools.accuracy(net, x_test, t_test, 1000))

