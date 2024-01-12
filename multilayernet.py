import layers, dltools, optimizer, dataset.mnist, pickle
import cupy as np

class MultiLayerNet(layers.Net):
    def __init__(self):
        super().__init__()

        self.layers = [
            layers.Affine(784, 50, optimizer.Momentum, learning_rate=0.01, momentum=0.9),
            #layers.Dropout(0.5),
            layers.BatchNormalization(optimizer.Momentum, learning_rate=0.01, momentum=0.9),
            layers.ReLU(),
            layers.Affine(50, 25, optimizer.Momentum, learning_rate=0.01, momentum=0.9),
            #layers.Dropout(0.5),
            layers.BatchNormalization(optimizer.Momentum, learning_rate=0.01, momentum=0.9),
            layers.ReLU(),
            layers.Affine(25, 10, optimizer.Momentum, learning_rate=0.01, momentum=0.9)
        ]
        self.layer_loss = layers.SoftmaxWithLoss()
        

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = dataset.mnist.load_mnist(one_hot_label=True)

    net = MultiLayerNet()

    iter_num = 10000
    batch_size = 100
    train_size = x_train.shape[0]

    for i in range(iter_num):
        batch_choice = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_choice]
        t_batch = t_train[batch_choice]

        loss = net.loss(x_batch, t_batch, True)
        net.gradient()
        net.learning()

        print(loss)

        if (i + 1) % (train_size // batch_size) == 0: # 每epoch计算一次测试集准确率
            print("### (%d / %d) Accuracy:" % (i + 1, iter_num), dltools.accuracy(net, x_test, t_test, 1000))

    with open("./multilayernet.pkl", "wb") as f:
        pickle.dump(net, f)

    #with open("./multilayernet.pkl", "rb") as f:
    #    net = pickle.load(f)

    print("Test loss is:", dltools.accuracy(net, x_test, t_test, 1000))

