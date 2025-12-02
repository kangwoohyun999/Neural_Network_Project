import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from data.mnist_reader import load_mnist
from model.multi_layer_net import MultiLayerNet
from common.optimizer import Adam

x_train, t_train = load_mnist('../dataset', kind='train')
x_test, t_test = load_mnist('../dataset', kind='t10k')

network = MultiLayerNet(
    input_size=784,
    hidden_size_list=[50, 50],
    output_size=10,
    activation='relu'
)

optimizer = Adam(lr=0.001)

iters_num = 5000
batch_size = 100
train_size = x_train.shape[0]

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % 500 == 0:
        print("iter", i,
              "| train acc:", network.accuracy(x_train[:1000], t_train[:1000]),
              "| test acc:", network.accuracy(x_test[:1000], t_test[:1000]))

print("Final Test Accuracy:", network.accuracy(x_test, t_test))
