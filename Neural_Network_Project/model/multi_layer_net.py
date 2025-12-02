import numpy as np
from collections import OrderedDict
from common.layers import *
from common.functions import softmax, cross_entropy_error

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu'):
        self.params = {}
        self.layers = OrderedDict()

        all_size_list = [input_size] + hidden_size_list + [output_size]
        for idx in range(1, len(all_size_list)):
            scale = np.sqrt(2.0 / all_size_list[idx - 1]) if weight_init_std == 'relu'                     else np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        for idx in range(1, len(all_size_list) - 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation' + str(idx)] = Relu()

        idx = len(all_size_list) - 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for key in self.params.keys():
            grads[key] = self.layers['Affine' + key[1:]].dW if key[0] == 'W'                          else self.layers['Affine' + key[1:]].db
        return grads
